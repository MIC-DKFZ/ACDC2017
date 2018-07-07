# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import matplotlib
matplotlib.use('Agg')
import lasagne
import theano.tensor as T
import numpy as np
import theano
import os
import cPickle
from utils import plotProgress
import time
from utils import soft_dice, hard_dice
from BatchGenerator import BatchGenerator_2D
from dataset_utils import load_dataset
from utils import get_split
import imp
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import Compose, RndTransform
from batchgenerators.transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms import GammaTransform, ConvertSegToOnehotTransform
from batchgenerators.transforms import RandomCropTransform


def create_data_gen_train(patient_data_train, BATCH_SIZE, num_classes,
                                  num_workers=5, num_cached_per_worker=2,
                                  do_elastic_transform=False, alpha=(0., 1300.), sigma=(10., 13.),
                                  do_rotation=False, a_x=(0., 2*np.pi), a_y=(0., 2*np.pi), a_z=(0., 2*np.pi),
                                  do_scale=True, scale_range=(0.75, 1.25), seeds=None):
    if seeds is None:
        seeds = [None]*num_workers
    elif seeds == 'range':
        seeds = range(num_workers)
    else:
        assert len(seeds) == num_workers
    data_gen_train = BatchGenerator_2D(patient_data_train, BATCH_SIZE, num_batches=None, seed=False,
                                       PATCH_SIZE=(352, 352))

    tr_transforms = []
    tr_transforms.append(MirrorTransform((2, 3)))
    tr_transforms.append(RndTransform(SpatialTransform((352, 352), list(np.array((352, 352))//2),
                                                       do_elastic_transform, alpha,
                                                       sigma,
                                                       do_rotation, a_x, a_y,
                                                       a_z,
                                                       do_scale, scale_range, 'constant', 0, 3, 'constant',
                                                       0, 0,
                                                       random_crop=False), prob=0.67,
                                      alternative_transform=RandomCropTransform((352, 352))))
    tr_transforms.append(ConvertSegToOnehotTransform(range(num_classes), seg_channel=0, output_key='seg_onehot'))

    tr_composed = Compose(tr_transforms)
    tr_mt_gen = MultiThreadedAugmenter(data_gen_train, tr_composed, num_workers, num_cached_per_worker, seeds)
    tr_mt_gen.restart()
    return tr_mt_gen


def run(config_file, fold=0):
    cf = imp.load_source('cf', config_file)
    print fold
    dataset_root = cf.dataset_root
    # ==================================================================================================================
    BATCH_SIZE = cf.BATCH_SIZE
    INPUT_PATCH_SIZE = cf.INPUT_PATCH_SIZE
    num_classes = cf.num_classes
    EXPERIMENT_NAME = cf.EXPERIMENT_NAME
    results_dir = os.path.join(cf.results_dir, "fold%d/"%fold)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    n_epochs = cf.n_epochs
    lr_decay = cf.lr_decay
    base_lr = cf.base_lr
    n_batches_per_epoch = cf.n_batches_per_epoch
    n_test_batches = cf.n_test_batches
    n_feedbacks_per_epoch = cf.n_feedbacks_per_epoch
    num_workers = cf.num_workers
    workers_seeds = cf.workers_seeds
    # ==================================================================================================================

    # this is seeded, will be identical each time
    train_keys, test_keys = get_split(fold)

    train_data = load_dataset(train_keys, root_dir=dataset_root)
    val_data = load_dataset(test_keys, root_dir=dataset_root)

    x_sym = cf.x_sym
    seg_sym = cf.seg_sym

    nt, net, seg_layer = cf.nt, cf.net, cf.seg_layer
    output_layer_for_loss = net
    #draw_to_file(lasagne.layers.get_all_layers(net), os.path.join(results_dir, 'network.png'))

    data_gen_validation = BatchGenerator_2D(val_data, BATCH_SIZE, num_batches=None, seed=False,
                                            PATCH_SIZE=INPUT_PATCH_SIZE)
    data_gen_validation = MultiThreadedAugmenter(data_gen_validation,
                                                 ConvertSegToOnehotTransform(range(num_classes), 0, "seg_onehot"),
                                                 1, 2, [0])

    # add some weight decay
    l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss,
                                                               lasagne.regularization.l2) * cf.weight_decay

    # the distinction between prediction_train and test is important only if we enable dropout
    prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False,
                                                 batch_norm_update_averages=False, batch_norm_use_averages=False)

    loss_vec = - soft_dice(prediction_train, seg_sym)

    loss = loss_vec.mean()
    loss += l2_loss
    acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym.argmax(-1)), dtype=theano.config.floatX)

    prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True,
                                                batch_norm_update_averages=False, batch_norm_use_averages=False)
    loss_val = - soft_dice(prediction_test, seg_sym)

    loss_val = loss_val.mean()
    loss_val += l2_loss
    acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym.argmax(-1)), dtype=theano.config.floatX)

    # learning rate has to be a shared variable because we decrease it with every epoch
    params = lasagne.layers.get_all_params(output_layer_for_loss, trainable=True)
    learning_rate = theano.shared(base_lr)
    updates = lasagne.updates.adam(T.grad(loss, params), params, learning_rate=learning_rate, beta1=0.9, beta2=0.999)

    dc = hard_dice(prediction_test, seg_sym.argmax(1), num_classes)

    train_fn = theano.function([x_sym, seg_sym], [loss, acc_train, loss_vec], updates=updates)
    val_fn = theano.function([x_sym, seg_sym], [loss_val, acc, dc])

    dice_scores=None
    data_gen_train = create_data_gen_train(train_data, BATCH_SIZE,
                                           num_classes, num_workers=num_workers,
                                           do_elastic_transform=True, alpha=(100., 350.), sigma=(14., 17.),
                                           do_rotation=True, a_x=(0, 2.*np.pi), a_y=(-0.000001, 0.00001),
                                           a_z=(-0.000001, 0.00001), do_scale=True, scale_range=(0.7, 1.3),
                                           seeds=workers_seeds)  # new se has no brain mask


    all_training_losses = []
    all_validation_losses = []
    all_validation_accuracies = []
    all_training_accuracies = []
    all_val_dice_scores = []
    epoch = 0


    while epoch < n_epochs:
        if epoch == 100:
            data_gen_train = create_data_gen_train(train_data, BATCH_SIZE,
                                                   num_classes, num_workers=num_workers,
                                                   do_elastic_transform=True, alpha=(0., 250.), sigma=(14., 17.),
                                                   do_rotation=True, a_x=(-2 * np.pi, 2 * np.pi),
                                                   a_y=(-0.000001, 0.00001), a_z=(-0.000001, 0.00001),
                                                   do_scale=True, scale_range=(0.75, 1.25),
                                                   seeds=workers_seeds)  # new se has no brain mask
        if epoch == 125:
            data_gen_train = create_data_gen_train(train_data, BATCH_SIZE,
                                                   num_classes, num_workers=num_workers,
                                                   do_elastic_transform=True, alpha=(0., 150.), sigma=(14., 17.),
                                                   do_rotation=True, a_x=(-2 * np.pi, 2 * np.pi),
                                                   a_y=(-0.000001, 0.00001), a_z=(-0.000001, 0.00001),
                                                   do_scale=True, scale_range=(0.8, 1.2),
                                                   seeds=workers_seeds)  # new se has no brain mask
        epoch_start_time = time.time()
        learning_rate.set_value(np.float32(base_lr* lr_decay**epoch))
        print "epoch: ", epoch, " learning rate: ", learning_rate.get_value()
        train_loss = 0
        train_acc_tmp = 0
        train_loss_tmp = 0
        batch_ctr = 0
        for data_dict in data_gen_train:
            data = data_dict["data"].astype(np.float32)
            seg = data_dict["seg_onehot"].astype(np.float32).transpose(0, 2, 3, 1).reshape((-1, num_classes))
            if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)) == 0:
                print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
                print "training_loss since last update: ", \
                    train_loss_tmp/np.floor(n_batches_per_epoch/(n_feedbacks_per_epoch-1)), " train accuracy: ", \
                    train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)
                all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/(n_feedbacks_per_epoch-1)))
                all_training_accuracies.append(train_acc_tmp/np.floor(n_batches_per_epoch/(n_feedbacks_per_epoch-1)))
                train_loss_tmp = 0
                train_acc_tmp = 0
                if len(all_val_dice_scores) > 0:
                    dice_scores = np.concatenate(all_val_dice_scores, axis=0).reshape((-1, num_classes))
                plotProgress(all_training_losses, all_training_accuracies, all_validation_losses,
                             all_validation_accuracies, os.path.join(results_dir, "%s.png" % EXPERIMENT_NAME),
                             n_feedbacks_per_epoch, val_dice_scores=dice_scores, dice_labels=["brain", "1", "2", "3", "4", "5"])
            loss_vec, acc, l = train_fn(data, seg)

            loss = loss_vec.mean()
            train_loss += loss
            train_loss_tmp += loss
            train_acc_tmp += acc
            batch_ctr += 1
            if batch_ctr > (n_batches_per_epoch-1):
                break

        train_loss /= n_batches_per_epoch
        print "training loss average on epoch: ", train_loss

        val_loss = 0
        accuracies = []
        valid_batch_ctr = 0
        all_dice = []
        for data_dict in data_gen_validation:
            data = data_dict["data"].astype(np.float32)
            seg = data_dict["seg_onehot"].astype(np.float32).transpose(0, 2, 3, 1).reshape((-1, num_classes))
            w = np.zeros(num_classes, dtype=np.float32)
            w[np.unique(seg.argmax(-1))] = 1
            loss, acc, dice = val_fn(data, seg)
            dice[w == 0] = 2
            all_dice.append(dice)
            val_loss += loss
            accuracies.append(acc)
            valid_batch_ctr += 1
            if valid_batch_ctr > (n_test_batches-1):
                break
        all_dice = np.vstack(all_dice)
        dice_means = np.zeros(num_classes)
        for i in range(num_classes):
            dice_means[i] = all_dice[all_dice[:, i]!=2, i].mean()
        val_loss /= n_test_batches
        print "val loss: ", val_loss
        print "val acc: ", np.mean(accuracies), "\n"
        print "val dice: ", dice_means
        print "This epoch took %f sec" % (time.time()-epoch_start_time)
        all_val_dice_scores.append(dice_means)
        all_validation_losses.append(val_loss)
        all_validation_accuracies.append(np.mean(accuracies))
        dice_scores = np.concatenate(all_val_dice_scores, axis=0).reshape((-1, num_classes))
        plotProgress(all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies,
                     os.path.join(results_dir, "%s.png" % EXPERIMENT_NAME), n_feedbacks_per_epoch, val_dice_scores=dice_scores,
                     dice_labels=["brain", "1", "2", "3", "4", "5"])
        with open(os.path.join(results_dir, "%s_Params.pkl" % (EXPERIMENT_NAME)), 'w') as f:
            cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)
        with open(os.path.join(results_dir, "%s_allLossesNAccur.pkl"% (EXPERIMENT_NAME)), 'w') as f:
            cPickle.dump([all_training_losses, all_training_accuracies, all_validation_losses,
                          all_validation_accuracies, all_val_dice_scores], f)
        epoch += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="fold", type=int)
    parser.add_argument("-c", help="config file", type=str)
    args = parser.parse_args()
    run(args.c, args.f)