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
import lasagne
matplotlib.use('Agg')
import numpy as np
import sys
import os
from BatchGenerator import BatchGenerator
from dataset_utils import load_dataset
from utils import get_split
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import Compose, RndTransform
from batchgenerators.transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms import GammaTransform, ConvertSegToOnehotTransform
from batchgenerators.transforms import RandomCropTransform, CutOffOutliersTransform
from batchgenerators.transforms import ZeroMeanUnitVarianceTransform
from transformers import MotionAugmentationTransform, Convert3DTo2DTransform, Convert2DTo3DTransform
from paths import results_folder, path_acdc_3d
from SegmentationNetwork import SegmentationNetwork
from lasagne.layers import Conv3DLayer, InputLayer, DimshuffleLayer, ReshapeLayer, DropoutLayer, \
    Upscale3DLayer, ElemwiseSumLayer, Pool3DLayer, ConcatLayer, batch_norm
from collections import OrderedDict
from theano.tensor import tensor5, matrix
from lasagne.layers import NonlinearityLayer


class UNet3D_ACDC(SegmentationNetwork):
    def __init__(self, n_input_channels=1, num_output_classes=2, pad='same', input_dim=(128, 128, 128),
                 base_n_filters=64, dropout=None, nonlinearity=lasagne.nonlinearities.rectify, instance_norm=True,
                 batch_size=2, use_and_update_bn_averages=False, void_labels=None):
        self.n_input_channels = n_input_channels
        self.num_classes = num_output_classes
        self.pad = pad
        self.input_dim = input_dim
        self.base_n_filters = base_n_filters
        self.dropout = dropout
        self.nonlinearity = nonlinearity
        self.instance_norm = instance_norm

        SegmentationNetwork.__init__(self, batch_size, use_and_update_bn_averages, void_labels=void_labels)

    def build_network(self):
        self.input_var = tensor5()
        self.output_var = matrix()
        net = OrderedDict()
        if self.instance_norm:
            norm_fct = batch_norm
            norm_kwargs = {'axes': (2, 3, 4)}
        else:
            norm_fct = batch_norm
            norm_kwargs = {'axes': 'auto'}

        self.input_layer = net['input'] = InputLayer(
            (self.batch_size, self.n_input_channels, self.input_dim[0], self.input_dim[1], self.input_dim[2]),
            self.input_var)

        net['contr_1_1'] = norm_fct(
            Conv3DLayer(net['input'], self.base_n_filters, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        net['contr_1_2'] = norm_fct(
            Conv3DLayer(net['contr_1_1'], self.base_n_filters, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        net['pool1'] = Pool3DLayer(net['contr_1_2'], (1, 2, 2))

        net['contr_2_1'] = norm_fct(
            Conv3DLayer(net['pool1'], self.base_n_filters * 2, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        net['contr_2_2'] = norm_fct(
            Conv3DLayer(net['contr_2_1'], self.base_n_filters * 2, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        l = net['pool2'] = Pool3DLayer(net['contr_2_2'], (1, 2, 2))
        if self.dropout is not None:
            l = DropoutLayer(l, p=self.dropout)

        net['contr_3_1'] = norm_fct(
            Conv3DLayer(l, self.base_n_filters * 4, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        net['contr_3_2'] = norm_fct(
            Conv3DLayer(net['contr_3_1'], self.base_n_filters * 4, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        l = net['pool3'] = Pool3DLayer(net['contr_3_2'], (1, 2, 2))
        if self.dropout is not None:
            l = DropoutLayer(l, p=self.dropout)

        net['contr_4_1'] = norm_fct(
            Conv3DLayer(l, self.base_n_filters * 8, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        net['contr_4_2'] = norm_fct(
            Conv3DLayer(net['contr_4_1'], self.base_n_filters * 8, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        l = net['pool4'] = Pool3DLayer(net['contr_4_2'], (1, 2, 2))
        if self.dropout is not None:
            l = DropoutLayer(l, p=self.dropout)

        net['encode_1'] = norm_fct(
            Conv3DLayer(l, self.base_n_filters * 16, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        l = net['encode_2'] = norm_fct(
            Conv3DLayer(net['encode_1'], self.base_n_filters * 16, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        net['upscale1'] = Upscale3DLayer(l, (1, 2, 2))

        l = net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']],
                                         cropping=(None, None, "center", "center", "center"))
        if self.dropout is not None:
            l = DropoutLayer(l, p=self.dropout)
        net['expand_1_1'] = norm_fct(
            Conv3DLayer(l, self.base_n_filters * 8, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        l = net['expand_1_2'] = norm_fct(
            Conv3DLayer(net['expand_1_1'], self.base_n_filters * 8, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        net['upscale2'] = Upscale3DLayer(l, (1, 2, 2))

        l = net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']],
                                         cropping=(None, None, "center", "center", "center"))
        if self.dropout is not None:
            l = DropoutLayer(l, p=self.dropout)
        net['expand_2_1'] = norm_fct(
            Conv3DLayer(l, self.base_n_filters * 4, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        ds2 = l = net['expand_2_2'] = norm_fct(
            Conv3DLayer(net['expand_2_1'], self.base_n_filters * 4, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        net['upscale3'] = Upscale3DLayer(l, (1, 2, 2))

        l = net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']],
                                         cropping=(None, None, "center", "center", "center"))
        if self.dropout is not None:
            l = DropoutLayer(l, p=self.dropout)
        net['expand_3_1'] = norm_fct(
            Conv3DLayer(l, self.base_n_filters * 2, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        l = net['expand_3_2'] = norm_fct(
            Conv3DLayer(net['expand_3_1'], self.base_n_filters * 2, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        net['upscale4'] = Upscale3DLayer(l, (1, 2, 2))

        net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']],
                                     cropping=(None, None, "center", "center", "center"))
        net['expand_4_1'] = norm_fct(
            Conv3DLayer(net['concat4'], self.base_n_filters, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)
        net['expand_4_2'] = norm_fct(
            Conv3DLayer(net['expand_4_1'], self.base_n_filters, 3, nonlinearity=self.nonlinearity, pad=self.pad,
                        W=lasagne.init.HeNormal(gain="relu")), **norm_kwargs)

        net['output_segmentation'] = Conv3DLayer(net['expand_4_2'], self.num_classes, 1, nonlinearity=None)

        ds2_1x1_conv = Conv3DLayer(ds2, self.num_classes, 1, 1, 'same', nonlinearity=lasagne.nonlinearities.linear,
                                   W=lasagne.init.HeNormal(gain='relu'))
        ds1_ds2_sum_upscale = Upscale3DLayer(ds2_1x1_conv, (1, 2, 2))
        ds3_1x1_conv = Conv3DLayer(net['expand_3_2'], self.num_classes, 1, 1, 'same',
                                   nonlinearity=lasagne.nonlinearities.linear,
                                   W=lasagne.init.HeNormal(gain='relu'))
        ds1_ds2_sum_upscale_ds3_sum = ElemwiseSumLayer((ds1_ds2_sum_upscale, ds3_1x1_conv))
        ds1_ds2_sum_upscale_ds3_sum_upscale = Upscale3DLayer(ds1_ds2_sum_upscale_ds3_sum, (1, 2, 2))

        self.seg_layer = l = ElemwiseSumLayer(
            (net['output_segmentation'], ds1_ds2_sum_upscale_ds3_sum_upscale))

        net['dimshuffle'] = DimshuffleLayer(l, (0, 2, 3, 4, 1))
        batch_size, n_z, n_rows, n_cols, _ = lasagne.layers.get_output(net['dimshuffle']).shape
        net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (batch_size * n_rows * n_cols * n_z, self.num_classes))
        self.output_layer = net['output_flattened'] = NonlinearityLayer(net['reshapeSeg'],
                                                                        nonlinearity=lasagne.nonlinearities.softmax)


def create_data_gen_train(patient_data_train, BATCH_SIZE, num_classes, patch_size,
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
    data_gen_train = BatchGenerator(patient_data_train, BATCH_SIZE, num_batches=None, seed=False,
                                    PATCH_SIZE=(10, 352, 352))

    # train transforms
    tr_transforms = []
    tr_transforms.append(MotionAugmentationTransform(0.1, 0, 20))
    tr_transforms.append(MirrorTransform((3, 4)))
    tr_transforms.append(Convert3DTo2DTransform())
    tr_transforms.append(
        RndTransform(SpatialTransform(patch_size[1:], 112,
                                      do_elastic_transform, alpha,
                                      sigma,
                                      do_rotation, a_x, a_y, a_z,
                                      do_scale, scale_range, 'constant', 0, 3, 'constant', 0, 0,
                                      random_crop=False), prob=0.67,
                     alternative_transform=RandomCropTransform(patch_size[1:])))
    tr_transforms.append(Convert2DTo3DTransform(patch_size))
    tr_transforms.append(RndTransform(GammaTransform((0.85, 1.3), False), prob=0.5))
    tr_transforms.append(RndTransform(GammaTransform((0.85, 1.3), True), prob=0.5))
    tr_transforms.append(CutOffOutliersTransform(0.3, 99.7, True))
    tr_transforms.append(ZeroMeanUnitVarianceTransform(True))
    tr_transforms.append(ConvertSegToOnehotTransform(range(num_classes), 0, 'seg_onehot'))

    tr_composed = Compose(tr_transforms)
    tr_mt_gen = MultiThreadedAugmenter(data_gen_train, tr_composed, num_workers, num_cached_per_worker, seeds)
    tr_mt_gen.restart()
    return tr_mt_gen



def get_network(mode="train"):
    assert mode in ['train', 'val']
    inp_size = INPUT_PATCH_SIZE
    if mode == 'val':
        inp_size = (None, None, None)
    net = UNet3D_ACDC(1, 4, pad="same", input_dim=inp_size, base_n_filters=26, dropout=0.5,
                 nonlinearity=lasagne.nonlinearities.leaky_rectify, instance_norm=False,  batch_size=BATCH_SIZE,
                 use_and_update_bn_averages=False, void_labels=None)
    return net


def get_train_val_generators(fold):
    tr_keys, te_keys = get_split(fold, split_seed)
    train_data = {i: dataset[i] for i in tr_keys}
    val_data = {i: dataset[i] for i in te_keys}

    data_gen_train = create_data_gen_train(train_data, BATCH_SIZE,
                                           num_classes, INPUT_PATCH_SIZE, num_workers=num_workers,
                                           do_elastic_transform=True, alpha=(0., 350.), sigma=(14., 17.),
                                           do_rotation=True, a_x=(0, 2.*np.pi), a_y=(-0.000001, 0.00001),
                                           a_z=(-0.000001, 0.00001), do_scale=True, scale_range=(0.7, 1.3),
                                           seeds=workers_seeds)  # new se has no brain mask

    data_gen_validation = BatchGenerator(val_data, BATCH_SIZE, num_batches=None, seed=False,
                                         PATCH_SIZE=INPUT_PATCH_SIZE)
    val_transforms = []
    val_transforms.append(ConvertSegToOnehotTransform(range(4), 0, 'seg_onehot'))
    data_gen_validation = MultiThreadedAugmenter(data_gen_validation, Compose(val_transforms), 1, 2, [0])
    return data_gen_train, data_gen_validation


dataset = load_dataset(root_dir=path_acdc_3d)
split_seed = 12345

np.random.seed(65432)
lasagne.random.set_rng(np.random.RandomState(98765))
sys.setrecursionlimit(2000)
BATCH_SIZE = 4
INPUT_PATCH_SIZE = (10, 224, 224)
num_classes = 4
num_input_channels = 1
EXPERIMENT_NAME = "UNet3D_final"
if not os.path.isdir(os.path.join(results_folder, "ACDC_lasagne")):
    os.mkdir(os.path.join(results_folder, "ACDC_lasagne"))
results_dir = os.path.join(results_folder, "ACDC_lasagne", EXPERIMENT_NAME)

if not os.path.isdir(results_dir):
    os.mkdir(results_dir)


n_epochs = 300
lr_decay = np.float32(0.98)
base_lr = np.float32(0.0005)
n_batches_per_epoch = 100
n_test_batches = 10
n_feedbacks_per_epoch = 10.
num_workers = 6
workers_seeds = [123, 1234, 12345, 123456, 1234567, 12345678]
loss = 'crossentropy'
solver = lasagne.updates.adam
l2_penalty = 1e-5
patience = 999  # this will ensure we actually just do the 300 epochs and dont stop early (we will be loading the last,
# not the best params) -> no overfitting on cross-validation


bayesian_prediction = True
num_repeats = 4
do_mirroring = True
plot_segmentation = False
save_segmentation = True
new_shape_must_be_divisible_by = 16
use_t1km_sub = False
preprocess_fn = None
min_size = INPUT_PATCH_SIZE

# 4d predictions
dataset_root_raw = "/media/fabian/My Book/datasets/ACDC/training/"
predictions_4d_out_folder = "/home/fabian/code/ACDC/submission_stuff/results/4d_predictions_train"
target_spacing = (10., 1.25, 1.25)

dataset_root_test = "/media/fabian/My Book/datasets/ACDC/testing/testing/"
test_out_folder = os.path.join(results_dir, "test_predictions")
if not os.path.isdir(test_out_folder):
    os.mkdir(test_out_folder)