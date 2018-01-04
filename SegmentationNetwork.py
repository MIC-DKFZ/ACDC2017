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
import cPickle
from NetworkArchitecture import NetworkArchitecture, SegmentationArchitecture
import theano

floatX = theano.config.floatX

def hard_dice_per_img_in_batch(y_pred, y_true, n_classes, BATCH_SIZE):
    import theano.tensor as T
    num_pixels_per_sample = y_true.shape[0] // BATCH_SIZE
    # y_true must be label map, not one hot encoding
    y_true = T.flatten(y_true)
    y_pred = T.argmax(y_pred, axis=1)

    dice = T.zeros((BATCH_SIZE, n_classes))
    y_pred = y_pred.reshape((BATCH_SIZE, num_pixels_per_sample))
    y_true = y_true.reshape((BATCH_SIZE, num_pixels_per_sample))

    for b in range(BATCH_SIZE):
        for i in range(n_classes):
            i_val = T.constant(i)
            y_true_i = T.eq(y_true[b], i_val)
            y_pred_i = T.eq(y_pred[b], i_val)
            dice = T.set_subtensor(dice[b, i], (T.constant(2.) * T.sum(y_true_i * y_pred_i) + T.constant(1e-7)) /
                                   (T.sum(y_true_i) + T.sum(y_pred_i) + T.constant(1e-7)))
    return dice


class SegmentationNetwork(SegmentationArchitecture):
    def __init__(self, batch_size, use_and_update_bn_averages, void_labels=None):
        self.batch_size = batch_size
        SegmentationArchitecture.__init__(self)
        self.base_lr = None
        self.lr_decay = None
        self.num_epochs = None
        self.batches_per_epoch = None
        self.seed = None
        self.loss = None
        self.solver = None
        self.use_and_update_bn_averages = use_and_update_bn_averages
        self.l2_penalty = None
        self.is_trained = False
        self.num_classes = self.output_layer.output_shape[1]
        self.void_labels = void_labels

        self.train_fn = self.val_fn = self.pred_seg_prob_det = self.pred_seg_prob_nondet = None
        self._output_det = self._output_nondet = None

    def _initialize_training(self):
        if self.seed is not False:
            np.random.seed(self.seed)
            lasagne.random.set_rng(np.random.RandomState(self.seed))
        # add some weight decay
        l2_loss = lasagne.regularization.regularize_network_params(self.output_layer, lasagne.regularization.l2) * 1e-5

        # the distinction between prediction_train and test is important only if we enable dropout
        prediction_train = lasagne.layers.get_output(self.output_layer, self.input_var, deterministic=False,
                                                     batch_norm_update_averages=self.use_and_update_bn_averages,
                                                     batch_norm_use_averages=False)

        loss_train = lasagne.objectives.categorical_crossentropy(prediction_train, self.output_var)

        loss_train = loss_train.mean()
        loss_train += l2_loss
        if self.void_labels is None:
            acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), self.output_var.argmax(-1)),
                               dtype=theano.config.floatX)
        else:
            mask = T.ones_like(prediction_train[:, 0], dtype='int32')
            for el in self.void_labels:
                mask = T.switch(T.eq(self.output_var.argmax(1), el), np.int32(0), mask)
            acc_train = (T.eq(T.argmax(prediction_train, axis=1), self.output_var.argmax(-1)) * mask).sum() / \
                        mask.astype('float32').sum()


        prediction_test = lasagne.layers.get_output(self.output_layer, self.input_var, deterministic=True,
                                                    batch_norm_update_averages=False,
                                                    batch_norm_use_averages=self.use_and_update_bn_averages)

        loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, self.output_var)


        loss_val = loss_val.mean()
        loss_val += l2_loss
        if self.void_labels is None:
            acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), self.output_var.argmax(-1)),
                         dtype=theano.config.floatX)
        else:
            mask = T.ones_like(prediction_train[:, 0], dtype='int32')
            for el in self.void_labels:
                mask = T.switch(T.eq(self.output_var.argmax(1), el), np.int32(0), mask)
            acc = (T.eq(T.argmax(prediction_test, axis=1), self.output_var.argmax(-1)) * mask).sum() / \
                  mask.astype('float32').sum()


        # learning rate has to be a shared variable because we decrease it with every epoch
        params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
        self.lr_shared = theano.shared(np.array([self.base_lr]).astype(floatX)[0])
        updates = lasagne.updates.adam(T.grad(loss_train, params), params, learning_rate=self.lr_shared, beta1=0.9,
                                       beta2=0.999)

        # create a convenience function to get the segmentation
        dc = hard_dice_per_img_in_batch(prediction_test, self.output_var.argmax(1), self.num_classes, self.batch_size)
        dc_tr = hard_dice_per_img_in_batch(prediction_train, self.output_var.argmax(1), self.num_classes,
                                           self.batch_size)

        self.train_fn = theano.function([self.input_var, self.output_var], [loss_train, acc_train, dc_tr],
                                        updates=updates)
        self.val_fn = theano.function([self.input_var, self.output_var], [loss_val, acc, dc])

    def _update_lr(self, epoch):
        self.lr_shared.set_value(np.array([self.base_lr * self.lr_decay ** epoch]).astype(floatX)[0])


    def _initialize_pred_seg(self):
        seg_output_det = lasagne.layers.get_output(self.seg_layer, self.input_var, deterministic=True,
                                                     batch_norm_update_averages=False,
                                                     batch_norm_use_averages=self.use_and_update_bn_averages)
        seg_output_nondet = lasagne.layers.get_output(self.seg_layer, self.input_var, deterministic=False,
                                                     batch_norm_update_averages=False,
                                                     batch_norm_use_averages=self.use_and_update_bn_averages)
        from Utils.general_utils import softmax_helper
        seg_output_det = softmax_helper(seg_output_det)
        seg_output_nondet = softmax_helper(seg_output_nondet)
        self.pred_seg_prob_det = theano.function([self.input_var], seg_output_det, updates=None)
        self.pred_seg_prob_nondet = theano.function([self.input_var], seg_output_nondet, updates=None)

    def _maybe_initialize_pred_seg(self):
        if self.pred_seg_prob_det is None:
            self._initialize_pred_seg()

    def _iter_batches(self, generator, num_iters, mode):
        assert mode in ['train', 'val', 'test'], \
            "unrecognized string for mode: %s. Use \'train\', \'test\' or \'val\'" % mode
        if mode == 'train':
            fn = self.train_fn
        else:
            fn = self.val_fn
        all_accs = []
        all_losses = []
        all_dice = []
        all_dice_weights = []

        for batch in range(num_iters):
            data_dict = generator.next()
            data = data_dict["data"].astype(floatX)
            seg = data_dict["seg_onehot"].astype(floatX).transpose(0, 2, 3, 4, 1).reshape((-1, self.num_classes))
            loss, acc, dc = fn(data, seg)

            batch_size = data_dict['data'].shape[0]
            dc_weights = np.zeros((batch_size, self.num_classes))
            for b in range(batch_size):
                dc_weights[b][np.unique(data_dict['seg_onehot'][b].argmax(0)).astype(int)] = 1

            all_accs.append(acc)
            all_losses.append(loss)
            all_dice.append(dc)
            all_dice_weights.append(dc_weights)

        all_dice = np.vstack(all_dice)
        all_dice_weights = np.vstack(all_dice_weights)
        assert all_dice.shape == all_dice_weights.shape
        dice_scores = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            dice_scores[c] = np.mean(all_dice[:, c][all_dice_weights[:, c] != 0])
        return np.mean(all_accs), np.mean(all_losses), dice_scores

    def train(self, train_gen, val_gen, n_epochs=100, n_batches_per_epoch=100, base_lr=1e-5, lr_decay=0.98, seed=1234,
              loss="dice", solver=lasagne.updates.adam, l2_penalty=1e-5, best_params_file=None, latest_params_file=None,
              patience=50, plot_fname=None):
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.num_epochs = n_epochs
        self.batches_per_epoch = n_batches_per_epoch
        self.seed = seed
        self.loss = loss
        self.solver = solver
        self.l2_penalty = l2_penalty

        if self.train_fn is None:
            self._initialize_training()

        best_val_loss = 1e99
        best_params = None
        val_loss_not_improved_in = 0

        all_tr_losses = []
        all_val_losses = []
        all_tr_accs = []
        all_val_accs = []
        all_dice_scores_tr = []
        all_dice_scores_val = []

        epoch = 0
        while epoch < self.num_epochs:
            self._update_lr(epoch)
            tr_acc, tr_loss, tr_dice = self._iter_batches(train_gen, self.batches_per_epoch, 'train')
            val_acc, val_loss, val_dice = self._iter_batches(val_gen, self.batches_per_epoch / 4, 'val')

            print("Epoch %03.0d finished: tr acc: %02.4f, tr loss: %02.4f, val acc: %02.4f, val loss: %02.4f" % (
                epoch, tr_acc, tr_loss, val_acc, val_loss))
            print("Dice scores tr: ", str(tr_dice))
            print("Dice scores val: ", str(val_dice))
            all_tr_losses.append(tr_loss)
            all_val_losses.append(val_loss)
            all_tr_accs.append(tr_acc)
            all_val_accs.append(val_acc)
            all_dice_scores_tr.append(tr_dice)
            all_dice_scores_val.append(val_dice)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = lasagne.layers.get_all_param_values(self.output_layer)
                val_loss_not_improved_in = 0
            else:
                val_loss_not_improved_in += 1

            if val_loss_not_improved_in >= patience:
                break

            epoch += 1
            if plot_fname is not None:
                self._print_progress(range(epoch), all_tr_losses, all_tr_accs, np.vstack(all_dice_scores_tr),
                                     all_val_losses, all_val_accs, np.vstack(all_dice_scores_val), plot_fname)

            if latest_params_file is not None:
                with open(latest_params_file, 'w') as f:
                    cPickle.dump(best_params, f)

        if best_params_file is not None:
            with open(best_params_file, 'w') as f:
                cPickle.dump(best_params, f)

        lasagne.layers.set_all_param_values(self.output_layer, best_params)
        self.is_trained = True

    def set_params(self, params):
        NetworkArchitecture.set_params(self, params)
        self.is_trained = True

    def load_params(self, fname):
        NetworkArchitecture.load_params(self, fname)
        self.is_trained = True

    def fine_tune(self):
        raise NotImplementedError

    def pred(self, X, deterministic=True):
        self._maybe_initialize_pred_seg()
        if not self.is_trained:
            Warning("Warning running SegmentationNetwork.predict(): Train the network first!")
        return self.pred_proba(X, deterministic).argmax(1)

    def pred_proba(self, X, deterministic=True):
        self._maybe_initialize_pred_seg()
        if not self.is_trained:
            Warning("Warning running SegmentationNetwork.predict_proba(): Train the network first!")
        if deterministic:
            return self.pred_seg_prob_det(X)
        else:
            return self.pred_seg_prob_nondet(X)

    def predict_in_batches(self, X, batch_size, shuffle=True, deterministic=True):
        if not self.is_trained:
            Warning("Warning running SegmentationNetwork.predict(): Train the network first!")
        return self.predict_proba_in_batches(X, batch_size, shuffle, deterministic).argmax(1)

    def predict_proba_in_batches(self, X, batch_size, shuffle=True, deterministic=True):
        '''
        Predicts in batches of size batch_size. Will loop around if X % batch_size != 0
        :param X:
        :param batch_size:
        :param shuffle:
        :return:
        '''
        self._maybe_initialize_pred_seg()
        if not self.is_trained:
            Warning("Warning running SegmentationNetwork.predict_proba(): Train the network first!")
        idx = range(len(X))
        if shuffle:
            np.random.shuffle(idx)
        num_batches = int(np.ceil(len(X) / float(batch_size)))
        add_for_full_batches = int(num_batches * batch_size - len(idx))
        idx += idx[:add_for_full_batches]
        preds = np.zeros(([int(num_batches * batch_size)] + list(self.output_layer.output_shape[1:])))
        for b in range(num_batches):
            preds[idx[b*batch_size : (b+1) * batch_size]] = self.pred_proba(X[idx[b*batch_size : (b+1) * batch_size]],
                                                                            deterministic)
        return preds[:len(X)]

    def _print_progress(self, progress, train_loss, train_acc, train_dice, val_loss, val_acc, val_dice, fname):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(24, 12))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(progress, train_loss, 'b--', linewidth=2)
        ax1.plot(progress, val_loss, color='b', linewidth=2)
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')

        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])


        if train_acc is not None and val_acc is not None: # we need this for the generative adversarial network
            ax2 = ax1.twinx()
            if train_acc is not None:
                ax2.plot(progress, train_acc, color=[1, 0, 0])
            if val_acc is not None:
                ax2.plot(progress, val_acc, '--', color=[1, 0, 0])
            ax2.set_ylabel('accuracy')
            for t2 in ax2.get_yticklabels():
                t2.set_color('r')
            ax2_legend_text = ['trainAcc', 'validAcc']
            ax2.legend(ax2_legend_text, loc="center right", bbox_to_anchor=(1.3, 0.4))
            ax2.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        ax3_legend_text = []

        ax3 = plt.subplot(1, 2, 2)
        dice_labels = range(len(val_dice[0]))
        assert len(train_dice) == len(train_loss)
        for c in xrange(train_dice.shape[1]):
            ax3.plot(progress, train_dice[:, c], linestyle=":", linewidth=4, markersize=10)
            ax3_legend_text.append("train_dc_%s" % str(dice_labels[c]))

        plt.gca().set_prop_cycle(None)

        assert len(val_dice) == len(val_loss)
        for c in xrange(val_dice.shape[1]):
            ax3.plot(progress, val_dice[:, c], linestyle="--", linewidth=4, markersize=10)
            ax3_legend_text.append("val_dc_%s" % str(dice_labels[c]))

        ax1.legend(['trainLoss', 'validLoss'], loc="center right", bbox_to_anchor=(1.3, 0.3))
        ax3.legend(ax3_legend_text, loc="center right", bbox_to_anchor=(1.3, 0.5))
        plt.savefig(fname)
        plt.close()


