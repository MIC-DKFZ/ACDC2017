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
from sklearn.metrics import confusion_matrix
matplotlib.use('Agg')
import numpy as np
from skimage.morphology import label
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from theano import tensor as T
from medpy import metric
from skimage.transform import resize

def postprocess_prediction(seg):
    # basically look for connected components and choose the largest one, delete everything else
    mask = seg != 0
    lbls = label(mask, 8)
    lbls_sizes = [np.sum(lbls==i) for i in np.unique(lbls)]
    largest_region = np.argmax(lbls_sizes[1:]) + 1
    seg[lbls != largest_region]=0
    return seg


def pad_patient_3D(patient, shape_must_be_divisible_by=16, min_size=None):
    shp = patient.shape
    new_shp = [shp[0], shp[1] + shape_must_be_divisible_by - shp[1] % shape_must_be_divisible_by, shp[2] +
               shape_must_be_divisible_by - shp[2] % shape_must_be_divisible_by]
    if min_size is not None:
        new_shp = np.max(np.vstack((np.array(new_shp), np.array(min_size))), 0)
    for i in range(len(shp) - 1):
        if shp[i + 1] % shape_must_be_divisible_by == 0:
            new_shp[i + 1] -= shape_must_be_divisible_by
    return reshape_by_padding_upper_coords(patient, new_shp, 0), shp


def predict_patient_3D_net(net, patient_data, do_mirroring, do_bayesian, num_repeats, BATCH_SIZE=None,
                           new_shape_must_be_divisible_by=16, preprocess_fn=None, min_size=None):
    if preprocess_fn is not None:
        patient_data = preprocess_fn(patient_data)

    patient, old_shape = pad_patient_3D(patient_data, new_shape_must_be_divisible_by, min_size)
    new_shp = patient.shape

    data = np.zeros(tuple([1] + [1] + list(new_shp)), dtype=np.float32)

    data[0, 0] = patient

    if BATCH_SIZE is not None:
        data = np.vstack([data] * BATCH_SIZE)

    all_preds = []

    if do_mirroring:
        x = 4
    else:
        x = 1
    for i in range(num_repeats):
        for m in range(x):
            data_for_net = np.array(data)
            if m == 0:
                pass
            if m == 1:
                data_for_net = data_for_net[:, :, :, :, ::-1]
            if m == 2:
                data_for_net = data_for_net[:, :, :, ::-1, :]
            if m == 3:
                data_for_net = data_for_net[:, :, :, ::-1, ::-1]

            p = net.pred_proba(data_for_net, not do_bayesian)

            if m == 0:
                pass
            if m == 1:
                p = p[:, :, :, :, ::-1]
            if m == 2:
                p = p[:, :, :, ::-1, :]
            if m == 3:
                p = p[:, :, :, ::-1, ::-1]


            all_preds.append(p)

    stacked = np.vstack(all_preds)[:, :, :old_shape[0], :old_shape[1], :old_shape[2]]
    predicted_segmentation = stacked.mean(0).argmax(0)
    try:
        predicted_segmentation = postprocess_prediction(predicted_segmentation)
    except:
        print "post processing failed, probably due to empty segmentation (which in turn is due to empty time steps" \
              " in the raw data)"
    bayesian_predictions = stacked
    softmax_pred = stacked.mean(0)
    return predicted_segmentation, bayesian_predictions, softmax_pred

def predict_patient_2D_net(pred_fn, patient_data, do_mirroring, num_repeats, BATCH_SIZE=None,
                           new_shape_must_be_divisible_by=16, preprocess_fn=None, min_size=None):
    if preprocess_fn is not None:
        patient_data = preprocess_fn(patient_data)
    patient, old_shape = pad_patient_3D(patient_data, new_shape_must_be_divisible_by, min_size=min_size)
    new_shp = patient.shape
    data = np.zeros(tuple([1] + [1] + list(new_shp[1:])), dtype=np.float32)
    seg_pred = np.zeros(patient.shape, dtype=np.uint8)
    seg_pred_softmax = np.zeros([4] + list(patient.shape), dtype=np.float32)

    if do_mirroring:
        x = 4
    else:
        x = 1
    for slice in range(patient.shape[0]):
        all_preds = []
        for i in range(num_repeats):
            for m in range(x):
                data[0, 0] = patient[slice]
                if m == 1:
                    data_m = data
                elif m == 2:
                    data_m = data[:, :, ::-1, ::-1]
                elif m == 3:
                    data_m = data[:, :, :, ::-1]
                else:
                    data_m = data[:, :, ::-1, :]
                if BATCH_SIZE is not None:
                    data_for_net = np.vstack([data_m] * BATCH_SIZE)
                else:
                    data_for_net = data_m
                p = pred_fn(data_for_net)
                if m == 1:
                    p = p
                elif m == 2:
                    p = p[:, :, ::-1, ::-1]
                elif m == 3:
                    p = p[:, :, :, ::-1]
                else:
                    p = p[:, :, ::-1, :]
                all_preds.append(p)

        stacked = np.vstack(all_preds)
        predicted_segmentation = stacked.mean(0).argmax(0)
        seg_pred[slice] = predicted_segmentation
        seg_pred_softmax[:, slice] = stacked.mean(0)
    seg_pred = postprocess_prediction(seg_pred)
    seg_pred = seg_pred[:old_shape[0], :old_shape[1], :old_shape[2]]
    patient = patient[:old_shape[0], :old_shape[1], :old_shape[2]]
    seg_pred_softmax = seg_pred_softmax[:, :old_shape[0], :old_shape[1], :old_shape[2]]
    return patient, seg_pred, seg_pred_softmax


def compute_typical_metrics(seg_gt, seg_pred, labels):
    assert seg_gt.shape == seg_pred.shape
    mask_pred = np.zeros(seg_pred.shape, dtype=bool)
    mask_gt = np.zeros(seg_pred.shape, dtype=bool)

    for l in labels:
        mask_gt[seg_gt == l] = True
        mask_pred[seg_pred == l] = True

    vol_gt = np.sum(mask_gt)
    vol_pred = np.sum(mask_pred)

    try:
        cm = confusion_matrix(mask_pred.astype(int).ravel(), mask_gt.astype(int).ravel())
        TN = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TP = cm[1][1]
        precision = TP / float(TP + FP)
        recall = TP / float(TP + FN)
        fpr = FP / float(FP + TN)
        false_omission_rate = FN / float(FN + TN)
    except:
        precision = np.nan
        recall = np.nan
        fpr = np.nan
        false_omission_rate = np.nan

    try:
        dice = metric.dc(mask_pred, mask_gt)
        if np.sum(mask_gt) == 0:
            dice = np.nan
    except:
        dice = np.nan

    try:
        assd = metric.assd(mask_gt, mask_pred)
    except:
        assd = np.nan

    return precision, recall, fpr, false_omission_rate, dice, assd, vol_gt, vol_pred


def get_split(fold, seed=12345):
    # this is seeded, will be identical each time
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    all_keys = np.arange(1, 101)
    splits = kf.split(all_keys)
    for i, (train_idx, test_idx) in enumerate(splits):
        train_keys = all_keys[train_idx]
        test_keys = all_keys[test_idx]
        if i == fold:
            break
    return train_keys, test_keys



def soft_dice(y_pred, y_true):
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    intersect = T.sum(y_pred * y_true, 0)
    denominator = T.sum(y_pred, 0) + T.sum(y_true, 0)
    dice_scores = T.constant(2) * intersect / (denominator + T.constant(1e-6))
    return dice_scores


def hard_dice(y_pred, y_true, n_classes):
    # y_true must be label map, not one hot encoding
    y_true = T.flatten(y_true)
    y_pred = T.argmax(y_pred, axis=1)

    dice = T.zeros(n_classes)

    for i in range(n_classes):
        i_val = T.constant(i)
        y_true_i = T.eq(y_true, i_val)
        y_pred_i = T.eq(y_pred, i_val)
        dice = T.set_subtensor(dice[i], (T.constant(2.) * T.sum(y_true_i * y_pred_i) + T.constant(1e-7)) /
                               (T.sum(y_true_i) + T.sum(y_pred_i) + T.constant(1e-7)))
    return dice


def plotProgress(all_training_losses, all_training_accs, all_validation_losses, all_valid_accur, fname,
                 samplesPerEpoch=10, val_dice_scores=None, dice_labels=None, ylim_score=None):
    fig, ax1 = plt.subplots(figsize=(16, 12))
    trainLoss_x_values = np.arange(1/float(samplesPerEpoch), len(all_training_losses)/float(samplesPerEpoch)+0.000001,
                                   1/float(samplesPerEpoch))
    val_x_values = np.arange(1, len(all_validation_losses)+0.001, 1)
    ax1.plot(trainLoss_x_values, all_training_losses, 'b--', linewidth=2)
    ax1.plot(val_x_values, all_validation_losses, color='b', linewidth=2)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    if ylim_score is not None:
        ax1.set_ylim(ylim_score)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax2 = ax1.twinx()
    ax2.plot(trainLoss_x_values, all_training_accs, 'r--', linewidth=2)
    ax2.plot(val_x_values, all_valid_accur, color='r', linewidth=2)
    ax2.set_ylabel('accuracy')
    for t2 in ax2.get_yticklabels():
        t2.set_color('r')
    ax2_legend_text = ['trainAcc', 'validAcc']

    if val_dice_scores is not None:
        assert len(val_dice_scores) == len(all_validation_losses)
        num_auc_scores_per_timestep = val_dice_scores.shape[1]
        for auc_id in xrange(num_auc_scores_per_timestep):
            ax2.plot(val_x_values, val_dice_scores[:, auc_id], linestyle=":", linewidth=4, markersize=10)
            ax2_legend_text.append(dice_labels[auc_id])

    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax1.legend(['trainLoss', 'validLoss'], loc="center right", bbox_to_anchor=(1.3, 0.4))
    ax2.legend(ax2_legend_text, loc="center right", bbox_to_anchor=(1.3, 0.6))
    plt.savefig(fname)
    plt.close()

def softmax_helper(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2,len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0,0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    if len(shape) == 2:
        res[0:0+int(shape[0]), 0:0+int(shape[1])] = image
    elif len(shape) == 3:
        res[0:0+int(shape[0]), 0:0+int(shape[1]), 0:0+int(shape[2])] = image
    return res


def resize_segmentation(segmentation, new_shape, order=3):
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if len(unique_labels) == 2 or order == 0:
        return resize(segmentation, new_shape, order, mode="constant", cval=0, clip=True)
    else:
        reshaped_multihot = np.zeros([len(unique_labels)] + list(new_shape), dtype=segmentation.dtype)
        for i, c in enumerate(unique_labels):
            reshaped_multihot[i] = np.round(resize((segmentation == c).astype(float), new_shape, order, mode="constant",
                                                   cval=0, clip=True))
        reshaped = unique_labels[np.argmax(reshaped_multihot, 0)]
        return reshaped


def resize_softmax_output(softmax_output, new_shape, order=3):
    '''

    :param softmax_output: c x x x y x z
    :param new_shape: x x y x z
    :param order:
    :return:
    '''
    new_shp = [softmax_output.shape[0]] + list(new_shape)
    result = np.zeros(new_shp, dtype=softmax_output.dtype)
    for i in range(softmax_output.shape[0]):
        result[i] = resize(softmax_output[i].astype(float), new_shape, order, "constant", 0, True)
    return result

