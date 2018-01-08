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
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import lasagne
import theano
import os
import sys
import theano.tensor
import theano.tensor as T
from dataset_utils import load_dataset
from utils import predict_patient_2D_net
from collections import OrderedDict
from utils import compute_typical_metrics
from utils import get_split, softmax_helper
import imp


def run_validation(pred_fn, results_out_folder, use_patients, BATCH_SIZE=None, n_repeats=1, min_size=None,
                   input_img_must_be_divisible_by=16, do_mirroring=True, preprocess_fn=None, save_segmentation=True,
                   plot_segmentation=False):
    all_results = OrderedDict()
    segmentation_groups = OrderedDict()
    segmentation_groups['LVM'] = [2]
    segmentation_groups['LVC'] = [3]
    segmentation_groups['RV'] = [1]
    segmentation_groups['complete'] = [1, 2, 3]
    for pat in use_patients.keys():
        print pat
        output_folder = os.path.join(results_out_folder, "%03.0d" % pat)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        this_patient = use_patients[pat]
        all_results[pat] = OrderedDict()
        for tpe in ["ed", "es"]:
            seg_combined = this_patient["%s_gt"%tpe]
            data = this_patient["%s_data"%tpe]
            patient, predicted_segmentation, softmax_pred = predict_patient_2D_net(pred_fn, this_patient["%s_data"%tpe],
                                                                                   do_mirroring, n_repeats, BATCH_SIZE,
                                                                                   input_img_must_be_divisible_by,
                                                                                   preprocess_fn, min_size=min_size)

            results = OrderedDict()
            for k in segmentation_groups:
                precision, recall, false_positive_rate, false_omission_rate, dice, assd, vol_gt, vol_pred = \
                    compute_typical_metrics(seg_combined, predicted_segmentation, segmentation_groups[k])
                results[k] = {}
                results[k]['precision'] = precision
                results[k]['recall'] = recall
                results[k]['false_positive_rate'] = false_positive_rate
                results[k]['false_omission_rate'] = false_omission_rate
                results[k]['dice'] = dice
                results[k]['assd'] = assd
                results[k]['vol_gt'] = vol_gt
                results[k]['vol_pred'] = vol_pred
            all_results[pat][tpe] = results

            # save results to human readable file
            with open(os.path.join(output_folder, "evaluation_metrics_%s.txt" % tpe), 'w') as f:
                for k in segmentation_groups.keys():
                    f.write("%s:\n" % (k))
                    for r in results[k].keys():
                        f.write("%s, %f\n" % (r, results[k][r]))
                    f.write("\n")

            with open(os.path.join(output_folder, "evaluation_metrics_%s.pkl" % tpe), 'w') as f:
                cPickle.dump(results, f)

            if save_segmentation:
                np.savez_compressed(os.path.join(output_folder, "gt_and_pred_segm_%s" % tpe),
                                    pred=predicted_segmentation, gt=seg_combined, softmax_pred=softmax_pred)

            if plot_segmentation:
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0.3, 0.5, 1)])

                output_folder_images = os.path.join(output_folder, "seg_slices_%s" % tpe)
                if not os.path.isdir(output_folder_images):
                    os.mkdir(output_folder_images)

                seg_combined[seg_combined == 4] = 0
                print "writing segmentation images"
                n_rot = 2
                num_x = 1
                num_y = 4
                for i in range(0, predicted_segmentation.shape[0]):
                    predicted_segmentation[i][0, 0:6] = [0, 1, 2, 3, 4, 5]
                    seg_combined[i][0, 0:6] = [0, 1, 2, 3, 4, 5]
                    errors = seg_combined[i] == predicted_segmentation[i]
                    errors[0, 0:2] = [True, False]
                    plt.figure(figsize=(18, 10))
                    ctr = 1

                    plt.subplot(num_x, num_y, ctr)
                    plt.imshow(np.rot90(data[0][i], n_rot), cmap="gray")
                    plt.title("CMI")
                    ctr += 1

                    plt.subplot(num_x, num_y, ctr)
                    plt.imshow(np.rot90(predicted_segmentation[i], n_rot), cmap=cmap)
                    plt.title("prediction")
                    ctr += 1

                    plt.subplot(num_x, num_y, ctr)
                    plt.imshow(np.rot90(seg_combined[i], n_rot), cmap=cmap)
                    plt.title("gt")
                    ctr += 1

                    plt.subplot(num_x, num_y, ctr)
                    plt.imshow(np.rot90(errors, n_rot), cmap="gray")
                    plt.title("errors")
                    ctr += 1

                    plt.tight_layout()

                    plt.savefig(
                        os.path.join(output_folder_images, "patient%d_segWholeDataset_z%03.0f" % (pat, i)))
                    plt.close()

        # save all results
        with open(os.path.join(results_out_folder, "all_metrics.pkl"), 'w') as f:
            cPickle.dump(all_results, f)

        # create a human readable csv with summary at the bottom
        global_averages_es = {}
        global_averages_ed = {}
        for k in segmentation_groups.keys():
            these_values = []
            for i in all_results.keys():
                if not np.isnan(all_results[i]['es'][k]['dice']):
                    these_values.append(all_results[i]['es'][k]['dice'])
            global_averages_es[k] = np.mean(these_values)
            these_values = []
            for i in all_results.keys():
                if not np.isnan(all_results[i]['ed'][k]['dice']):
                    these_values.append(all_results[i]['ed'][k]['dice'])
            global_averages_ed[k] = np.mean(these_values)

        with open(os.path.join(results_out_folder, "global_average_dice.txt"), 'w') as f:
            f.write("es:\n")
            for k in segmentation_groups.keys():
                f.write("%s: %f\n" % (k, global_averages_es[k]))
            f.write("\ned:\n")
            for k in segmentation_groups.keys():
                f.write("%s: %f\n" % (k, global_averages_ed[k]))
            f.write("\ncombined:\n")
            for k in segmentation_groups.keys():
                f.write("%s: %f\n" % (k, np.mean((global_averages_ed[k], global_averages_es[k]))))


def run(config_file, fold=0):
    cf = imp.load_source('cf', config_file)
    print fold
    dataset_root = cf.dataset_root
    # this is seeded, will be identical each time
    train_keys, test_keys = get_split(fold)

    val_data = load_dataset(test_keys, root_dir=dataset_root)

    use_patients = val_data
    experiment_name = cf.EXPERIMENT_NAME
    results_folder = os.path.join(cf.results_dir, "fold%d/" % fold)
    mode='val'

    BATCH_SIZE = cf.BATCH_SIZE

    n_repeats = cf.val_num_repeats

    x_sym = T.tensor4()

    nt, net, seg_layer = cf.nt, cf.net, cf.seg_layer
    output_layer = seg_layer

    best_epoch = 299

    results_out_folder = results_folder + "ep%03.0d_MA" % (best_epoch)
    if not os.path.isdir(results_out_folder):
        os.mkdir(results_out_folder)
    results_out_folder += "/%s_mirror"%mode
    if not os.path.isdir(results_out_folder):
        os.mkdir(results_out_folder)

    with open(os.path.join(results_folder, "%s_Params.pkl" % (experiment_name)), 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer, params)

    print "compiling theano functions"
    output = softmax_helper(lasagne.layers.get_output(output_layer, x_sym, deterministic=not
    cf.val_bayesian_prediction, batch_norm_update_averages=False, batch_norm_use_averages=False))
    pred_fn = theano.function([x_sym], output)
    _ = pred_fn(np.random.random((BATCH_SIZE, 1, 384, 352)).astype(np.float32))
    run_validation(pred_fn, results_out_folder, use_patients, BATCH_SIZE=BATCH_SIZE, n_repeats=n_repeats,
                   save_segmentation=cf.val_save_segmentation, plot_segmentation=cf.val_plot_segmentation,
                   min_size=cf.val_min_size, do_mirroring=cf.val_do_mirroring,
                   input_img_must_be_divisible_by=cf.val_input_img_must_be_divisible_by,
                   preprocess_fn=cf.val_preprocess_fn)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="fold", type=int)
    parser.add_argument("-c", help="config file", type=str)
    args = parser.parse_args()
    run(args.c, args.f)