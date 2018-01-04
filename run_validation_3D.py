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


import cPickle
import imp
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("configs")
import numpy as np
from utils import compute_typical_metrics
from collections import OrderedDict
from utils import predict_patient_3D_net

def run_validation(val_patients, net, do_mirroring, do_bayesian, num_repeats, validation_folder,
                   plot_segmentation=False, save_segmentation=True, BATCH_SIZE=None,
                   new_shape_must_be_divisible_by=16, preprocess_fn=None, min_size=None):
    all_results = OrderedDict()
    segmentation_groups = OrderedDict()
    segmentation_groups['LVM'] = [2]
    segmentation_groups['LVC'] = [3]
    segmentation_groups['RV'] = [1]
    segmentation_groups['complete'] = [1, 2, 3]

    for pat in val_patients.keys():
        print pat
        all_results[pat] = {}
        for tpe in ['es', 'ed']:
            seg_combined = np.copy(val_patients[pat]['%s_gt' % tpe])
            data = val_patients[pat]['%s_data' % tpe]
            predicted_segmentation, bayesian_predictions, softmax_pred = \
                predict_patient_3D_net(net,
                                       np.copy(val_patients[pat][
                                           '%s_data' % tpe]),
                                       do_mirroring,
                                       do_bayesian, num_repeats,
                                       BATCH_SIZE,
                                       new_shape_must_be_divisible_by,
                                       preprocess_fn,
                                        min_size)
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
            patient_folder = os.path.join(validation_folder, "%03.0d"%pat)
            if not os.path.isdir(patient_folder):
                os.mkdir(patient_folder)
            with open(os.path.join(patient_folder, "evaluation_metrics_%s.txt"%tpe), 'w') as f:
                for k in segmentation_groups.keys():
                    f.write("%s:\n" % (k))
                    for r in results[k].keys():
                        f.write("%s, %f\n" % (r, results[k][r]))
                    f.write("\n")

            # pickle results as well
            with open(os.path.join(patient_folder, "evaluation_metrics_%s.pkl"%tpe), 'w') as f:
                cPickle.dump(results, f)

            if save_segmentation:
                np.savez_compressed(os.path.join(patient_folder, "gt_and_pred_segm_%s"%tpe),
                                    pred=predicted_segmentation, gt=seg_combined, softmax_pred=softmax_pred)

            if plot_segmentation:
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0.3, 0.5, 1)])

                output_folder_images = os.path.join(patient_folder, "seg_slices_%s"%tpe)
                if not os.path.isdir(output_folder_images):
                    os.mkdir(output_folder_images)

                seg_combined[seg_combined==4] = 0
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
    with open(os.path.join(validation_folder, "all_metrics.pkl"), 'w') as f:
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

    with open(os.path.join(validation_folder, "global_average_dice.txt"), 'w') as f:
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
    print config_file
    net = cf.get_network(mode='val')


    out_folder = os.path.join(cf.results_dir, "fold%d" % fold)
    try:
        net.load_params(os.path.join(out_folder, "best_params2.pkl"))
    except IOError:
        try:
            print "could not load best params, trying latest:"
            net.load_params(os.path.join(out_folder, "latest_params.pkl"))
        except IOError:
            raise RuntimeError("Could not open parameters, error message: %s" % sys.exc_info())
    except Exception:
            raise RuntimeError("Exception during loading params: %s" % sys.exc_info())

    validation_folder = os.path.join(cf.results_dir, "fold%d" % fold, "validation")
    if not os.path.isdir(validation_folder):
        os.mkdir(validation_folder)

    net._initialize_pred_seg()

    do_bayesian = cf.bayesian_prediction
    num_repeats = cf.num_repeats
    do_mirroring = cf.do_mirroring
    plot_segmentation = cf.plot_segmentation
    save_segmentation = cf.save_segmentation
    BATCH_SIZE = cf.BATCH_SIZE
    new_shape_must_be_divisible_by = cf.new_shape_must_be_divisible_by
    preprocess_fn = cf.preprocess_fn

    tr_keys, val_keys = cf.get_split(fold)
    val_keys.sort()

    dataset = cf.dataset
    val_patients = {i: dataset[i] for i in val_keys}


    net.pred_proba(np.random.random((BATCH_SIZE, cf.num_input_channels, 11, 384, 352)).astype(np.float32),
                   not do_bayesian) # preallocate gpu memory
    run_validation(val_patients, net, do_mirroring, do_bayesian, num_repeats, validation_folder, plot_segmentation,
                   save_segmentation, BATCH_SIZE, new_shape_must_be_divisible_by, preprocess_fn, cf.min_size)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="fold", type=int)
    parser.add_argument("-c", help="config file", type=str)
    args = parser.parse_args()
    run(args.c, args.f)