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
import numpy as np
import cPickle
import lasagne
import theano
import os
import sys
sys.path.append("../")
import theano.tensor
import SimpleITK as sitk
from utils import predict_patient_2D_net, get_split, softmax_helper, resize_softmax_output
import imp
from preprocess_test_set import generate_patient_info, preprocess
from dataset_utils import resize_image


def predict_test(pred_fn, results_out_folder, test_keys, dataset_root_raw, BATCH_SIZE=None, n_repeats=1, min_size=None,
                 input_img_must_be_divisible_by=16, do_mirroring=True, preprocess_fn=lambda x:x,
                 target_spacing=(None, 1.25, 1.25)):
    patient_info = generate_patient_info(dataset_root_raw)
    for patient_id in test_keys:
        print patient_id
        if not os.path.isdir(results_out_folder):
            os.mkdir(results_out_folder)

        ed_image = sitk.ReadImage(
            os.path.join(dataset_root_raw, "patient%03.0d" % patient_id,
                         "patient%03.0d_frame%02.0d.nii.gz" % (patient_id, patient_info[patient_id]["ed"])))
        es_image = sitk.ReadImage(
            os.path.join(dataset_root_raw, "patient%03.0d" % patient_id,
                         "patient%03.0d_frame%02.0d.nii.gz" % (patient_id, patient_info[patient_id]["es"])))
        old_spacing = np.array(ed_image.GetSpacing())[[2, 1, 0]]

        target_spacing[0] = old_spacing[0]

        ed_data = resize_image(sitk.GetArrayFromImage(ed_image).astype(float), old_spacing, target_spacing, 3).astype(
            np.float32)
        es_data = resize_image(sitk.GetArrayFromImage(es_image).astype(float), old_spacing, target_spacing, 3).astype(
            np.float32)

        _, _, seg_pred_softmax_es = predict_patient_2D_net(pred_fn, es_data, do_mirroring, n_repeats,
                                                           BATCH_SIZE, input_img_must_be_divisible_by, preprocess_fn,
                                                           min_size)
        _, _, seg_pred_softmax_ed = predict_patient_2D_net(pred_fn, ed_data, do_mirroring, n_repeats,
                                                           BATCH_SIZE,
                                                           input_img_must_be_divisible_by,
                                                           preprocess_fn, min_size)

        np.savez_compressed(
            os.path.join(results_out_folder, "patient%03.0d_2D_net.npz" % patient_id), es=seg_pred_softmax_es,
            ed=seg_pred_softmax_ed)


def run(config_file, fold=0):
    cf = imp.load_source('cf', config_file)
    print fold
    # this is seeded, will be identical each time
    test_keys = range(101, 151)

    experiment_name = cf.EXPERIMENT_NAME
    results_folder = os.path.join(cf.results_dir,  "fold%d/"%fold)
    dataset_root_raw = cf.dataset_root_test

    BATCH_SIZE = cf.BATCH_SIZE
    n_repeats = cf.val_num_repeats

    x_sym = cf.x_sym

    nt, net, seg_layer = cf.nt, cf.net, cf.seg_layer
    output_layer = seg_layer

    test_out_folder = cf.test_out_folder
    if not os.path.isdir(test_out_folder):
        os.mkdir(test_out_folder)
    test_out_folder = os.path.join(test_out_folder, "fold%d"%fold)
    if not os.path.isdir(test_out_folder):
        os.mkdir(test_out_folder)

    with open(os.path.join("../",results_folder, "%s_Params.pkl" % (experiment_name)), 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer, params)

    print "compiling theano functions"
    output = softmax_helper(lasagne.layers.get_output(output_layer, x_sym, deterministic=not cf.val_bayesian_prediction,
                                                      batch_norm_update_averages=False, batch_norm_use_averages=False))
    pred_fn = theano.function([x_sym], output)
    _ = pred_fn(np.random.random((BATCH_SIZE, 1, 384, 352)).astype(np.float32))
    predict_test(pred_fn, test_out_folder, test_keys, dataset_root_raw, BATCH_SIZE=BATCH_SIZE,
               n_repeats=n_repeats, min_size=cf.min_size,
               input_img_must_be_divisible_by=cf.val_input_img_must_be_divisible_by, do_mirroring=cf.val_do_mirroring,
               target_spacing=list(cf.target_spacing), preprocess_fn=preprocess)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="fold", type=int)
    parser.add_argument("-c", help="config file", type=str)
    args = parser.parse_args()
    run(args.c, args.f)

