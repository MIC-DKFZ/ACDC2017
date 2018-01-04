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
import lasagne
import os
import sys
sys.path.append("../")
import SimpleITK as sitk
from utils import predict_patient_3D_net, get_split, softmax_helper, resize_softmax_output
import imp
from preprocess_test_set import generate_patient_info, preprocess
from dataset_utils import resize_image


def predict_test(net, results_out_folder, test_keys, dataset_root_raw, BATCH_SIZE=None, n_repeats=1, min_size=None,
                 input_img_must_be_divisible_by=16, do_mirroring=True, preprocess_fn=lambda x:x,
                 target_spacing=(10, 1.25, 1.25), bayesian_prediction=True):
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


        ed_data = resize_image(sitk.GetArrayFromImage(ed_image).astype(float), old_spacing, target_spacing, 1).astype(
            np.float32)
        es_data = resize_image(sitk.GetArrayFromImage(es_image).astype(float), old_spacing, target_spacing, 1).astype(
            np.float32)
        _, _, seg_pred_softmax_es = predict_patient_3D_net(net, es_data, do_mirroring, bayesian_prediction, n_repeats,
                                                           BATCH_SIZE, input_img_must_be_divisible_by, preprocess_fn,
                                                           min_size)
        _, _, seg_pred_softmax_ed = predict_patient_3D_net(net, ed_data, do_mirroring, bayesian_prediction, n_repeats,
                                                           BATCH_SIZE,
                                                           input_img_must_be_divisible_by,
                                                           preprocess_fn, min_size)

        np.savez_compressed(
            os.path.join(results_out_folder, "patient%03.0d_3D_net.npz" % patient_id), es=seg_pred_softmax_es,
            ed=seg_pred_softmax_ed)


def run(config_file, fold=0):
    cf = imp.load_source('cf', config_file)
    net = cf.get_network(mode='val')

    out_folder = os.path.join(cf.results_dir, "fold%d" % fold)
    try:
        net.load_params(os.path.join("../", out_folder, "best_params2.pkl"))
    except IOError:
        try:
            print "could not load best params, trying latest:"
            net.load_params(os.path.join("../", out_folder, "latest_params.pkl"))
        except IOError:
            raise RuntimeError("Could not open parameters, error message: %s" % sys.exc_info())
    except Exception:
        raise RuntimeError("Exception during loading params: %s" % sys.exc_info())

    net._initialize_pred_seg()

    BATCH_SIZE = cf.BATCH_SIZE
    new_shape_must_be_divisible_by = cf.new_shape_must_be_divisible_by
    n_repeats = cf.num_repeats // 2

    print fold
    dataset_root_raw = cf.dataset_root_test
    test_keys = range(101, 151)

    np.random.seed(65432)
    lasagne.random.set_rng(np.random.RandomState(98765))

    test_out_folder = cf.test_out_folder
    if not os.path.isdir(test_out_folder):
        os.mkdir(test_out_folder)
    test_out_folder = os.path.join(test_out_folder, "fold%d" % fold)
    if not os.path.isdir(test_out_folder):
        os.mkdir(test_out_folder)

    _ = net.pred_proba(np.random.random((BATCH_SIZE, 1, 11, 384, 352)).astype(np.float32), False)

    predict_test(net, test_out_folder, test_keys, dataset_root_raw, BATCH_SIZE=BATCH_SIZE, n_repeats=n_repeats,
               min_size=cf.min_size, input_img_must_be_divisible_by=new_shape_must_be_divisible_by,
               do_mirroring=cf.do_mirroring, target_spacing=cf.target_spacing,
                 bayesian_prediction=cf.bayesian_prediction, preprocess_fn=preprocess)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="fold", type=int)
    parser.add_argument("-c", help="config file", type=str)
    args = parser.parse_args()
    run(args.c, args.f)
