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


import numpy as np
import SimpleITK as sitk
from preprocess_test_set import generate_patient_info
import os
import sys
sys.path.append("../")
from utils import postprocess_prediction
from skimage.transform import resize
import imp


def run(config_file_2d, config_file_3d, output_folder):
    cf_2d = imp.load_source("cf_2d", config_file_2d)
    cf_3d = imp.load_source("cf_3d", config_file_3d)

    dataset_base_dir = cf_2d.dataset_root_test
    results_folder_3D = os.path.join(cf_3d.results_dir, "test_predictions/")
    results_folder_2D = os.path.join(cf_2d.results_dir, "test_predictions/")
    patient_info = generate_patient_info(dataset_base_dir)

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)


    def resize_softmax_pred(softmax_output, new_shape, order=3):
        reshaped = np.zeros([len(softmax_output)] + list(new_shape), dtype=float)
        for i in range(len(softmax_output)):
            reshaped[i] = resize(softmax_output[i].astype(float), new_shape, order, mode="constant", cval=0, clip=True)
        return reshaped

    for patient in range(101, 151):
        for tpe in ['ed', 'es']:
            all_softmax = []
            raw_itk = sitk.ReadImage(os.path.join(dataset_base_dir, "patient%03.0d"%patient,
                                                  "patient%03.0d_frame%02.0d.nii.gz" %
                                                  (patient, patient_info[patient][tpe])))
            raw = sitk.GetArrayFromImage(raw_itk)
            for f in range(5):
                res_3d = np.load(os.path.join(results_folder_3D, "fold%d" % f, "patient%03.0d_3D_net.npz" % patient))
                res_2d = np.load(os.path.join(results_folder_2D, "fold%d" % f, "patient%03.0d_2D_net.npz" % patient))
                # resize softmax to original image size
                softmax_3d = resize_softmax_pred(res_3d[tpe], raw.shape, 3)
                softmax_2d = resize_softmax_pred(res_2d[tpe], raw.shape, 3)

                all_softmax += [softmax_3d[None], softmax_2d[None]]
            predicted_seg = postprocess_prediction(np.vstack(all_softmax).mean(0).argmax(0))

            itk_seg = sitk.GetImageFromArray(predicted_seg.astype(np.uint8))
            itk_seg.CopyInformation(raw_itk)
            sitk.WriteImage(itk_seg, os.path.join(output_folder, "patient%03.0d_%s.nii.gz" % (patient, tpe.upper())))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c2d", help="config file for 2d network", type=str)
    parser.add_argument("-c3d", help="config file for 3d network", type=str)
    parser.add_argument("-o", help="output folder", type=str)
    args = parser.parse_args()
    run(args.c2d, args.c3d, args.o)