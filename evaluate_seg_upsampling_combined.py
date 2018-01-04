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
from dataset_utils import load_dataset, generate_patient_info
import os
from collections import OrderedDict
from medpy import metric
from skimage.transform import resize


dataset_base_dir = "/media/fabian/My Book/datasets/ACDC/training/" # dataset as downloaded from the website
results_folder_3D = "results/UNet3D_final"
results_folder_2D = "results/results_segmentation_2D_new_dataset_ds_tmp_"
patient_metrics = OrderedDict()
patient_info = generate_patient_info(dataset_base_dir)
output_folder = "predicted_segmentations_train"
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)


for fold in range(5):
    validation_folder_3D = os.path.join(results_folder_3D, "fold%d" % fold, "validation")
    validation_folder_2D = os.path.join(results_folder_2D + "fold%d" % fold, "ep299_MA", "val_mirror")
    patients_in_this_fold = [int(i) for i in os.listdir(validation_folder_2D) if
                             os.path.isdir(os.path.join(validation_folder_2D, i))]
    for patient in patients_in_this_fold:
        patient_metrics[patient] = {}
        patient_dir_2D = os.path.join(validation_folder_2D, "%03.0d"%patient)
        patient_dir_3D = os.path.join(validation_folder_3D, "%03.0d"%patient)
        for tpe in ['ed', 'es']:
            patient_metrics[patient][tpe] = {}
            gt = sitk.ReadImage(os.path.join(dataset_base_dir, "patient%03.0d"%patient,
                                         "patient%03.0d_frame%02.0d_gt.nii.gz" % (patient, patient_info[patient][tpe])))
            spacing_gt = np.array(gt.GetSpacing())[[2, 1, 0]]
            gt = sitk.GetArrayFromImage(gt)

            # seg 3D
            segs = np.load(os.path.join(patient_dir_3D, "gt_and_pred_segm_%s.npz" % tpe))
            seg_pred = segs['softmax_pred']
            # spacing of seg_pred is (10, 1.25, 1.25)
            seg_resized_3D = []
            for c in range(0, 4):
                seg_resized_3D.append(resize((seg_pred[c]).astype(float), gt.shape, 3, 'edge', 0)[None])
            seg_3D = np.vstack(seg_resized_3D).argmax(0)

            # seg 2D
            segs = np.load(os.path.join(patient_dir_2D, "gt_and_pred_segm_%s.npz" % tpe))
            seg_pred = segs['softmax_pred']
            # spacing of seg_pred is (10, 1.25, 1.25)
            seg_resized_2D = []
            for c in range(0, 4):
                seg_resized_2D.append(resize((seg_pred[c]).astype(float), gt.shape, 3, 'edge', 0)[None])
            seg_2D = np.vstack(seg_resized_2D).argmax(0)

            global_avg = np.mean(np.vstack((np.vstack(seg_resized_3D)[None], np.vstack(seg_resized_2D)[None])), 0)
            seg_resized = np.argmax(global_avg, 0)
            print "combined"
            for c in range(1, 4):
                patient_metrics[patient][tpe]["combined_"+str(c)] = metric.dc(gt == c, seg_resized == c)
                print metric.dc(gt == c, seg_resized == c)
            print "2d"
            for c in range(1, 4):
                patient_metrics[patient][tpe]["2d_"+str(c)] = metric.dc(gt == c, seg_2D == c)
                print metric.dc(gt == c, seg_2D == c)
            print "3d"
            for c in range(1, 4):
                patient_metrics[patient][tpe]["3d_"+str(c)] = metric.dc(gt == c, seg_3D == c)
                print metric.dc(gt == c, seg_3D == c)



all_RV_dc_combined = []
all_MLV_dc_combined = []
all_LVC_dc_combined = []
all_RV_dc_2d = []
all_MLV_dc_2d = []
all_LVC_dc_2d = []
all_RV_dc_3d = []
all_MLV_dc_3d = []
all_LVC_dc_3d = []
for p in patient_metrics.values():
    for tp in p.values():
        all_RV_dc_combined.append(tp["combined_"+str(1)])
        all_MLV_dc_combined.append(tp["combined_"+str(2)])
        all_LVC_dc_combined.append(tp["combined_"+str(3)])
        all_RV_dc_2d.append(tp["2d_"+str(1)])
        all_MLV_dc_2d.append(tp["2d_"+str(2)])
        all_LVC_dc_2d.append(tp["2d_"+str(3)])
        all_RV_dc_3d.append(tp["3d_"+str(1)])
        all_MLV_dc_3d.append(tp["3d_"+str(2)])
        all_LVC_dc_3d.append(tp["3d_"+str(3)])

print np.round(np.mean(all_RV_dc_combined), 3), np.round(np.mean(all_MLV_dc_combined), 3), \
    np.round(np.mean(all_LVC_dc_combined), 3)
print np.round(np.mean(all_RV_dc_2d), 3), np.round(np.mean(all_MLV_dc_2d), 3), np.round(np.mean(all_LVC_dc_2d), 3)
print np.round(np.mean(all_RV_dc_3d), 3), np.round(np.mean(all_MLV_dc_3d), 3), np.round(np.mean(all_LVC_dc_3d), 3)

rv_dc_ed = all_RV_dc_combined[0::2]
rv_dc_es = all_RV_dc_combined[1::2]
lv_dc_ed = all_LVC_dc_combined[0::2]
lv_dc_es = all_LVC_dc_combined[1::2]
mlv_dc_ed = all_MLV_dc_combined[0::2]
mlv_dc_es = all_MLV_dc_combined[1::2]

diseases = ['DCM', 'HCM', 'MINF', 'NOR', 'RV']
for i, d in enumerate(diseases):
    print d
    print "rv ed: ", np.round(np.mean(rv_dc_ed[i * 20: (i+1) * 20]), 3)
    print "rv es: ", np.round(np.mean(rv_dc_es[i * 20: (i+1) * 20]), 3)
    print "mean rv:", np.round((np.mean(rv_dc_ed[i * 20: (i+1) * 20]) + np.mean(rv_dc_es[i * 20: (i+1) * 20])) / 2., 3)
    print "mlv ed: ", np.round(np.mean(mlv_dc_ed[i * 20: (i+1) * 20]), 3)
    print "mlv es: ", np.round(np.mean(mlv_dc_es[i * 20: (i+1) * 20]), 3)
    print "mean mlv:", np.round((np.mean(mlv_dc_ed[i * 20: (i+1) * 20]) +
                                 np.mean(mlv_dc_es[i * 20: (i+1) * 20])) / 2., 3)
    print "lvc ed: ", np.round(np.mean(lv_dc_ed[i * 20: (i+1) * 20]), 3)
    print "lvc es: ", np.round(np.mean(lv_dc_es[i * 20: (i+1) * 20]), 3)
    print "mean lvc:", np.round((np.mean(lv_dc_ed[i * 20: (i+1) * 20]) + np.mean(lv_dc_es[i * 20: (i+1) * 20])) / 2., 3)

