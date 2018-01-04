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


from batchgenerators.dataloading import DataLoaderBase
import numpy as np
from batchgenerators.augmentations.utils import resize_image_by_padding, random_crop_3D_image_batched, \
    random_crop_2D_image_batched, center_crop_3D_image_batched

class BatchGenerator(DataLoaderBase):
    def __init__(self, data, BATCH_SIZE, PATCH_SIZE=(12, 256, 256), num_batches=None, seed=False, random_crop=True):
        self.PATCH_SIZE = PATCH_SIZE
        self._random_crop = random_crop
        DataLoaderBase.__init__(self, data, BATCH_SIZE, num_batches=num_batches, seed=seed)

    def generate_train_batch(self):
        data = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1], self.PATCH_SIZE[2]),
                        dtype=np.float32)
        seg = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1], self.PATCH_SIZE[2]),
                       dtype=np.float32)
        types = np.random.choice(['ed', 'es'], self.BATCH_SIZE, True)
        patients = np.random.choice(self._data.keys(), self.BATCH_SIZE, True)
        pathologies = []
        for nb in range(self.BATCH_SIZE):
            if np.any(np.array(self._data[patients[nb]][types[nb]+'_data'].shape) < np.array(self.PATCH_SIZE)):
                shp = self._data[patients[nb]][types[nb]+'_data'].shape
                tmp_data = resize_image_by_padding(self._data[patients[nb]][types[nb]+'_data'], (max(shp[0],
                                         self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1]), max(shp[2],
                                         self.PATCH_SIZE[2])), pad_value=0)
                tmp_seg = resize_image_by_padding(self._data[patients[nb]][types[nb]+'_gt'], (max(shp[0],
                                         self.PATCH_SIZE[0]), max(shp[1], self.PATCH_SIZE[1]), max(shp[2],
                                         self.PATCH_SIZE[2])), pad_value=0)
            else:
                tmp_data = self._data[patients[nb]][types[nb]+'_data']
                tmp_seg = self._data[patients[nb]][types[nb]+'_gt']
            # not the most efficient way but whatever...
            tmp = np.zeros((1, 2, tmp_data.shape[0], tmp_data.shape[1], tmp_data.shape[2]))
            tmp[0, 0] = tmp_data
            tmp[0, 1] = tmp_seg
            if self._random_crop:
                tmp = random_crop_3D_image_batched(tmp, self.PATCH_SIZE)
            else:
                tmp = center_crop_3D_image_batched(tmp, self.PATCH_SIZE)
            data[nb, 0] = tmp[0, 0]
            seg[nb, 0] = tmp[0, 1]
            pathologies.append(self._data[patients[nb]]['pathology'])
        return {'data':data, 'seg':seg, 'types':types, 'patient_ids': patients, 'pathologies':pathologies}


class BatchGenerator_2D(DataLoaderBase):
    def __init__(self, data, BATCH_SIZE, PATCH_SIZE=(256, 256), num_batches=None, seed=False):
        self.PATCH_SIZE = PATCH_SIZE
        DataLoaderBase.__init__(self, data, BATCH_SIZE, num_batches=num_batches, seed=seed)

    def generate_train_batch(self):
        data = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)
        seg = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)
        types = np.random.choice(['ed', 'es'], self.BATCH_SIZE, True)
        patients = np.random.choice(self._data.keys(), self.BATCH_SIZE, True)
        pathologies = []
        for nb in range(self.BATCH_SIZE):
            shp = self._data[patients[nb]][types[nb]+'_data'].shape
            slice_id = np.random.choice(shp[0])
            tmp_data = resize_image_by_padding(self._data[patients[nb]][types[nb]+'_data'][slice_id], (max(shp[1],
                                                self.PATCH_SIZE[0]), max(shp[2], self.PATCH_SIZE[1])), pad_value=0)
            tmp_seg = resize_image_by_padding(self._data[patients[nb]][types[nb]+'_gt'][slice_id], (max(shp[1],
                                                self.PATCH_SIZE[0]), max(shp[2], self.PATCH_SIZE[1])), pad_value=0)

            # not the most efficient way but whatever...
            tmp = np.zeros((1, 2, tmp_data.shape[0], tmp_data.shape[1]))
            tmp[0, 0] = tmp_data
            tmp[0, 1] = tmp_seg
            tmp = random_crop_2D_image_batched(tmp, self.PATCH_SIZE)
            data[nb, 0] = tmp[0, 0]
            seg[nb, 0] = tmp[0, 1]
            pathologies.append(self._data[patients[nb]]['pathology'])
        return {'data':data, 'seg':seg, 'types':types, 'patient_ids': patients, 'pathologies':pathologies}
