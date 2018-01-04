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
from batchgenerators.transforms import AbstractTransform

def motion_augmentation(data, seg=None, p_augm=0.1, mu=0, sigma=10):
    for b in range(data.shape[0]):
        for z in range(data.shape[2]):
            if np.random.random() < p_augm:
                offset = np.round(np.random.normal(mu, sigma, 2)).astype(int) # dont want to interpolate again!
                new_slice = np.zeros(data.shape[3:5], dtype=np.float32)
                if seg is not None:
                    new_slice_seg = np.zeros(seg.shape[3:5], dtype=np.int32)
                if offset[0] < 0:
                    offset[0] = np.abs(offset[0])
                    new_slice[offset[0]:, :] = data[b, 0, z, :data.shape[3] - offset[0], :]
                    if seg is not None:
                        new_slice_seg[offset[0]:, :] = seg[b, 0, z, :seg.shape[3] - offset[0], :]
                elif offset[0] > 0:
                    new_slice[:data.shape[3] - offset[0], :] = data[b, 0, z, offset[0]:, :]
                    if seg is not None:
                        new_slice_seg[:seg.shape[3] - offset[0], :] = seg[b, 0, z,offset[0]:, :]
                if offset[1] < 0:
                    offset[1] = np.abs(offset[1])
                    new_slice[:, offset[1]:] = data[b, 0, z, :, :data.shape[4] - offset[1]]
                    if seg is not None:
                        new_slice_seg[:, offset[1]:] = seg[b, 0, z, :, :seg.shape[4] - offset[1]]
                elif offset[1] > 0:
                    new_slice[:, :data.shape[4] - offset[1]] = data[b, 0, z, :, offset[1]:]
                    if seg is not None:
                        new_slice_seg[:, :seg.shape[4] - offset[1]] = seg[b, 0, z, :, offset[1]:]
                data[b, 0, z] = new_slice
                if seg is not None:
                    seg[b, 0, z] = new_slice_seg
        return data, seg


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict, shape=(10, 224, 224)):
    shp = data_dict['orig_shape_data']
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shape[0], shape[1], shape[2]))
    shp = data_dict['orig_shape_seg']
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shape[0], shape[1], shape[2]))
    return data_dict


class MotionAugmentationTransform(AbstractTransform):
    def __init__(self, p_augm, mu, sigma):
        self.sigma = sigma
        self.mu = mu
        self.p_augm = p_augm

    def __call__(self, **data_dict):
        data = data_dict.get("data")
        seg = data_dict.get("seg")
        data, seg = motion_augmentation(data, seg, self.p_augm, self.mu, self.sigma)
        data_dict['data'] = data
        if seg is not None:
            data_dict['seg'] = seg
        return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self, original_shape):
        self.original_shape = original_shape

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict, self.original_shape)