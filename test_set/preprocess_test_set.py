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


import os
import numpy as np


def preprocess(data):
    return (data - data.mean()) / data.std()


def generate_patient_info(folder):
    patient_info={}
    for id in range(101, 151):
        fldr = os.path.join(folder, 'patient%03.0d'%id)
        if not os.path.isdir(fldr):
            print "could not find dir of patient ", id
            continue
        nfo = np.loadtxt(os.path.join(fldr, "Info.cfg"), dtype=str, delimiter=': ')
        patient_info[id] = {}
        patient_info[id]['ed'] = int(nfo[0, 1])
        patient_info[id]['es'] = int(nfo[1, 1])
        patient_info[id]['height'] = float(nfo[2, 1])
        patient_info[id]['weight'] = float(nfo[4, 1])
    return patient_info
