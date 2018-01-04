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


import sys
import lasagne
import cPickle
from abc import abstractmethod
import theano

floatX = theano.config.floatX

class NetworkArchitecture(object):
    def __init__(self):
        self.input_layer = self.output_layer = self.input_var = self.output_var = None
        self.build_network()
        assert self.input_layer is not None, "build_network() must define self.input_layer"
        assert self.output_layer is not None, "build_network() must define self.output_layer"
        assert self.input_var is not None, "build_network() must define self.input_var"
        assert self.output_var is not None, "build_network() must define self.output_var"

    @abstractmethod
    def build_network(self):
        pass

    def save_params(self, fname, **kwargs):
        with open(fname, 'w') as f:
            cPickle.dump(lasagne.layers.get_all_param_values(self.output_layer, **kwargs), f)

    def load_params(self, fname):
        with open(fname, 'r') as f:
            lasagne.layers.set_all_param_values(self.output_layer, cPickle.load(f))

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.output_layer, params)

    def plot_architecture(self, fname):
        try:
            from Utils.draw_net import draw_to_file
            draw_to_file(lasagne.layers.get_all_layers(self.output_layer), fname)
        except ImportError:
            print "Could not plot network architecture due to import error: ", sys.exc_info()
        except IOError:
            print "Could not plot network due to invalid file name %s" % fname, " error: ", sys.exc_info()
        except:
            print "Could not plot network due to error: ", sys.exc_info()

class SegmentationArchitecture(NetworkArchitecture):
    def __init__(self):
        self.seg_layer = None
        NetworkArchitecture.__init__(self)
        assert self.seg_layer is not None, "SegmentationNetworkArchitecture.build_network() must define self.seg_layer"
