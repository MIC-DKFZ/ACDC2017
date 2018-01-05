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



from lasagne.layers import InputLayer, DimshuffleLayer, ReshapeLayer, ConcatLayer, NonlinearityLayer, batch_norm, \
    ElemwiseSumLayer, DropoutLayer, Pool2DLayer, Upscale2DLayer
from collections import OrderedDict
from lasagne.init import HeNormal
from lasagne.nonlinearities import linear
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer


def build_UNet_relu_BN_ds(n_input_channels=1, input_var=None, BATCH_SIZE=None, num_output_classes=2, pad='same',
                          input_dim=(128, 128), base_n_filters=64, dropout=None,
                          nonlinearity=lasagne.nonlinearities.rectify, bn_axes=(2, 3)):
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1]), input_var)

    net['contr_1_1'] = batch_norm(ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)

    net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)

    net['contr_3_1'] = batch_norm(ConvLayer(l, base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)

    net['contr_4_1'] = batch_norm(ConvLayer(l, base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)

    net['encode_1'] = batch_norm(ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad,
                                           W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['encode_2'] = batch_norm(ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity,
                                               pad=pad, W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['upscale1'] = Upscale2DLayer(l, 2)

    l = net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)
    net['expand_1_1'] = batch_norm(ConvLayer(l, base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity,
                                                 pad=pad, W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['upscale2'] = Upscale2DLayer(l, 2)

    l = net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)
    net['expand_2_1'] = batch_norm(ConvLayer(l, base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    ds2 = l = net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], base_n_filters*4, 3,
                                                       nonlinearity=nonlinearity, pad=pad,
                                                       W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['upscale3'] = Upscale2DLayer(l, 2)

    l = net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)
    net['expand_3_1'] = batch_norm(ConvLayer(l, base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['expand_3_2'] = batch_norm(ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity,
                                                 pad=pad, W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['upscale4'] = Upscale2DLayer(l, 2)

    net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = batch_norm(ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['expand_4_2'] = batch_norm(ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)

    net['output_segmentation'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=None)

    ds2_1x1_conv = ConvLayer(ds2, num_output_classes, 1, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    ds1_ds2_sum_upscale = Upscale2DLayer(ds2_1x1_conv, 2)
    ds3_1x1_conv = ConvLayer(net['expand_3_2'], num_output_classes, 1, 1, 'same', nonlinearity=linear,
                             W=HeNormal(gain='relu'))
    ds1_ds2_sum_upscale_ds3_sum = ElemwiseSumLayer((ds1_ds2_sum_upscale, ds3_1x1_conv))
    ds1_ds2_sum_upscale_ds3_sum_upscale = Upscale2DLayer(ds1_ds2_sum_upscale_ds3_sum, 2)

    l = seg_layer = ElemwiseSumLayer((net['output_segmentation'], ds1_ds2_sum_upscale_ds3_sum_upscale))

    net['dimshuffle'] = DimshuffleLayer(l, (0, 2, 3, 1))
    batch_size, n_rows, n_cols, _ = lasagne.layers.get_output(net['dimshuffle']).shape
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (batch_size * n_rows * n_cols, num_output_classes))
    net['output_flattened'] = NonlinearityLayer(net['reshapeSeg'], nonlinearity=lasagne.nonlinearities.softmax)

    return net, net['output_flattened'], seg_layer

