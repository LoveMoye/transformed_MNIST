import warnings

warnings.filterwarnings(action='ignore')

import collections
import math

import tensorflow as tf
from tensorflow.keras import layers
Global_params = collections.namedtuple('Global_params', [
    'width_coefficient', 'depth_coefficient', 'dropout_rate',
    'batch_norm_momentum', 'batch_norm_epsilon'
])
Block_params = collections.namedtuple('Block_params', [
    'channels', 'output_filters','kernel_size', 'strides', 'padding',
    'activation', 'batch_norm', 'expand_ratio', 'se_ratio', 'num_repeat', 'kernel_initializer'
])

Block_params.__new__.__defaults__ = (None,) * len(Block_params._fields)


class ConvBlock:

    def __init__(self, block_params):
        '''

        :param block_params: 'Block_params' ,the parameter of tensorflow layers
        '''
        self._channels = block_params.channels
        self._kernel_size = block_params.kernel_size
        self._strides = block_params.strides
        self._padding = block_params.padding
        self._activation = block_params.activation
        self._batch_norm = block_params.batch_norm
        self._batch_norm_momentum = block_params.batch_norm_momentum
        self._batch_norm_epsilon = block_params.batch_norm_epsilon

        self.make_block()

    def make_block(self):
        self._conv_layer = layers.Conv2D(self._channels,
                                         kernel_size=self._kernel_size,
                                         strides=self._strides,
                                         padding=self._padding,
                                         )

        self._batch_layer = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                                      epsilon=self._batch_norm_epsilon)

        self._activation_layer = layers.Activation(self._activation)

    def out(self, input_tensors):
        '''

        :param input_tensors: previous_tensor
        :return: output_tensor
        '''

        x = input_tensors

        x = self._conv_layer(x)
        if self._batch_norm:
            x = self._batch_layer(x)

        if self._activation is not None:
            x = self._activation_layer(x)

        return x

    def out_without_activation(self, input_tensors):
        '''

        :param input_tensors: previous_tensor
        :return: output_tensor
        '''

        x = input_tensors

        x = self._conv_layer(x)
        if self._batch_norm:
            x = self._batch_layer(x)

        return x


class Residual_Block:

    def __init__(self, block_params):
        '''

        :param block_params: 'Block_params' ,the parameter of tensorflow layers
        '''
        self._block_params = block_params
        self._activation = block_params.activation
        self._batch_norm = block_params.batch_norm
        self._batch_norm_momentum = block_params.batch_norm_momentum
        self._batch_norm_epsilon = block_params.batch_norm_epsilon

        self.make_block()

    def make_block(self):
        self.conv_layer_1 = ConvBlock(self._block_params)
        self.conv_layer_2 = ConvBlock(self._block_params)
        self.bn_layer = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                                  epsilon=self._batch_norm_epsilon)
        self.activation_layer = layers.Activation(self._activation)

    def out(self, input_tensor):
        '''

        :param input_tensor: 'tensor', input_tensor (Conv_Block)
        :return: 'tensor', output_tensor
        '''

        x = input_tensor

        x = self.conv_layer_1.out(x)
        x = self.conv_layer_2.out_without_activation(x)
        x = layers.add([input_tensor, x])
        if self._batch_norm:
            x = self.bn_layer(x)
        x = self.activation_layer(x)

        return x


class MBConvBlock:

    def __init__(self, global_params, block_params):
        '''

        :param block_params: 'Block_parmas', the parameters of tensorflow layers
        :param global_params: 'Global_params'.
        '''
        self._dropout_rate = global_params.dropout_rate
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon

        self._expand_ratio = block_params.expand_ratio
        self._se_ratio = block_params.se_ratio
        self._num_repeat = block_params.num_repeat
        self._channels = block_params.channels
        self._output_filters = block_params.output_filters
        self._kernel_size = block_params.kernel_size
        self._strides = block_params.strides
        self._padding = block_params.padding
        self._activation = block_params.activation
        self._batch_norm = block_params.batch_norm
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._kernel_initializer = block_params.kernel_initializer

        self.make_block()

    def make_block(self):

        # Expansion phase
        self.filters_expand = self._channels * self._expand_ratio
        self.conv_expand = layers.Conv2D(self.filters_expand,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    )
        self.bn_expand = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                                   epsilon=self._batch_norm_epsilon)
        self.act_expand = layers.Activation(self._activation)

        # Depthwise Conv block
        self.conv_depthwise = layers.DepthwiseConv2D(kernel_size = self._channels,
                                                     strides = self._strides,
                                                     padding =self._padding
                                                     )
        self.bn_depthwise = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                                        epsilon=self._batch_norm_epsilon)
        self.act_depthwise = layers.Activation(self._activation)

        # Output phase
        self.conv_out = layers.Conv2D(self._output_filters,
                                      kernel_size=(1, 1),
                                      strides=1,
                                      padding='same'
                                      )
        self.bn_out = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                                epsilon=self._batch_norm_epsilon)

        self.dropout = layers.Dropout(rate = self._dropout_rate)

    def mbconv_block(self, input_tensor):
        # Expansion phase
        x = input_tensor

        x = self.conv_expand(x)
        if self._batch_norm:
            x = self.bn_expand(x)
        x = self.act_expand(x)

        # Depthwise conv block
        x = self.conv_depthwise(x)
        if self._batch_norm:
            x = self.bn_depthwise(x)
        x = self.act_expand(x)

        # SE Block
        x = self.se_block(x)

        # Output phase
        x = self.conv_out(x)
        if self._batch_norm:
            x = self.bn_out(x)

    def se_block(self, input_tensor):

        x = input_tensor
        # squeeze phase
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1,1,-1))(x)

        # reduce phase
        reduced_filters = max(1, (self._channels * self._se_ratio))
        x = layers.Conv2D(reduced_filters,
                          kernel_size = (1, 1),
                          strides = 1,
                          padding = 'same',
                          activation='swish',
                          use_bias=True)
        x = layers.Conv2D(self.filters_expand,
                          kernel_size=(1, 1),
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          use_bias=True)
        x = layers.Multiply([input_tensor, x])

        return x

    def out(self, input_tensors):
        '''

        :param input_tensors: 'tensor', a previous tensors
        :return: an output tensors
        '''

        x = input_tensors

        for i in range(self._num_repeat):
            if i == 0:
                x = self.make_block(x)
            else:
                x_indentity = tf.identity(x)
                x = self.make_block(x)
                x = layers.Add([x_indentity, x])
                if self._batch_norm:
                    x = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                                    epsilon=self._batch_norm_epsilon)(x)

        return x


def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters =int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(repeats * depth_coefficient))









