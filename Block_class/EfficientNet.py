import warnings

warnings.filterwarnings(action='ignore')

from tensorflow.keras import layers
import tensorflow as tf

import collections
import math

Global_params = collections.namedtuple('Global_params', [
    'width_coefficient', 'depth_coefficient', 'dropout_rate',
    'batch_norm','batch_norm_momentum', 'batch_norm_epsilon', 'activation'
])
Block_params = collections.namedtuple('Block_params', [
    'channels', 'output_filters','kernel_size', 'strides', 'padding',
    'expand_ratio', 'se_ratio', 'num_repeat'
])

CONV_KERNEL_INITIALIZER = {
    'class_name' : 'VarianceScaling',
    'config' : {
        'scale' : 2.0,
        'mode' : 'fan_out',
        'distribution' : 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name' : 'VarianceScaling',
    'config' : {
        'scale' : 1.0 / 3.0,
        'mode' : 'fan_out',
        'distribution' : 'uniform'
    }
}

Block_params_list = [
    Block_params(channels=32, output_filters=16, kernel_size = 3, strides=1, padding='same',
                 expand_ratio=0, se_ratio=0.25,num_repeat=1),
    Block_params(channels=16, output_filters=24, kernel_size = 3, strides=2, padding='same',
                expand_ratio=6, se_ratio=0.25,num_repeat=2),
    Block_params(channels=24, output_filters=40, kernel_size = 5, strides=2, padding='same',
                expand_ratio=6, se_ratio=0.25, num_repeat=2),
    Block_params(channels=40, output_filters=80, kernel_size = 3, strides=2, padding='same',
                expand_ratio=6, se_ratio=0.25,num_repeat=3),
    Block_params(channels=80, output_filters=112, kernel_size = 5, strides=1, padding='same',
                expand_ratio=6, se_ratio=0.25,num_repeat=3),
    Block_params(channels=112, output_filters=192, kernel_size = 5, strides=2, padding='same',
                expand_ratio=6, se_ratio=0.25,num_repeat=4),
    Block_params(channels=192, output_filters=320, kernel_size = 3, strides=1, padding='same',
                expand_ratio=6, se_ratio=0.25,num_repeat=1)
]

global_param = Global_params(width_coefficient=1.0, depth_coefficient=1.0, dropout_rate = 0.2,
                             batch_norm = True,batch_norm_momentum=0.99, batch_norm_epsilon=0.001,
                             activation='swish')


def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters =int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(repeats * depth_coefficient))

class MBConvBlock:

    def __init__(self, global_params, block_params):
        '''

        :param block_params: 'Block_parmas', the parameters of tensorflow layers
        :param global_params: 'Global_params'.
        '''
        self._dropout_rate = global_params.dropout_rate
        self._batch_norm = global_params.batch_norm
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        self._activation = global_params.activation

        self._expand_ratio = block_params.expand_ratio
        self._se_ratio = block_params.se_ratio
        self._num_repeat = block_params.num_repeat
        self._channels = block_params.channels
        self._output_filters = block_params.output_filters
        self._kernel_size = block_params.kernel_size
        self._strides = block_params.strides
        self._padding = block_params.padding
        self.after_expand_filters = None
        self.dropout = layers.Dropout(rate = self._dropout_rate)

    def mbconv_block(self, input_tensor):
        # Expansion phase
        x = input_tensor


        if self._expand_ratio != 0:
            self.filters_expand = self._channels * self._expand_ratio
            x = layers.Conv2D(self.filters_expand,
                              kernel_size=(1, 1),
                              strides=1,
                              padding='same',
                              kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
            if self._batch_norm:
                x = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                                epsilon=self._batch_norm_epsilon)(x)
            x = layers.Activation(self._activation)(x)
        else:
            self.filters_expand = self._channels


        # Depthwise conv block
        x = layers.DepthwiseConv2D(kernel_size = self._kernel_size,
                                 strides = self._strides,
                                 padding =self._padding,
                                 depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                 use_bias=False)(x)
        if self._batch_norm:
            x = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                                epsilon=self._batch_norm_epsilon)(x)
        x = layers.Activation(self._activation)(x)

        # SE Block
        x = self.se_block(x)

        # Output phase
        x = layers.Conv2D(self._output_filters,
                              kernel_size=(1, 1),
                              strides=1,
                              padding='same',
                              kernel_initializer=CONV_KERNEL_INITIALIZER
                              )(x)
        if self._batch_norm:
            x = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                                epsilon=self._batch_norm_epsilon)(x)

        return x

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
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          use_bias=True)(x)
        x = layers.Conv2D(self.filters_expand,
                          kernel_size=(1, 1),
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          use_bias=True)(x)
      
        x = layers.multiply([input_tensor, x])

        return x

    def out(self, input_tensors):
        '''

        :param input_tensors: 'tensor', a previous tensors
        :return: an output tensors
        '''

        x = input_tensors

        for i in range(self._num_repeat):

            if i == 0:
                x = self.mbconv_block(x)
            else:
                self._channels = self._output_filters
                self._strides = 1
                x_indentity = tf.identity(x)
                x = self.mbconv_block(x)
                x = self.dropout(x)
                x = layers.add([x_indentity, x])

        return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_resolution,
                 dropout_rate = 0.2,
                 depth_divisor = 8,
                 classes = 10,
                 global_params = global_param,
                 block_param_list = Block_params_list,
                 ):
    '''

    :param width_coefficient:
    :param depth_coefficient:
    :param default_resolution:
    :param dropout_rate:
    :param depth_divisor:
    :param input_shape:
    :return:
    '''

    input = layers.Input(shape = (224, 224, 1))
    x = layers.Conv2D(32,
                      kernel_size=(3,3),
                      strides = 2,
                      padding = 'valid',
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(input)
    if global_params.batch_norm:
        x = layers.BatchNormalization(momentum=global_params.batch_norm_momentum,
                                      epsilon=global_params.batch_norm_epsilon
                                      )(x)
    x = layers.Activation(global_params.activation)(x)

    for i in range(len(Block_params_list)):
        args = block_param_list[i]
        args = args._replace(channels=round_filters(args.channels, width_coefficient, depth_divisor),
                             output_filters=round_filters(args.output_filters, width_coefficient, depth_divisor),
                             num_repeat=round_repeats(args.num_repeat, depth_coefficient))

        x = MBConvBlock(global_param, args).out(x)

    x = layers.Conv2D(1280,
                      kernel_size=(1, 1),
                      strides = 1,
                      padding='same',
                      kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    if global_params.batch_norm:
        x = layers.BatchNormalization(momentum=global_params.batch_norm_momentum,
                                      epsilon=global_params.batch_norm_epsilon
                                      )(x)
        
    x = layers.Activation(global_params.activation)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(global_params.dropout_rate)(x)
    predictions = layers.Dense(classes,
                     activation='softmax',
                     kernel_initializer=DENSE_KERNEL_INITIALIZER,
                     use_bias=True)(x)

    model = tf.keras.Model(inputs = input, outputs = predictions)

    return model

efficientnet = EfficientNet(1.0,1.1,28)
efficientnet.summary()