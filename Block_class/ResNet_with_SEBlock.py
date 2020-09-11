import warnings

warnings.filterwarnings(action='ignore')

import collections

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

Global_params = collections.namedtuple('Global_params', [
    'activation', 'batch_norm'
])

Block_params = collections.namedtuple('Block_params', [
    'channels', 'kernel_size', 'strides', 'padding', 'se_ratio', 'block_number', 'repeat_number'
])

Block_params.__new__.__defaults__ = (None,) * len(Block_params._fields)

global_params = Global_params(activation = 'relu', batch_norm=False)

Block_params_list = [
    Block_params(channels = 128, kernel_size = (3,3), strides = 1, padding = 'same', se_ratio = 0.25, block_number=2, repeat_number=0),
    Block_params(channels = 256, kernel_size = (3,3), strides = 1, padding = 'same', se_ratio = 0.25, block_number=3, repeat_number=0),
    Block_params(channels = 256, kernel_size = (3,3), strides = 1, padding = 'same', se_ratio = 0.25, block_number=4, repeat_number=0)
]



class Residual_Block:

    def __init__(self, global_params, block_params):
        '''

        :param block_params: 'Block_params' ,the parameter of tensorflow layers
        '''
        self._activation = global_params.activation
        self._batch_norm = global_params.batch_norm

        self._se_ratio = block_params.se_ratio
        self._channels = block_params.channels
        self._kernel_size = block_params.kernel_size
        self._strides = block_params.strides
        self._padding = block_params.padding
        self._block_number = block_params.block_number
        self._repeat_number = block_params.repeat_number

    def se_block(self, input_tensor):
        x = input_tensor
        block_number = self._block_number
        repeat_number = self._repeat_number

        # squeeze phase
        x = layers.GlobalAveragePooling2D(name = 'gap{}_squeeze_{}'.format(block_number, repeat_number))(x)
        x = layers.Reshape((1, 1, -1), name = 'reshape{}_{}'.format(block_number, repeat_number))(x)

        # reduce phase
        reduced_filters = max(1, (self._channels * self._se_ratio))
        x = layers.Conv2D(reduced_filters,
                          kernel_size=(1, 1),
                          strides=1,
                          padding='same',
                          activation='relu',
                          name = 'conv{}_relu_{}'.format(block_number,repeat_number))(x)
        x = layers.Conv2D(self._channels,
                          kernel_size=(1, 1),
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          name = 'conv{}_sigmoid_{}'.format(block_number, repeat_number))(x)
        # Excitaion
        x = layers.multiply([input_tensor, x], name = 'excite{}_{}'.format(block_number, repeat_number))

        return x

    def out(self, input_tensor):
        '''

        :param input_tensor: 'tensor', input_tensor (Conv_Block)
        :return: 'tensor', output_tensor
        '''

        x = input_tensor
        block_number = self._block_number
        repeat_number = self._repeat_number

        x = layers.Conv2D(self._channels,
                          kernel_size = self._kernel_size,
                          strides = self._strides,
                          padding = self._padding,
                          activation = self._activation,
                          name = 'conv{}_res_{}_1'.format(block_number, repeat_number))(x)
        x = layers.Conv2D(self._channels,
                          kernel_size = self._kernel_size,
                          strides = self._strides,
                          padding = self._padding,
                          name = 'conv{}_res_{}_2'.format(block_number, repeat_number))(x)
        x = layers.add([input_tensor, x], name = 'add{}_{}'.format(block_number, repeat_number))
        if self._batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(self._activation, name = 'activation{}_{}'.format(block_number, repeat_number))(x)

        return x

    def out_with_se_block(self, input_tensor):
        x = input_tensor

        block_number = self._block_number
        repeat_number = self._repeat_number
        x = layers.Conv2D(self._channels,
                          kernel_size=self._kernel_size,
                          strides=self._strides,
                          padding=self._padding,
                          activation=self._activation,
                          name = 'conv{}_res_{}_1'.format(block_number, repeat_number))(x)
        x = layers.Conv2D(self._channels,
                          kernel_size=self._kernel_size,
                          strides=self._strides,
                          padding=self._padding,
                          name = 'conv{}_res_{}_2'.format(block_number, repeat_number))(x)
        x = self.se_block(x)

        x = layers.add([input_tensor, x], name = 'add{}_{}'.format(block_number, repeat_number))
        if self._batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(self._activation, name = 'activation{}_{}'.format(block_number, repeat_number))(x)

        return x

def init_model(conv_2_layers, conv_3_layers, learning_rate, drop_rate, block_params_list):
    img_input = layers.Input(shape=(28, 28, 1))

    #Conv_Block_1
    conv1 = layers.Conv2D(128,
                          kernel_size = (3, 3),
                          strides = 1,
                          padding = 'same',
                          activation='relu',
                          name = 'conv1_1')(img_input)

    # Residual_Module_with_SE_Block_1
    for i in range(conv_2_layers):
        if i == 0:
            # First layer
            conv2_0 = layers.Conv2D(128,
                          kernel_size = (3, 3),
                          strides = 1,
                          padding = 'same',
                          activation='relu',
                          name = 'conv2_0')(conv1)
            args = block_params_list[0]
            args = args._replace(block_number = 2, repeat_number = i + 1)
            conv2 = Residual_Block(global_params, args).out_with_se_block(conv2_0)
        else:
            args = args._replace(repeat_number = i + 1)
            conv2 = Residual_Block(global_params, args).out_with_se_block(conv2)

    conv2 = layers.MaxPool2D((2,2), strides = 2, name = 'pool2')(conv2)

    # Residual_Module_with_SE_Block_2
    for i in range(conv_3_layers):
        if i == 0:
            # First layer
            conv3_0 = layers.Conv2D(256,
                          kernel_size = (3, 3),
                          strides = 1,
                          padding = 'same',
                          activation='relu',
                          name = 'conv3_0')(conv2)
            args = block_params_list[1]
            args = args._replace(block_number=3, repeat_number=i+1)
            conv3 = Residual_Block(global_params, args).out_with_se_block(conv3_0)
        else:
            args = args._replace(repeat_number=i+1)
            conv3 = Residual_Block(global_params, args).out_with_se_block(conv3)

    conv3 = layers.MaxPool2D((2, 2), strides=2, name = 'pool3')(conv3)

    # Residual_Module_with_SE_Block_3
    conv4_0 = layers.Conv2D(256,
                          kernel_size = (3, 3),
                          strides = 1,
                          padding = 'same',
                          activation='relu',
                          name = 'conv4_0')(conv3)
    args = block_params_list[2]
    args = args._replace(block_number = 4, repeat_number = 1)
    conv4 = Residual_Block(global_params, args).out(conv4_0)

    conv4 = layers.MaxPool2D((2, 2), strides=2, name = 'pool4')(conv4)

    # FC layers
    img_features = layers.Flatten()(conv4)
    img_features = layers.Dense(512, name = 'fc_1')(img_features)
    img_features = layers.Activation('relu', name ='fc_activation_1')(img_features)
    img_features = layers.Dropout(rate=drop_rate, name = 'fc_dropout_1')(img_features)
    img_features = layers.Dense(512, name = 'fc_2')(img_features)
    img_features = layers.Activation('relu', name ='fc_activation_2')(img_features)
    img_features = layers.Dropout(rate=drop_rate, name = 'fc_dropout_2')(img_features)

    # Output layer
    digit_pred = layers.Dense(10, activation='softmax', name = 'prediction')(img_features)

    model = keras.Model(inputs=img_input, outputs=digit_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

my_model = init_model(5, 5, 0.00001, 0.5, block_params_list= Block_params_list)
my_model.summary()