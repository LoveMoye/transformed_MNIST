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
        self._block_number = block_params.block_number
        self.dropout = layers.Dropout(rate=self._dropout_rate, name = 'dropout{}'.format(self._block_number))

    def mbconv_block(self, input_tensor):
        # Expansion phase
        x = input_tensor

        if self._expand_ratio != 0:
            self.filters_expand = self._channels * self._expand_ratio
            x = layers.Conv2D(self.filters_expand,
                              kernel_size=(1, 1),
                              strides=1,
                              padding='same',
                              kernel_initializer=CONV_KERNEL_INITIALIZER,
                              name = 'conv{}_expand_{}'.format(self._block_number, self._repeat_number))(x)
            if self._batch_norm:
                x = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                              epsilon=self._batch_norm_epsilon,
                                              name = 'bn{}_expand_{}'.format(self._block_number, self._repeat_number))(x)
            x = layers.Activation(self._activation,
                                  name = 'activation{}_expand_{}'.format(self._block_number, self._repeat_number))(x)
        else:
            self.filters_expand = self._channels

        # Depthwise conv block
        x = layers.DepthwiseConv2D(kernel_size=self._kernel_size,
                                   strides=self._strides,
                                   padding=self._padding,
                                   depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                   name = 'depth_conv{}_{}'.format(self._block_number, self._repeat_number))(x)
        if self._batch_norm:
            x = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                          epsilon=self._batch_norm_epsilon,
                                          name = 'depth_bn{}_{}'.format(self._block_number, self._repeat_number))(x)
        x = layers.Activation(self._activation,
                              name = 'depth_activation{}_{}'.format(self._block_number, self._repeat_number))(x)

        # SE Block
        x = self.se_block(x)

        # Output phase
        x = layers.Conv2D(self._output_filters,
                          kernel_size=(1, 1),
                          strides=1,
                          padding='same',
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name = 'conv{}_out_{}'.format(self._block_number, self._repeat_number))(x)
        if self._batch_norm:
            x = layers.BatchNormalization(momentum=self._batch_norm_momentum,
                                          epsilon=self._batch_norm_epsilon,
                                          name = 'bn{}_out_{}'.format(self._block_number, self._repeat_number))(x)

        return x

    def se_block(self, input_tensor):

        x = input_tensor
        # squeeze phase
        x = layers.GlobalAveragePooling2D(name = 'gap{}_squeeze{}'.format(self._block_number, self._repeat_number))(x)
        x = layers.Reshape((1, 1, -1), name = 'reshape{}_{}'.format(self._block_number, self._repeat_number))(x)

        # reduce phase
        reduced_filters = max(1, (self._channels * self._se_ratio))
        x = layers.Conv2D(reduced_filters,
                          kernel_size=(1, 1),
                          strides=1,
                          padding='same',
                          activation='relu',
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name = 'conv{}_relu_{}'.format(self._block_number, self._repeat_number))(x)
        x = layers.Conv2D(self.filters_expand,
                          kernel_size=(1, 1),
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name = 'conv{}_sigmoid_{}'.format(self._block_number, self._repeat_number))(x)

        x = layers.multiply([input_tensor, x], name = 'excite{}_{}'.format(self._block_number, self._repeat_number))

        return x

    def out(self, input_tensors):
        '''

        :param input_tensors: 'tensor', a previous tensors
        :return: an output tensors
        '''

        x = input_tensors

        for i in range(self._num_repeat):
            self._repeat_number = i
            if i == 0:
                x = self.mbconv_block(x)
            else:
                self._channels = self._output_filters
                self._strides = 1
                x_indentity = tf.identity(x, name = 'identity{}_{}'.format(self._block_number, self._repeat_number))
                x = self.mbconv_block(x)
                x = self.dropout(x)
                x = layers.add([x_indentity, x], name = 'add{}_{}'.format(self._block_number, self._repeat_number))

        return x