Global_params = collections.namedtuple('Global_params', [
    'dropout_rate', 'batch_norm','batch_norm_momentum', 'batch_norm_epsilon', 'activation'
])

Block_params = collections.namedtuple('Block_params', [
    'channels', 'output_filters','kernel_size', 'strides', 'padding',
    'expand_ratio', 'se_ratio', 'num_repeat', 'block_number'
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