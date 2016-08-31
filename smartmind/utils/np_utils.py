from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


__all__ = ['conv_output_length']


def conv_output_length(input_length, kernel_size, padding, stride, dilation=1):
    if input_length is None:
        return None
    assert padding in {'SAME', 'VALID'}

    dilated_filter_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    if padding == 'SAME':
        output_length = input_length
    else:
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride
