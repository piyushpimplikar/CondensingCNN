from tensorflow.keras.layers import Conv2D
import math
import tensorflow as tf
def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    # if isinstance(kernel_size, list):
    #     assert 'num_experts' not in kwargs  # MixNet + CondConv combo not supported currently
    #     assert 'groups' not in kwargs  # MixedConv groups are defined by kernel list
    #     # We're going to use only lists for defining the MixedConv2d kernel groups,
    #     # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
    #     m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    # else:
    depthwise = kwargs.pop('depthwise', False)
    # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
    groups = in_channels if depthwise else kwargs.pop('groups', 1)

    m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m

def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        stride = kwargs.get('stride')
        if padding == 1:
            padding = "same"
        else:
            padding = "valid"
        return Conv2D(out_chs, kernel_size=kernel_size, padding=padding, strides=(stride,stride), dilation_rate=kwargs.get('dilation'), use_bias=kwargs.get('bias'), groups=kwargs.get('groups'))


def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic
# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class Conv2dSame(Conv2D):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv2d_same(
        x, weight, bias = None, stride = (1, 1),
        padding= (0, 0), dilation = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return Conv2D(filters=weight, use_bias=bias, strides=stride, padding="valid", dilation_rate=dilation, groups=groups)

# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k, s, d = (1, 1), value = 0):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = tf.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return x

# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)
