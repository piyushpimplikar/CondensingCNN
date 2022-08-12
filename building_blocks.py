
from typing import Type, Any, Callable, Union, List, Optional
import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, Dropout, AveragePooling2D
from utils import drop_path
from operations import *
from utils import SeparableConv2d


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> Conv2D:
    """3x3 convolution with padding"""
    if dilation == 1:
        padding = "same"
    else:
        padding = "valid"
    return Conv2D(out_planes, kernel_size=(3,3), strides=(stride, stride), padding=padding, groups=groups, use_bias=False, dilation_rate=dilation)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)

def sep_conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> SeparableConv2d:
    """3x3 convolution with padding"""
    if dilation == 1:
        padding = "same"
    else:
        padding = "valid"
    return SeparableConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2D:
    """1x1 convolution"""
    return Conv2D(out_planes, kernel_size=(1,1), strides=(stride, stride), use_bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        #TODO
        concated_features = tf.concat(inputs, 1)
        # concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class BasicBlock(tf.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[tf.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., tf.Module]] = None,
        separable: bool = True,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNormalization
        #if groups != 1 or base_width != 64:
        if groups != 1: # or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if separable:
            self.conv1 = sep_conv3x3(inplanes, planes, stride)  #conv3x3(inplanes, planes, stride)
            self.conv2 = sep_conv3x3(planes, planes) #conv3x3(planes, planes)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)

        self.bn1 = norm_layer()
        self.relu = ReLU()
        self.bn2 = norm_layer()
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: tf.Variable) -> tf.Variable:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(tf.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.


    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[tf.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., tf.Module]] = None,
        separable: bool = True,
        expansion: int = 4,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNormalization
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer()
        # self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer()
        # self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * expansion)
        self.bn3 = norm_layer()
        # self.bn3 = norm_layer(planes * expansion)
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: tf.Variable) -> tf.Variable:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out





class _DenseLayer(tf.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', BatchNormalization(num_input_features)),
        self.add_module('relu1', ReLU()),
        self.add_module('conv1', Conv2D(bn_size * growth_rate,
                        kernel_size=(1,1), strides=(1,1), use_bias=False)),
        self.add_module('norm2', BatchNormalization(bn_size * growth_rate)),
        self.add_module('relu2', ReLU()),
        self.add_module('conv2', Conv2D(growth_rate,
                        kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def __call__(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = tf.keras.Model.save_weights.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = Dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class BasicDenseLayer(tf.Module):
    def __init__(self, planes, bn_size):
        super(BasicDenseLayer, self).__init__()
        self.add_module('norm1', BatchNormalization(planes)),
        self.add_module('relu1', ReLU()),
        self.add_module('conv1', Conv2D(bn_size * planes,
                        kernel_size=(1,1), strides=(1,1), use_bias=False)),
        self.add_module('norm2', BatchNormalization(bn_size * planes)),
        self.add_module('relu2', ReLU()),
        self.add_module('conv2', Conv2D(planes,
                        kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False)),

    def __call__(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        return new_features






class Transition(Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', BatchNormalization(num_input_features))
        self.add_module('relu', ReLU())
        self.add_module('conv', Conv2D(num_output_features,
                                          kernel_size=(1,1), strides=(1,1), use_bias=False))
        self.add_module('pool', AveragePooling2D(pool_size=(2,2), strides=2))


class DenseBlock(tf.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def __call__(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return tf.concat(features, 1)



class Cell(tf.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)
    self._ops = tf.Module()
    # self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def __call__(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return tf.concat([states[i] for i in self._concat], dim=1)

