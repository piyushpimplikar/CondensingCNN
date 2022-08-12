from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, ReLU, Conv2D, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow as tf
OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: AveragePooling2D(pool_size=(3,3), strides=(stride,stride), padding="same"),
  'max_pool_3x3' : lambda C, stride, affine: MaxPooling2D(pool_size=(3,3), strides=(stride,stride), padding="same"),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: Sequential(
    ReLU(),
    Conv2D(C, kernel_size=(1,7), strides=(1, stride), padding=(0, 3), use_bias=False),
    Conv2D(C, kernel_size=(7,1), strides=(stride, 1), padding=(3, 0), use_bias=False),
    BatchNormalization()
    ),
}

class ReLUConvBN(tf.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = Sequential(
      ReLU(),
      Conv2D(C_out, (kernel_size,kernel_size), strides=(stride,stride), padding=padding, use_bias=False),
      BatchNormalization()
    )

  def forward(self, x):
    return self.op(x)

class DilConv(tf.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = Sequential(
      ReLU(),
      Conv2D(C_in, kernel_size=(kernel_size,kernel_size), strides=(stride,stride), padding=padding, dilation=dilation, groups=C_in, use_bias=False),
      Conv2D(C_out, kernel_size=(1,1), padding="valid", use_bias=False),
      BatchNormalization(),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(tf.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = Sequential(
      ReLU(),
      Conv2D(C_in, kernel_size=(kernel_size,kernel_size), strides=(stride,stride), padding=padding, groups=C_in, use_bias=False),
      Conv2D(C_in, kernel_size=(1,1), padding="valid", use_bias=False),
      BatchNormalization(),
      ReLU(),
      Conv2D(C_in, kernel_size=(kernel_size,kernel_size), strides=(1,1), padding=padding, groups=C_in, use_bias=False),
      Conv2D(C_out, kernel_size=(1,1), padding="valid", use_bias=False),
      BatchNormalization(),
      )

  def forward(self, x):
    return self.op(x)


class Identity(tf.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(tf.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

class FactorizedReduce(tf.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = ReLU()
    self.conv_1 = Conv2D(C_out // 2, 1, strides=(2,2), padding="valid", use_bias=False)
    self.conv_2 = Conv2D(C_out // 2, 1, strides=(2,2), padding="valid", use_bias=False)
    self.bn = BatchNormalization()

  def forward(self, x):
    x = self.relu(x)
    out = tf.concat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

