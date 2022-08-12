import platform

from global_layer import *
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, ReLU, Conv2D, BatchNormalization, GlobalAveragePooling2D,\
    Dense, Flatten, Softmax, AveragePooling2D
from tensorflow.keras.activations import selu
from tensorflow.keras.models import Sequential, load_model
import math

class ResNet(tf.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        global_ft = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., tf.Module]] = None,
        n1: int = 64,
        n2: int = 128,
        n3: int = 128,
        n4: int = 128,
        cell_type : str = 'default',   
        args = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNormalization
        self._norm_layer = norm_layer


        pde_args = {
                'K':             args.K, 
                'separable':     args.separable, 
                'nonlinear_pde': args.non_linear, 
                'cDx' :          args.cDx,
                'cDy' :          args.cDy,
                'dx' :           args.dx,
                'dy' :           args.dy,
                'dt' :           args.dt, 
                'init_h0_h':     args.init_h0_h,
                'use_silu' :     args.use_silu,
                'use_res' :      args.use_res,
                'constant_Dxy':  args.constant_Dxy,
                'custom_uv':     args.custom_uv,
                'custom_dxy':    args.custom_dxy,
                'no_f' :         args.no_f,
                'cell_type' :    cell_type,
                'old_style' :    False, # True,
                'no_diffusion' :    args.no_diffusion,
                'no_advection' :    args.no_advection
        }


        self.global_ft = global_ft

        self.inplanes = n1
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = Conv2D(self.inplanes, kernel_size=(3,3), strides=(1,1), padding="same",use_bias=False)
        self.bn1 = norm_layer()
        self.relu = ReLU()
        
        self.separable = args.separable 
        self.layer1 = self._make_layer(block, n1, layers[0])
        self.layer2 = self._make_layer(block, n2, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, n3, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

        self.original = len( layers ) == 3 
        if self.original == False:
            self.layer4 = self._make_layer(block, n4, layers[3], stride=2, dilate=replace_stride_with_dilation[1])
        else:
            assert ( n3 == n4 )
        
        if self.global_ft:
            self.global1 = GlobalFeatureBlock_Diffusion(n1, pde_args)   
            self.global2 = GlobalFeatureBlock_Diffusion(n2, pde_args) 
            self.global3 = GlobalFeatureBlock_Diffusion(n3, pde_args)
        
        self.avgpool = AdaptiveAveragePooling2D((1,1))
        # self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes)

        for m in self.submodules:
            if isinstance(m, Conv2D):
                m.keras_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # param = m.get_weights()
                # n = param.size(0) * param.size(2) * param.size(3)
                # param.data.normal_().mul_(math.sqrt(2. / n))

        # elif isinstance(m, (BatchNormalization, nn.GroupNorm)):
        #     nn.init.constant_(m.weight, 1)
        #     nn.init.constant_(m.bias, 0)





        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.submodules:
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential()
            downsample.add(conv1x1(self.inplanes, planes * block.expansion, stride))
            downsample.add(BatchNormalization())

        # layers = []
        layers = Sequential()
        layers.add(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, separable=self.separable))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.add(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, separable=self.separable))




        # b = Sequential(*layers)
        return layers
        # return Sequential(*layers)

    def _forward_impl(self, x: tf.Variable) -> tf.Variable:
        # See note [TorchScript super()]


        debug = False
        if debug: print('x = ', x.shape)
        
        x = self.conv1(x)
        if debug: print('conv1 = ', x.shape)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        if debug: print('layer1 = ', x.shape)
            
        if self.global_ft:
            x = self.global1(x)
            if debug: print('global1 = ', x.shape)


        x = self.layer2(x)
        if debug: print('layer2 = ', x.shape)
            
        if self.global_ft:
            x = self.global2(x)
            if debug: print('global2 = ', x.shape)

        x = self.layer3(x)
        if debug: print('layer3 = ', x.shape)

        if self.global_ft:
            x = self.global3(x)
            if debug: print('global3 = ', x.shape)
            
        if self.original == False:
            x = self.layer4(x)
            if debug: print('layer4 = ', x.shape)

        x = self.avgpool(x)
        if debug: print('L4 avgpool = ', x.shape)
            
        x = Flatten()(x)
        x = self.fc(x)
        if debug: print('fc = ', x.shape)
        # if debug: assert (1 == 2)

        x = Softmax()(x)


        return x

    def __call__(self, x: tf.Variable) -> tf.Variable:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_model(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet32(pretrained: bool = False, progress: bool = True, m : int = 5, **kwargs: Any) -> ResNet:
    return _resnet('resnet32', BasicBlock, [m, m, m, m], pretrained, progress, global_ft = False,
                   **kwargs)

def pdenet(pretrained: bool = False, progress: bool = True, m : int = 2, **kwargs: Any) -> ResNet:
    return _resnet('PDE32', BasicBlock, [m, m, m, m], pretrained, progress, global_ft = True,
                   **kwargs)


def resnet_original(pretrained: bool = False, progress: bool = True, m : int = 5, **kwargs: Any) -> ResNet:
    return _resnet('resnet-original', BasicBlock, [m, m, m], pretrained, progress, global_ft = False,
                   **kwargs)


def pdenet_original(pretrained: bool = False, progress: bool = True, m : int = 1, **kwargs: Any) -> ResNet:
    return _resnet('pde-original', BasicBlock, [m, m, m], pretrained, progress, global_ft = True,
                   **kwargs)




