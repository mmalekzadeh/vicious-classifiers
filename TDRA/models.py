import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from typing import Tuple, Union
from collections import OrderedDict

### WideResNet
# Source: https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py

# class _Swish(torch.autograd.Function):
#   """Custom implementation of swish."""

#   @staticmethod
#   def forward(ctx, i):
#     result = i * torch.sigmoid(i)
#     ctx.save_for_backward(i)
#     return result

#   @staticmethod
#   def backward(ctx, grad_output):
#     i = ctx.saved_variables[0]
#     sigmoid_i = torch.sigmoid(i)
#     return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


# class Swish(nn.Module):
#   """Module using custom implementation."""

#   def forward(self, input_tensor):
#     return _Swish.apply(input_tensor)


class _Block(nn.Module):
  """WideResNet Block."""

  def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU):
    super().__init__()
    self.batchnorm_0 = nn.BatchNorm2d(in_planes)
    self.relu_0 = activation_fn()
    # We manually pad to obtain the same effect as `SAME` (necessary when
    # `stride` is different than 1).
    self.conv_0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                            padding=0, bias=False)
    self.batchnorm_1 = nn.BatchNorm2d(out_planes)
    self.relu_1 = activation_fn()
    self.conv_1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                            padding=1, bias=False)
    self.has_shortcut = in_planes != out_planes
    if self.has_shortcut:
      self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                stride=stride, padding=0, bias=False)
    else:
      self.shortcut = None
    self._stride = stride

  def forward(self, x):
    if self.has_shortcut:
      x = self.relu_0(self.batchnorm_0(x))
    else:
      out = self.relu_0(self.batchnorm_0(x))
    v = x if self.has_shortcut else out
    if self._stride == 1:
      v = F.pad(v, (1, 1, 1, 1))
    elif self._stride == 2:
      v = F.pad(v, (0, 1, 0, 1))
    else:
      raise ValueError('Unsupported `stride`.')
    out = self.conv_0(v)
    out = self.relu_1(self.batchnorm_1(out))
    out = self.conv_1(out)
    out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
    return out


class _BlockGroup(nn.Module):
  """WideResNet block group."""

  def __init__(self, num_blocks, in_planes, out_planes, stride,
               activation_fn=nn.ReLU):
    super().__init__()
    block = []
    for i in range(num_blocks):
      block.append(
          _Block(i == 0 and in_planes or out_planes,
                 out_planes,
                 i == 0 and stride or 1,
                 activation_fn=activation_fn))
    self.block = nn.Sequential(*block)

  def forward(self, x):
    return self.block(x)


class WideResNet(nn.Module):
  """WideResNet."""

  def __init__(self,
               num_classes: int = 10,
               depth: int = 28,
               width: int = 1,
               activation_fn: nn.Module = nn.ReLU,
               mean: Union[Tuple[float, ...], float] = None,
               std: Union[Tuple[float, ...], float] = None,
               padding: int = 0,
               num_input_channels: int = 3):
    super().__init__()
    self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
    self.std = torch.tensor(std).view(num_input_channels, 1, 1)
    self.mean_cuda = None
    self.std_cuda = None
    self.padding = padding
    num_channels = [16, 16 * width, 32 * width, 64 * width]
    assert (depth - 4) % 6 == 0
    num_blocks = (depth - 4) // 6
    self.init_conv = nn.Conv2d(num_input_channels, num_channels[0],
                               kernel_size=3, stride=1, padding=1, bias=False)
    self.layer = nn.Sequential(
        _BlockGroup(num_blocks, num_channels[0], num_channels[1], 1,
                    activation_fn=activation_fn),
        _BlockGroup(num_blocks, num_channels[1], num_channels[2], 2,
                    activation_fn=activation_fn),
        _BlockGroup(num_blocks, num_channels[2], num_channels[3], 2,
                    activation_fn=activation_fn))
    self.batchnorm = nn.BatchNorm2d(num_channels[3])
    self.relu = activation_fn()
    self.logits = nn.Linear(num_channels[3], num_classes)
    self.num_channels = num_channels[3]

  def forward(self, x):
    if self.padding > 0:
      x = F.pad(x, (self.padding,) * 4)
    if x.is_cuda:
      if self.mean_cuda is None:
        self.mean_cuda = self.mean.cuda()
        self.std_cuda = self.std.cuda()
      out = (x - self.mean_cuda) / self.std_cuda
    else:
      out = (x - self.mean) / self.std
    out = self.init_conv(out)
    out = self.layer(out)
    out = self.relu(self.batchnorm(out))
    out = F.avg_pool2d(out, 8)
    out = out.view(-1, self.num_channels)
    return self.logits(out)

#####################
#####################
#####################

class _DecBlock(nn.Module):
  """WideResNet Block."""

  def __init__(self, k, in_planes, out_planes, stride, activation_fn=nn.ReLU):
    super().__init__()
    self.batchnorm_0 = nn.BatchNorm2d(in_planes)
    self.relu_0 = activation_fn()
    # We manually pad to obtain the same effect as `SAME` (necessary when
    # `stride` is different than 1).
    self.has_shortcut = in_planes != out_planes
    if self.has_shortcut:
        self.conv_0 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride,
                            padding=1, bias=False)
    else:
        self.conv_0 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)                            
    self.batchnorm_1 = nn.BatchNorm2d(out_planes)
    self.relu_1 = activation_fn()
    self.conv_1 = nn.ConvTranspose2d(out_planes, out_planes, kernel_size=3, stride=1,
                            padding=1, bias=False)
   
    if self.has_shortcut:
        if k == 0:
            self.shortcut = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1,
                            stride=stride+1, padding=1, bias=False)
        else:
            self.shortcut = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2,
                            stride=stride, padding=0, bias=False)
    else:
      self.shortcut = None
    self._stride = stride

  def forward(self, x):
    if self.has_shortcut:
      x = self.relu_0(self.batchnorm_0(x))
    else:
      out = self.relu_0(self.batchnorm_0(x))
    v = x if self.has_shortcut else out
    # if self._stride == 1:
    #   v = F.pad(v, (1, 1, 1, 1))
    # elif self._stride == 2:
    #   v = F.pad(v, (0, 1, 0, 1))
    # else:
    #   raise ValueError('Unsupported `stride`.')
    out = self.conv_0(v)
    out = self.relu_1(self.batchnorm_1(out))
    out = self.conv_1(out)
    # print("o",out.shape)
    # if self.has_shortcut:
    #     print("s", self.shortcut(x).shape)
    # else:
    #     print("x",x.shape)
    out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
    return out


class _DecBlockGroup(nn.Module):
  """WideResNet block group."""

  def __init__(self, k, num_blocks, in_planes, out_planes, stride,
               activation_fn=nn.ReLU):
    super().__init__()
    block = []
    for i in range(num_blocks):
      block.append(
          _DecBlock(k,
                 i == 0 and in_planes or out_planes,
                 out_planes,
                 i == 0 and stride or 1,
                 activation_fn=activation_fn)
                 )
    self.block = nn.Sequential(*block)

  def forward(self, x):
    return self.block(x)


class DecWideResNet(nn.Module):
  """WideResNet."""

  def __init__(self,
               num_classes: int = 10,
               depth: int = 28,
               width: int = 1,
               activation_fn: nn.Module = nn.ReLU,
               mean: Union[Tuple[float, ...], float] = None,
               std: Union[Tuple[float, ...], float] = None,
               padding: int = 0,
               num_input_channels: int = 3,
               input_type = "Raw"
              ):
    super().__init__()
    self.num_classes = num_classes
    self.width = width
    self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
    self.std = torch.tensor(std).view(num_input_channels, 1, 1)
    self.mean_cuda = None
    self.std_cuda = None
    self.padding = padding
    if width > 1:
        num_channels = [64 * width, 32 * width, 16 * width,  16]
    else: 
        num_channels = [64 * width, 32 * width, 16 * width,  8]
    assert (depth - 4) % 6 == 0
    num_blocks = (depth - 4) // 6
    self.last_conv = nn.ConvTranspose2d(num_channels[3], num_input_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
    self.layer = nn.Sequential(
        _DecBlockGroup(0, num_blocks, num_channels[0], num_channels[1], 2,
                    activation_fn=activation_fn),
        _DecBlockGroup(1, num_blocks, num_channels[1], num_channels[2], 2,
                    activation_fn=activation_fn),
        _DecBlockGroup(2, num_blocks, num_channels[2], num_channels[3], 2,
                    activation_fn=activation_fn))
    self.num_channels = num_channels[3]
    self.fc = nn.Sequential(
            nn.Linear(self.num_classes,
                      64*4*4 * width),
        )
    self.final_recon = nn.Sigmoid()
    self.label_embed = nn.Embedding(self.num_classes,self.num_classes)
    
    self.input_type = input_type
    self.smx = nn.Softmax(dim=1) 
    
  def forward(self, x, y=None):
    # if y is not None:
    #     y = self.label_embed(y)
    #     x = torch.cat((x, y), 1)
    if self.input_type == "Softmax":
        x = self.smx(x)
    out = self.fc(x)
    out = out.view(out.size(0), 64 * self.width , 4, 4)
    out = self.layer(out)
    out = self.last_conv(out)
    out = self.final_recon(out)
    return out
    
    
###############################
###############################
###############################
class Parameterized(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super(Parameterized, self).__init__()        
        self.model = nn.Sequential(
                    nn.Linear(num_inputs, num_inputs*20),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.25),
                    nn.Linear(num_inputs*20, num_inputs*10),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.25),
                    nn.Linear(num_inputs*10, num_classes)
                    )
    
    def forward(self, inputs):             
        return self.model(inputs)

###############################
###############################
###############################
# src: https://github.com/YooJiHyeong/SinIR/blob/main/networks/network.py
class PostModel(nn.Module):
    def __init__(self, img_ch, net_ch, 
                mean: Union[Tuple[float, ...], float] = None,
                std: Union[Tuple[float, ...], float] = None
               ):
        super().__init__()
        
        self.mean = torch.tensor(mean).view(img_ch, 1, 1)
        self.std = torch.tensor(std).view(img_ch, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        
        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, net_ch, 1, 1, 0)
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(net_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, img_ch, 1, 1, 0),
            nn.Sigmoid()
        )
        self.layers = nn.Sequential(
            *[ConvBlock(net_ch, net_ch) for _ in range(6)]
        )

    def forward(self, x):
        
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            x = (x - self.mean_cuda) / self.std_cuda
        else:
            x = (x - self.mean) / self.std
      
        x = self.from_rgb(x)

        dense = [x]
        for l in self.layers:
            x = l(x)
            for d in dense:
                x = x + d

        x = self.to_rgb(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_c, out_c, 3, 1, 0),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.layer(x)
        

#################################################

### AEWideResNet
class AEWideResNet(nn.Module):
  """AutoEncoder WideResNet."""

  def __init__(self,
               depth: int = 28,
               width: int = 1,
               activation_fn: nn.Module = nn.ReLU,
               mean: Union[Tuple[float, ...], float] = None,
               std: Union[Tuple[float, ...], float] = None,
               padding: int = 0,
               num_input_channels: int = 3):
    super().__init__()
    self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
    self.std = torch.tensor(std).view(num_input_channels, 1, 1)
    self.mean_cuda = None
    self.std_cuda = None
    self.padding = padding
    num_channels = [16, 16 * width, 32 * width, 64 * width]
    assert (depth - 4) % 6 == 0
    num_blocks = (depth - 4) // 6
    self.init_conv = nn.Conv2d(num_input_channels, num_channels[0],
                               kernel_size=3, stride=1, padding=1, bias=False)
    self.layer = nn.Sequential(
        _BlockGroup(num_blocks, num_channels[0], num_channels[1], 1,
                    activation_fn=activation_fn),
        _BlockGroup(num_blocks, num_channels[1], num_channels[2], 2,
                    activation_fn=activation_fn),
        _BlockGroup(num_blocks, num_channels[2], num_channels[3], 2,
                    activation_fn=activation_fn))
    self.batchnorm = nn.BatchNorm2d(num_channels[3])
    self.relu = activation_fn()
    self.num_channels = num_channels[3]
    
    self.mid_conv = nn.Conv2d(num_channels[3], num_channels[3],
                              kernel_size=3, stride=2, padding=1, bias=False)

    if width > 1:
        dec_num_channels = [64 * width, 32 * width, 16 * width,  16]
    else: 
        dec_num_channels = [64 * width, 32 * width, 16 * width,  8]
    
    self.last_conv = nn.ConvTranspose2d(dec_num_channels[3], num_input_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
    self.dec_layer = nn.Sequential(
        _DecBlockGroup(0, num_blocks, dec_num_channels[0], dec_num_channels[1], 2,
                    activation_fn=activation_fn),
        _DecBlockGroup(1, num_blocks, dec_num_channels[1], dec_num_channels[2], 2,
                    activation_fn=activation_fn),
        _DecBlockGroup(2, num_blocks, dec_num_channels[2], dec_num_channels[3], 2,
                    activation_fn=activation_fn))
    self.dec_num_channels = dec_num_channels[3]
    self.final_recon = nn.Sigmoid()

    
  def forward(self, x):
    if self.padding > 0:
      x = F.pad(x, (self.padding,) * 4)
    if x.is_cuda:
      if self.mean_cuda is None:
        self.mean_cuda = self.mean.cuda()
        self.std_cuda = self.std.cuda()
      out = (x - self.mean_cuda) / self.std_cuda
    else:
      out = (x - self.mean) / self.std
    out = self.init_conv(out)
    out = self.layer(out)
    out = self.relu(self.batchnorm(out))

    out = self.mid_conv(out)    
    
    out = self.dec_layer(out)
    out = self.last_conv(out)
    out = self.final_recon(out)
    
    return out