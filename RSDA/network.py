import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torchvision import models
import torch.nn.functional as F

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class ResNet(nn.Module):
  def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=True, radius=10.0, class_num=1000):
    super(ResNet, self).__init__()
    # get pre-trained model
    model_resnet = models.resnet50(pretrained=True)
    # get layers from pre-trained model
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    # add layers to container
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
    # if we use bottleneck and  a new class (always we use both)
    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls

    if new_cls:
        if self.use_bottleneck:
            # layer for bottleneck
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            # layer for SLR layer
            self.fc = SLR_layer(bottleneck_dim, class_num)
            # init weights for bottleneck layer
            self.bottleneck.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            # NOT USE
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features
    self.radius=radius

  def forward(self, x,label=None,weight=None):
    # apply feature extractor
    x = self.feature_layers(x)
    # flatten
    x = x.view(x.size(0), -1)
    # if we use bottleneck
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    # apply norm
    x = self.radius*x/(torch.norm(x,dim=1,keepdim=True)+1e-10)
    # SLR layer
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, radius=10.0, max_iter=10000):
    super(AdversarialNetwork, self).__init__()
    self.radius=radius
    self.ad_layer1 = SP_layer(in_feature, hidden_size+1, self.radius)
    self.ad_layer2 = SP_layer(hidden_size+1, hidden_size+1, self.radius)
    self.ad_layer3 = SLR_layer(hidden_size+1,1)
    self.sigmoid=nn.Sigmoid()
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))

    x=self.ad_layer1(x)
    x=self.ad_layer2(x)
    x=self.ad_layer3(x)
    y=self.sigmoid(x)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias=Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r=input.norm(dim=1).detach()[0]
        cosine = F.linear(input, F.normalize(self.weight),r*torch.tanh(self.bias))
        output=cosine
        return output

class SP_layer(nn.Module):
    def __init__(self, in_feature, out_feature, radius):
        super(SP_layer,self).__init__()
        self.radius=radius
        self.linear=nn.Linear(in_feature-1,out_feature-1)
        self.dropout=nn.Dropout(0.5)
        self.apply(init_weights)

    def forward(self, x):
        v = self.log(x, r=self.radius)
        v = self.linear(v)
        v = self.dropout(v)
        x = self.exp(v, r=self.radius)
        x = self.srelu(x, r=self.radius)
        return x

    def exp(self,v, o=None, r=1.0):
        if v.is_cuda == True:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if o is None:
            o = torch.cat([torch.zeros(1, v.size(1)), r * torch.Tensor([[1]])], dim=1).to(device)
        theta = torch.norm(v, dim=1, keepdim=True) / r
        v = torch.cat([v, torch.zeros(v.size(0), 1).to(device)], dim=1)
        return torch.cos(theta) * o + torch.sin(theta) * F.normalize(v, dim=1) * r

    def log(self,x, o=None, r=1.0):
        if x.is_cuda == True:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if o is None:
            o = torch.cat([torch.zeros(1, x.size(1) - 1), r * torch.Tensor([[1]])], dim=1).to(device)
        c = F.cosine_similarity(x, o, dim=1).view(-1, 1)
        theta = torch.acos(self.shrink(c))
        v = F.normalize(x - c * o, dim=1)[:, :-1]
        return r * theta * v

    def shrink(self,x, epsilon=1e-4):
        x[torch.abs(x) > (1 - epsilon)] = x[torch.abs(x) > (1 - epsilon)] * (1 - epsilon)
        return x

    def srelu(self,x, r=1.0):
        return r * F.normalize(F.relu(x), dim=1)


# ***********
# Our modules
# ***********

class ShallowNet(nn.Module):
  def __init__(self, input_size, hidden_size, use_bottleneck=True, bottleneck_dim=64, radius=10.0, class_num=1000):
    super(ShallowNet, self).__init__()

    self.features = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU()
    )
    # set
    self.use_bottleneck = use_bottleneck

    if self.use_bottleneck:
        # bottleneck
        self.bottleneck = nn.Linear(hidden_size, bottleneck_dim)
        self.bottleneck_bn = nn.BatchNorm1d(bottleneck_dim)
        self.bottleneck_relu = nn.ReLU()
        self.bottleneck_dp = nn.Dropout()

        # layer for SLR layer
        self.fc = SLR_layer(bottleneck_dim, class_num)

        # init weights for bottleneck layer
        self.bottleneck.apply(init_weights)
        self.__in_features = bottleneck_dim
    else:
        # output layer
        self.fc = nn.Linear(hidden_size, class_num)
        # initialize output layer
        self.fc.apply(init_weights)
        self.__in_features = hidden_size

    self.radius = radius


  def forward(self, x):
      # reshape "x"
      x = x.view(x.size(0), -1)
      # apply feature extractor
      x = self.features(x)
      # reshape "x"
      #x = x.view(x.size(0), -1)
      # using bottleneck
      if self.use_bottleneck:
          x = self.bottleneck(x)
          x = self.bottleneck_bn(x)
          x = self.bottleneck_relu(x)
          x = self.bottleneck_dp(x)

      # apply norm
      x = self.radius * x / (torch.norm(x, dim=1, keepdim=True) + 1e-10)
      # SLR layer
      y = self.fc(x)

      return x, y


  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    # return the parameters of the deep neural network
    if self.use_bottleneck:
        parameter_list = [{"params": self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params": self.bottleneck.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params": self.fc.parameters(), "lr_mult":1, 'decay_mult':2}]
    else:
        parameter_list = [{"params": self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.fc.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list



class DFN(nn.Module):
  def __init__(self, input_size, hidden_size, use_bottleneck=True, bottleneck_dim=100, radius=10.0, class_num=1000):
    super(DFN, self).__init__()

    # For SEED and SEED-IV data
    input_size = input_size
    hidden_size = hidden_size

    # Feature extractor
    self.feature_extractor = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(hidden_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Dropout()
    )
    # set
    self.use_bottleneck = use_bottleneck

    if self.use_bottleneck:
        # bottleneck
        self.bottleneck = nn.Linear(hidden_size, bottleneck_dim)
        self.bottleneck_bn = nn.BatchNorm1d(bottleneck_dim)
        self.bottleneck_relu = nn.ReLU()
        self.bottleneck_dp = nn.Dropout()

        # layer for SLR layer
        self.fc = SLR_layer(bottleneck_dim, class_num)
        # init weights for bottleneck layer
        self.bottleneck.apply(init_weights)
        self.__in_features = bottleneck_dim
    else:
        # output layer
        self.fc = nn.Linear(hidden_size, class_num)
        # initialize output layer
        self.fc.apply(init_weights)
        self.__in_features = hidden_size

    self.radius = radius


  def forward(self, x):

      # Flatten
      x = x.view(x.size(0), -1)

      # apply feature extractor
      x = self.feature_extractor(x)

      #print(x.size())
      #input("so")

      # using bottleneck
      if self.use_bottleneck:
          x = self.bottleneck(x)
          x = self.bottleneck_bn(x)
          x = self.bottleneck_relu(x)
          x = self.bottleneck_dp(x)

      # apply norm
      x = self.radius * x / (torch.norm(x, dim=1, keepdim=True) + 1e-10)

      # SLR layer
      y = self.fc(x)

      return x, y


  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    # return the parameters of the deep neural network
    if self.use_bottleneck:
        parameter_list = [{"params": self.feature_extractor.parameters(), "lr_mult":1, 'decay_mult':2}, \
                          {"params": self.bottleneck.parameters(), "lr_mult":1, 'decay_mult':2}, \
                          {"params": self.fc.parameters(), "lr_mult":1, 'decay_mult':2}]
    else:
        parameter_list = [{"params": self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
                        {"params":self.fc.parameters(), "lr_mult":1, 'decay_mult':2}]
    return parameter_list


class Discriminator(nn.Module):
  def __init__(self, in_feature, hidden_size, radius=10.0, max_iter=10000):
    super(Discriminator, self).__init__()

    self.radius = radius
    self.ad_layer1 = SP_layer(in_feature, hidden_size+1, self.radius)
    self.ad_layer2 = SP_layer(hidden_size+1, hidden_size+1, self.radius)
    self.ad_layer3 = SLR_layer(hidden_size+1 , 1)
    self.sigmoid = nn.Sigmoid()
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))

    x=self.ad_layer1(x)
    x=self.ad_layer2(x)
    x=self.ad_layer3(x)
    y=self.sigmoid(x)
    return y

  def output_num(self):
    return 1

  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]