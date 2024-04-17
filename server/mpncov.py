# Copyright (c) 2018 Peihua Li and Jiangtao Xie
# See https://github.com/jiangtaoxie/fast-MPN-COV/blob/master/LICENSE
# for license for this file.

import torch
import torch.nn as nn
from torch.autograd import Function


class Basemodel(nn.Module):
    """Load backbone model and reconstruct it into three part:
       1) feature extractor
       2) global image representaion
       3) classifier
    """
    def __init__(self):
        super(Basemodel, self).__init__()
        basemodel = MPNCOVResNet(Bottleneck, [3, 4, 6, 3])
        basemodel = self._reconstruct_mpncovresnet(basemodel)

        self.features = basemodel.features
        self.representation = basemodel.representation
        self.classifier = basemodel.classifier
        self.representation_dim = basemodel.representation_dim

    def _reconstruct_mpncovresnet(self, basemodel):
        model = nn.Module()

        model.features = nn.Sequential(*list(basemodel.children())[:-1])
        model.representation_dim = basemodel.layer_reduce.weight.size(0)

        model.representation = None
        model.classifier = basemodel.fc
        return model

    def forward(self, x):
        x = self.features(x)
        x = self.representation(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class Newmodel(Basemodel):
    def __init__(self, representation, num_classes, freezed_layer):
        super(Newmodel, self).__init__()
        representation_method = representation['function']
        representation.pop('function')
        representation_args = representation
        representation_args['input_dim'] = self.representation_dim
        self.representation = representation_method(**representation_args)
        fc_input_dim = self.representation.output_dim
        representation['function']=MPNCOV

        self.classifier = nn.Linear(fc_input_dim, num_classes)

        index_before_freezed_layer = 0
        if freezed_layer:
            for m in self.features.children():
                if index_before_freezed_layer < freezed_layer:
                    m = self._freeze(m)
                index_before_freezed_layer += 1

    def _freeze(self, modules):
        for param in modules.parameters():
            param.requires_grad = False
        return modules


class MPNCOV(nn.Module):
     """Matrix power normalized Covariance pooling (MPNCOV)
        implementation of fast MPN-COV (i.e.,iSQRT-COV)
        https://arxiv.org/abs/1712.01034
     Args:
         iterNum: #iteration of Newton-schulz method
         is_sqrt: whether perform matrix square root or not
         is_vec: whether the output is a vector or not
         input_dim: the #channel of input feature
         dimension_reduction: if None, it will not use 1x1 conv to
                               reduce the #channel of feature.
                              if 256 or others, the #channel of feature
                               will be reduced to 256 or others.
     """
     def __init__(
             self, iterNum=3, is_sqrt=True, is_vec=True, input_dim=2048,
             dimension_reduction=None):

         super(MPNCOV, self).__init__()
         self.iterNum=iterNum
         self.is_sqrt = is_sqrt
         self.is_vec = is_vec
         self.dr = dimension_reduction
         if self.dr is not None:
             self.conv_dr_block = nn.Sequential(
               nn.Conv2d(input_dim, self.dr, kernel_size=1, stride=1,
                         bias=False),
               nn.BatchNorm2d(self.dr),
               nn.ReLU(inplace=True)
             )
         output_dim = self.dr if self.dr else input_dim
         if self.is_vec:
             self.output_dim = int(output_dim*(output_dim+1)/2)
         else:
             self.output_dim = int(output_dim*output_dim)
         self._init_weight()

     def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

     def _cov_pool(self, x):
         return Covpool.apply(x)
     def _sqrtm(self, x):
         return Sqrtm.apply(x, self.iterNum)
     def _triuvec(self, x):
         return Triuvec.apply(x)

     def forward(self, x):
         if self.dr is not None:
             x = self.conv_dr_block(x)
         x = self._cov_pool(x)
         if self.is_sqrt:
             x = self._sqrtm(x)
         if self.is_vec:
             x = self._triuvec(x)
         return x


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class MPNCOVResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MPNCOVResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.layer_reduce = nn.Conv2d(
            512 * block.expansion, 256, kernel_size=1, stride=1, padding=0,
            bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(int(256*(256+1)/2), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 1x1 Conv. for dimension reduction
        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)

        x = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(x, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = ((-1./M/M)*torch.ones(M,M,device = x.device) +
                  (1./M)*torch.eye(M,M,device = x.device))
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y

     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input


class Sqrtm(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = (3.0 * torch.eye(dim,dim,device = x.device).view(
             1, dim, dim).repeat(batchSize,1,1).type(dtype))
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(
             batchSize, iterN, dim, dim, requires_grad=False,
             device=x.device).type(dtype)
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(
             batchSize,iterN,1,1).type(dtype)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            YZY = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5 * (I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
               Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
               Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            YZY = (0.5 * Y[:,iterN-2,:,:].bmm(
                I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:])))
         y = YZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, YZY, normA, Y, Z)
         ctx.iterN = iterN
         return y

     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = (grad_output *
                        torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x))
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(
             2 * torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(
             batchSize, 1, 1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
         else:
            dldY = 0.5 * (
                der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) -
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) -
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) -
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         der_NSiter = der_NSiter.transpose(1, 2)
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             update = (
             (der_postComAux[i] - (grad_aux[i] / (normA[i] * normA[i]))) *
                 torch.ones(dim, device = x.device).diag().type(dtype))
             grad_input[i,:,:] += update
         return grad_input, None


class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().reshape(dim*dim)
         index = I.nonzero()
         y = torch.zeros(batchSize, int(dim*(dim+1)/2),
                         device=x.device).type(dtype)
         y = x[:,index]
         ctx.save_for_backward(input,index)
         return y

     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         grad_input = torch.zeros(batchSize, dim*dim, device=x.device,
                                  requires_grad=False).type(dtype)
         grad_input[:,index] = grad_output
         grad_input = grad_input.reshape(batchSize,dim,dim)
         return grad_input


def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

def TriuvecLayer(var):
    return Triuvec.apply(var)
