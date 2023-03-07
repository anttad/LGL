import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import math
# from torch.utils import _single, _pair, _triple, _reverse_repeat_tuple
# from torch.nn.conv import _ConvNd
# from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union

class LinearGML(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, freeze=False, with_std=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearGML, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs)) # on prend la transposée en inversant in et out features pour pouvoir calculer les distance plus facilement
        # self.with_std = with_std
        self.precision =  Parameter(torch.empty((1,out_features), **factory_kwargs)) #sqrt of the precision, we have to use the square function later to enforce positivity 
        if not with_std:
            self.precision.requires_grad = False 
        if freeze: 
            self.weight.requires_grad = False 
        #TODO inclure la matrice de covariance ?
        
        # if bias:
        #     self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        # else:
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # nn.init.uniform_(self.weight,-4,4)
        # nn.init.kaiming_uniform_(self.weight, a=np.sqrt(self.in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     nn.init.uniform_(self.bias, -bound, bound)
        # if self.with_std:
        nn.init.constant_(self.precision,1)

    def forward(self, input: Tensor) -> Tensor:
        # pdist = nn.PairwiseDistance(p=2)
        # output = pdist(input,self.weight)
        # return output
        return torch.mul(torch.square(self.precision),torch.cdist(input,self.weight))


    # def backward(self,grad_value):
    #     print("grad_value: ", grad_value)
    #     return grad_value.clone()
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        
        
class LinearGML2(nn.Module):
       
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None, freeze=False,with_cov=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearGML2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight = Parameter(torch.empty((out_features,1, in_features), **factory_kwargs)) # on prend la transposée en inversant in et out features pour pouvoir calculer les distance plus facilement
        if freeze: 
            self.weight.requires_grad = False 
        # self.cov_list = [torch.empty((in_features,in_features),**factory_kwargs ) for _ in range(out_features)]
        
        # self.covs = torch.empty((in_features,in_features,out_features),**factory_kwargs ) 
        self.prec_decomp = Parameter(torch.empty((out_features,in_features,in_features),**factory_kwargs )) #list of matrices V such that P = V^T V , with P = inv(Sigma) the inverse of cov matrix
        self.L_diags =  Parameter(torch.empty((out_features,in_features),**factory_kwargs ))
        # print(in_features*(in_features-1)/2)
        self.L_lower =  Parameter(torch.empty((out_features,int(in_features*(in_features-1)/2)),**factory_kwargs ))
        if not with_cov:
            self.L_lower.requires_grad = False
        # self.L = Parameter(torch.empty((out_features,in_features,in_features),**factory_kwargs ))
        # self.L.requires_grad=False
        # if bias:
        #     self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        # else:
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # nn.init.uniform_(self.weight, -4,4)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        nn.init.uniform_(self.prec_decomp, -1,1)
        #TODO Initialize L_diag, L_lower
        nn.init.constant_(self.L_lower, 0)
        nn.init.constant_(self.L_diags,1)
        # nn.init.zeros_(self.L)
        # for i in range(self.out_features):
        #     nn.init.uniform_(self.cov_list[i], -1,1)
        
        # nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # def forward_old(self, inputs: Tensor) -> Tensor:
    #     B = inputs.shape[0] # extract batch_size
    #     pinputs =  torch.matmul(inputs, self.prec_decomp ) #TODO Problem ça fonctionnera pas
    #     pweights = torch.matmul(self.weight,self.prec_decomp)
    #     pweights = pweights.repeat(1,B,1)
        
    #     dist_ = torch.add(pinputs,pweights, alpha=-1)
    #     output = torch.norm(dist_,dim=2)
    #     return torch.t(output)
    
    def forward(self, inputs: Tensor) -> Tensor:
        B = inputs.shape[0] # extract batch_size
        self.L_diags_sqr = torch.square(self.L_diags)
        tril_indices = torch.tril_indices(row=self.in_features, col=self.in_features, offset=-1)
        self.L=torch.zeros_like(self.prec_decomp)
        for k in range(self.out_features):
            self.L[k,:,:] = torch.diag(self.L_diags_sqr[k])
        self.L[:,tril_indices[0], tril_indices[1]] = self.L_lower
        pinputs =  torch.matmul(inputs, self.L ) #TODO Problem ça fonctionnera pas
        pweights = torch.matmul(self.weight,self.L)
        pweights = pweights.repeat(1,B,1)
        
        dist_ = torch.add(pinputs,pweights, alpha=-1)
        output = torch.norm(dist_,dim=2)
        
        return torch.t(output)

    def get_L_matrix(self):
        L_diags_sqr = torch.square(self.L_diags)
        tril_indices = torch.tril_indices(row=self.in_features, col=self.in_features, offset=-1)
        L=torch.zeros_like(self.prec_decomp)
        for k in range(self.out_features):
            L[k,:,:] = torch.diag(L_diags_sqr[k])
        L[:,tril_indices[0], tril_indices[1]] = self.L_lower
        return L
    
    def get_P_matrix(self):
        L = self.get_L_matrix()
        P = torch.matmul(L,torch.transpose(L,1,2))
        return P
        
    # def backward(self,grad_value):
    #     print("grad_value: ", grad_value)
    #     return grad_value.clone()
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
        
        

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).type(torch.FloatTensor).cuda() #.to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x         
        
    
# class Conv2d(_ConvNd):
    

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size,
#         stride= 1,
#         padding = 0,
#         dilation = 1,
#         groups: int = 1,
#         bias: bool = True,
#         padding_mode: str = 'zeros',  # TODO: refine this type
#         device=None,
#         dtype=None
#     ) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         kernel_size_ = _pair(kernel_size)
#         stride_ = _pair(stride)
#         padding_ = padding if isinstance(padding, str) else _pair(padding)
#         dilation_ = _pair(dilation)
#         super(Conv2d, self).__init__(
#             in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
#             False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

#     def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
#         if self.padding_mode != 'zeros':
#             return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
#                             weight, bias, self.stride,
#                             _pair(0), self.dilation, self.groups)
#         return F.conv2d(input, weight, bias, self.stride,
#                         self.padding, self.dilation, self.groups)

#     def forward(self, input: Tensor) -> Tensor:
#         return self._conv_forward(input, self.weight, self.bias)
          
