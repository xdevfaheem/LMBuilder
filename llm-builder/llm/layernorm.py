import torch
from torch import nn
import torch.nn.functional as F


class LayerNormWithBias(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support passing `bias` argument for nn.LayerNorm"""

    def __init__(self, ndim, bias):
        super(LayerNormWithBias).__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        

class RMSNorm(nn.Module):

    """
    Root Mean Square Layer Normalization. 
    To reduce (avoid) Internal Covariance Shift (ICS)

    RMSNorm Claims re-centering (0-mean) is the success of LayerNorm rather than re-scaling (variance-1).
    Basically they claims, success of layernorm is not because of re-centering and re-scaling,
    but mostly because of re-scaling (to have a variance of 1)
    """

    def __init__(
            self,
            normalized_shape,
            eps=1e-5,
            weight=True,
            dtype=None,
            device=None,
    ):
        super(RMSNorm).__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(
                torch.ones(normalized_shape, dtype=dtype, device=device)
            )
        else:
            self.register_parameter('weight', None)

    def rms_norm(self, x: torch.Tensor, weight=None, eps=1e-5) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + eps)
        if weight is not None:
            return x_normed * weight
        return x_normed
    
    def forward(self, x):
        return self.rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)
