import torch
from torch import nn


# class for Custom Softmax Activation used in Attention Mechanism and Final Projection, which gives Probablic Distribution over the given logits/tensors
class LogSoftmax(nn.Module):

    def __init__(self, axis, keepdim=True):

        super(LogSoftmax, self).__init__()

        self.axis = axis # axis along the softmax is applied
        self.keepdims = keepdim # whether to keep the structure of the dimension but shape will be 1 on guven axis or it'll be squeezed along the gven axis

    def forward(self, x):

        """
        Input:
            x: Attention Vector
        Returns:
            Probablity Distribution along the given axis
        """
        
        # logsumexp is used to prevent underflow by division by large numbers. you can also use normal sumexp
        logsumexp = torch.logsumexp(x, dim=self.axis, keepdim=self.keepdims)
        prob = torch.exp(x - logsumexp) # Element Wise Subtraction
        return prob # Output Probablities
