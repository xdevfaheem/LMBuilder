import torch
from torch import nn
from layernorm import LayerNormWithBias

# MLP (FF) layer of the LLM
class MLP(nn.Module):

    def __init__(self,
                 d_model=512,
                 expansion_factor=4,
                 activation="relu",
                 dropout=0.1,
                 bias=True
                 ):

        """
        Feed Forward Layer (MLP)

        Args:
            embedding_dim : Dimension of the Model
            expansion_factor : Factor which determines the Inner Dimension of the feed forward layer
            activation : Activation Type to use between feed forward layer. Default is 'relu'
            dropout : Dropout Percentage. default is '0.1'
        """

        super(MLP, self).__init__()

        self.fc = nn.Linear(d_model, expansion_factor * d_model, bias=bias)
        self.act = nn.ReLU() if activation == "relu" else nn.GELU() if activation == "gelu" else nn.ELU()
        self.proj = nn.Linear(expansion_factor * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # Layer Normalization
        self.layer_norm = LayerNormWithBias(d_model, bias)

    def forward(self, x):

        """
        Forward Pass Through Point-Wise Feed-Forward Network
        Args:
            x : Input to the feed forward layer.
        Returns:
            Flowed input through the FF Layer.
        """
    
        x = self.fc(x) # Fully Connected Layers
        x = self.act(x) # Activation Layer
        x = self.proj(x) # Linear Projection Layer
        x = self.dropout(x) # Final Dropout Layer
        return x
    
