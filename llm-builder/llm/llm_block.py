import torch
from torch import nn
from attention import CasualMultiHeadAttention
from mlp import MLP
from layernorm import LayerNormWithBias


# class for Single Decoder Layer in Transformers
class Regressive_Block(nn.Module):

    def __init__(self,
                 max_len,
                 embedding_dim=512,
                 num_heads=8,
                 activation="relu",
                 expansion_factor=4,
                 dropout=0.1,
                 bias=True
                 ):

        """
        Single Decoder Layer in Transformer Architechture (TA), consist of transformer block used in both encoder and decoder and
        a Masked Multi Head Attention with casual masking.

        Args:
            embedding_dim : Embedding Dimension
            num_heads : Number of heads
            activation : Activation to use inbetween feed forward layer. default is `relu`
            expansion_factor : decides the inner dimension of feed forward layer
            dropout : percentage of layers to drop. default is None
        """

        super(DecoderBlock, self).__init__()

        self.attention = CasualMultiHeadAttention(max_len,
                                                   embedding_dim=embedding_dim,
                                                   n_heads=num_heads,
                                                   dropout=dropout,
                                                   bias=bias
                                                   )

        self.ff_layer = MLP(d_model=embedding_dim,
                                    expansion_factor=expansion_factor,
                                    activation=activation,
                                    dropout=dropout,
                                    bias=bias
                                    )

        self.layer_norm = LayerNormWithBias(embedding_dim, bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        """
        Forward Pass through Decoder Layer
        Inputs:
            x - Decoder Block's Input
        Returns:
            Decoder Block's Output after the input went through MHA and MLP with some dropout and add & norm.
        """

        # Masked Multi Head Attention
        attention_out, attention_scores = self.attention(x, return_attention=True) # Performing casual attention on decoder's input
        self.last_attn_scores = attention_scores
        # Residual Connection & Dropout & Layer Normalization
        attention_norm = self.dropout(self.layer_norm(attention_out + x))
        # Multi Layer Perceptron (MLP/FF)
        mlp_out = self.ff_layer(attention_norm)
        # Residual Connection & Dropout  & Layer Normalization
        ff_norm = self.dropout(self.layer_norm(mlp_out + attention_norm))
        
        return ff_norm
