import torch
from torch import nn
from logsoftmax import LogSoftMax
from lightning_utilities.core.imports import RequirementCache

# class for Multi Head Attention Module
class MultiHeadAttention(nn.Module):

    def __init__(self, max_seq_len, embedding_dim=512, n_heads=8, dropout=0.1, bias=True):

        """
        Multi Head Attention Module

        Arguments:
            embedding_dim : Embedding Dimension of the input Sequence
            n_heads : Number of Attention Heads to run in parallel
        """

        super(MultiHeadAttention, self).__init__()

        self.embedding_dim = embedding_dim # Embedding Dimension of the model
        self.dropout = dropout
        self.attention_dropout = nn.Dropout(dropout)

        assert embedding_dim % n_heads == 0, "Embedding  Dimension divided by no. of heads should give no remainder. so that it can be equally splitted"

        self.n_heads = n_heads # Number of Attention Heads. eg., 8 by default
        self.head_size = int(embedding_dim // n_heads) # Embedding Dimension for a single head
        self.softmax = LogSoftMax(axis=-1, keepdim=True) # Custom Softmax layer
        
        # get cuda compute capablity
        major, minor = (torch.cuda.get_device_capability(torch.cuda.current_device()))
        _cuda_ver = float(f"{major}.{minor}")
        
        self.flash_attn = ( # Requirements for flash attention
            RequirementCache("flash-attn>=2.0.0.post1") and
            (_cuda_ver > 8.0) and 
            (float(torch.version.cuda) >= 11.6) and 
            (float(".".join(torch.__version__.split('.')[:2])) > 1.12) and 
            (find_spec("packaging") is not None) and 
            (find_spec("ninja") is not None)
        )
        if not self.flash_attn:
            print("WARNING: using slow attention. Flash Attention is not Supported!")
            # causal mask to ensure that attention is only applied to the left in the input sequence (lower left of the attention matrix)
            self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0))

        # Weighted Matricies
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=bias) # weighted matricies to transform/project the query matrix to perform self attention
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=bias)  # weighted matricies to transform/project the key to perfom self attention
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=bias)  # weighted matricies to transform/project the value matrix

        self.final_fc_layer = nn.Linear(self.head_size * self.n_heads, embedding_dim, bias=bias) # Final Layer for projecting the q, k, v matricies into a single tensor of shape(embedding_dim, embedding_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def _casual_self_attention(self, query, key, value, scale=False):
        
        """
        Casual Self-Attention:
        
        A Quick Reference about Casual Masking:
            In Self Attention, query and key matricies are same without masking, one word representation in query can see every other word representation in key matrix. this way,
            a word can see every other word within a same sentence but, In Casual Self-Atention, Every word in sentence can only see preceding tokens to predict the next token. Here, casual masking
            is used to prevent current token from seeing the future token.

        Args:
            query : query representationtensor with shape (batch_size, heads, seq_len, head_size)
            key   : key representation tensor with shape (batch_size, heads, seq_len, head_size)
            value : value representation tensor with shape (batch_size, heads, seq_len, head_size)
            scale : whether to scale the dot product of the query and transposed key

        Returns:
            Output of Attention Mechanism with shape (batch_size, seq_len, heads, head_size)
        """
        
        assert query.shape[-1] == key.shape[-1] == value.shape[-1], "Embedding dimensions of q, k, v aren't all the same"

        if scale:
            depth = query.shape[-1] # Scaling factor for the dot product attention
        else:
            depth = 1

        T = query.shape[-2] # seq_len

        # query shape: batch_size, n_heads, q_len, head_size. e.g: (32x8x10x64)
        # key shape: batch_size, n_heads, k_len, head_size. e.g: (32x8x10x64). k_transposed shape: batch_size, n_heads, head_size, k_len. eg., (32, 8, 64, 10)
        # product shape should be: batch_size, heads, q_len, k_len, e.g: (32x8x10x10)
        dots = torch.matmul(query,key.transpose(2,3))
        dots = torch.where(self.bias[:,:, :T, :T], dots, torch.full_like(dots, float('-inf')))

        scores = self.softmax(dots / torch.sqrt(torch.tensor(depth))) # perform softmax operation dot product scaled by the scaling factor
        scores = self.attention_dropout(scores)

        # scores shape: batch_size, heads, q_len, v_len, e.g: (32x8x10x10)
        # value shape: batch_size, heads, v_len, head_size, e.g: (32x8x10x64)
        # output: batch_size, heads, q_len, head_size, e.g: (32x8x10x64)
        weights = torch.matmul(scores, value)
        weights.permute(0, 2, 1, 3).contiguous() # Swapping the second and first fimension of the weights matrix. resulting matrix has a shape of [batch_size, seq_len, heads, head_size]

        return weights, scores

    def forward(self, x, return_attention=False):

        """
        Forward pass through Multi-Head Attention module.

        Inputs:
            x (Tensor) : The Input to MHA wich is to be splitted as q, k, v
        Returns:
            output (Tensor): The output of the multi-head attention step, of shape (batch_size, seq_len, dim).
            attention_weights (Tensor): Attention weights of shape (batch_size, heads, seq_len, seq_len). it is optional.
        """

        # Input of size: batch_size x sequence length x embedding dims
        batch_size = x.size(0)
        seq_len = x.size(1)

        # project the queries, keys and values by their respective weight matrices
        key = self.key(x)  # [batch_size, seq_len, embedding_dim]
        query = self.query(x)  # [batch_size, seq_len, embedding_dim]
        value = self.value(x)  # [batch_size, seq_len, embedding_dim]

        # reshape from (batch_size x seq_len x embed_size) -> (batch_size x seq_len x n_heads x head_size)
        # example: from (32x10x512) -> (32x10x8x64)
        query = query.view(batch_size, seq_len, self.n_heads, self.head_size).permute(0, 2, 1, 3) # [batch_size, seqLen, n_head, head_size] -> [batch_size, n_head, seqLen, head_size]
        key = key.view(batch_size, seq_len, self.n_heads, self.head_size).permute(0, 2, 1, 3) # [batch_size, seqLen, n_head, head_size] -> [batch_size, n_head, seqLen, head_size]
        value = value.view(batch_size, seq_len, self.n_heads, self.head_size).permute(0, 2, 1, 3) # [batch_size, seqLen, n_head, head_size] -> [batch_size, n_head, seqLen, head_size]
    
        # causal self-attention; Self-attend: (B, H, T, HS) x (B, H, HS, T) -> (B, HS, T, T)
        if self.flash_attn:
            # efficient attention using Flash Attention CUDA kernels
            weights = flash_attn_func(query, key, value, dropout_p=(self.dropout if self.training else 0.0), softmax_scale=(1.0/(torch.sqrt(torch.tensor(query.shape[-1])))), causal=True)

        else:
            # our implemetation
            weights, attention_scores = self._casual_self_attention(query, key, value, scale=True) # batch_size, heads, v_len, head_size,

        output = self.final_fc_layer(weights.view(batch_size, seq_len, self.n_heads * self.head_size)) # (batch_size, seq_len, embedding_dims)

        if return_attention:
            if self.flash_attn:
                return self.proj_dropout(output), None
            return self.proj_dropout(output), attention_scores

        else:
            return self.proj_dropout(output)
