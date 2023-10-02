
from torchinfo import summary
from prettytable import PrettyTable
import torch
import torch.nn as nn
import copy
import math
from lightning_utilities.core.imports import RequirementCache
from importlib.util import find_spec
from dataclasses import dataclass
from torch.nn import functional as F
from utils import number_to_words
from flash_attn import flash_attn_func

@dataclass
class GPTConfig:
    
    """
    target_vocab_size : Target Vocab Size for Final Projection
    max_seq_len : Maximum Length of the Sequence/words
    embedding_dim : Dimensionality for Embedding
    num_blocks : Number of Decoder Blocks
    activation : Activation to use inbetween feed forward layer. default is `relu`
    expansion_factor : decides the projection of inbetween neurons in feed forward layer
    num_heads : Number of Attention Heads
    droput : percentage of layers to dropout to prevent overfitting and for a stable training. default is None
    """
    
    max_seq_len=1024
    vocab_size=8000
    embed_dropout=0.1
    dropout=0.1
    bias=True
    num_heads = 8
    num_blocks = 4
    expansion_factor =4
    activation_type="relu"
    embedding_dim = 512
    pretrained_embeddings = None
    freeze_embeddings = False


# class for Embedding Layer to project the input sequences to Multi Dimensional Space
class EmbeddingLayer(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim=512,
                 pretrained_embeddings=None,
                 freeze_embeddings=True
                 ):

        """
	    Class to transform word sequences to Multi Dimensional Space (Numerical Reprensentation)

	    Arguments:
	        vocab_size : Vocablary Size
	        embedding_dim : Dimension to Represent words sequence (Feature for a single word).
	                        eg., 256, 512 (As the Dimension Increases, More Dependencies/Context can be Capture as well need more computation)

	    For example, if you have a batch of 64 sequences, each containing 15 words, and the embedding dimension is 512,
	    the output tensor will be of size 64x15x512, where each element in the tensor represents a numerical embedding.
	    """

        super(EmbeddingLayer, self).__init__()
        self.embed_dim = embedding_dim

        if pretrained_embeddings is not None:
            if pretrained_embeddings.size(1) != embedding_dim:
                raise ValueError("The embedding dimension does not match the pretrained embeddings.")
            self.embedder = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings)
        else:
            self.embedder = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """
        Inputs:
            x : The input word(s) or sequence of words that need to be embedded.
		Returns:
  			Embeddings of the Given sequence with shape (B x T x C)
        """
        x = self.embedder(x) # This gives a tensor representing the embeddings of the input words.
        embedded = x * torch.sqrt(torch.tensor(self.embed_dim)) # This scaling factor is often used to prevent gradient explosions when training deep networks.
        return embedded # The resulting tensor is the numerical representation of the input in the embedding space.
    
# class for Positional Encoding to injects some information to the embedded values about the relative or absolute position of the tokens in the sequence
class VannilaPositionalEncoding(nn.Module):

    def __init__(self, max_seq_len, embedding_dim=512):

        """
        class for Positional Embedding or Positional Encoding in Transfomer Architechture

        Arguments:
            max_len : Maximum Length of the Sequence
            embedding_dim : Dimension of the Embedding, This Must be Same as Embedding vector
        """

        super(VannilaPositionalEncoding, self).__init__()

        self.embedding_dim = embedding_dim
        self.n = 10000.0

        positional_encoding = torch.zeros(max_seq_len, embedding_dim)  # Matrix Filled with zeros of shape (max_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1) # Positions/Index of the Words in a sequence

        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(torch.log(torch.tensor(self.n)) / embedding_dim))

        """
        for i in position:
            for j in torch.arange(0, embedding_dim, 2)):
                positional_encoding[i, 2*j] = self._pe_sin(i, j)
                positional_encoding[i, 2*j+1] = self._pe_cos(i, j)

        You Can use this if you want but it can be done efficiently using vectorized operation done below.
        """

        # Vectorized Operation
        positional_encoding[:, 0::2] = torch.sin(position * div_term) # apply sin functions for every two coloumn of pos_emb matrix starting from 0. This term `position * div_term` has shape (max_seq_len, embedding_dim/2)
        positional_encoding[:, 1::2] = torch.cos(position * div_term) # apply cosine functions for every two coloumn of pos_emb matrix starting from 0.

        pe = positional_encoding.unsqueeze(0) # Add Extra Batch Dimension along the first axis
        self.register_buffer('pe', pe) # Register Buffer to make it a part of the module's state_dict

    def _pe_sin(self, position, i): # internal sin function
        return torch.sin(position / torch.pow(self.n, ((2 * i) / self.embedding_dim)))

    def _pe_cos(self, position, i): # internal cosine function
        return torch.cos(position / torch.pow(self.n, ((2 * i) / self.embedding_dim)))

    def forward(self, x):
        """
        Inputs:
            x : Embedded Sequence
        Returns:
            The Positional Encoding injected embedding vector.
        """
        # print(x.shape, self.pe.shape)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False) # Slicing the seq_len dimension upto sequence length
        return x # [batch_size, seq_len, embedding_dim]
    
# class for Custom Softmax Activation used in Attention Mechanism and Final Projection, which gives Probablic Distribution over the given logits/tensors
class CustomSoftmax(nn.Module):

    def __init__(self, axis, keepdim=True):

        super(CustomSoftmax, self).__init__()

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

class CustomLayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support passing `bias` argument """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

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
        self.softmax = CustomSoftmax(axis=-1, keepdim=True) # Custom Softmax layer
        
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
        self.layer_norm = CustomLayerNorm(d_model, bias)

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
    
# class for Single Decoder Layer in Transformers
class DecoderBlock(nn.Module):

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

        self.attention = MultiHeadAttention(max_len,
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

        self.layer_norm = CustomLayerNorm(embedding_dim, bias)
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

class GPT(nn.Module):

    def __init__(self, config):
        """
        The Decoder part of the Transformer architecture
        Arguments:
            
        """
        super(GPT, self).__init__()
        assert config.vocab_size is not None, "vocabulary size of the input is not given"
        assert config.max_seq_len is not None, "Maximum Sequence Length of the Model is not given"
        self.config = config

        vocab_size = config.vocab_size
        max_len = config.max_seq_len
        embedding_dim = config.embedding_dim
        num_blocks = config.num_blocks
        num_heads = config.num_heads

        self.decoder_embedding = EmbeddingLayer(vocab_size,
                                                embedding_dim=embedding_dim,
                                                pretrained_embeddings=config.pretrained_embeddings,
                                                freeze_embeddings=config.freeze_embeddings
                                                )

        self.position_encoder = VannilaPositionalEncoding(max_len,
                                                   embedding_dim=embedding_dim
                                                   )

        self.embedding_dropout = nn.Dropout(config.embed_dropout)

        self.final_layer_norm = CustomLayerNorm(embedding_dim, config.bias)

        self.decoder_layers = nn.ModuleList([DecoderBlock(
                                                max_len,
                                                embedding_dim=embedding_dim,
                                                num_heads=num_heads,
                                                activation=config.activation_type,
                                                expansion_factor=config.expansion_factor,
                                                dropout=config.dropout,
                                                bias=config.bias
                                                )
                                    for _ in range(num_blocks)
                                    ])

        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # self.decoder_embedding.embedder.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)

        self.count_parameters()
        
    def _init_weights(self, module: torch.nn.Module) -> None:
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        # print module name
        if isinstance(module, nn.Embedding):
            # RWKV: set it to 1e-4
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1)))
            # torch.nn.init.normal_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            # fan-in variance scaling intializer
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1)))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX       
        for name, p in module.named_parameters():
            if (name.endswith("proj.weight") and isinstance(module, MLP)):  #if use xformer swiglu, fc2 layer will be renamed to w
                torch.nn.init.normal_(p, mean=0.0, std=(1/math.sqrt(p.shape[-1]) / config.num_blocks))


    # Print a Comprehensive Summary of the Model, Modules, Submodules, Parameter Counts
    def _model_summary(self, model, generator):
        review_batch, label, mask_batch = next(generator)
        print(summary(model, input_data=[review_batch.to("cuda:0"), mask_batch.to("cuda:0")]))


    # Utility function to print the Modules, SubModules and their Corresponding trainable parmeters in a Clean Table Structure
    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"\nTotal Trainable Params: {total_params} ({number_to_words(total_params)})")

    def forward(self, x, y=None):

        """
        Forward Pass through Decoder Block

        Inputs:
            encoder_output : output from the encoder block (encoder's representation of encoder's input)
            x : decoder's input
        Returns:
            Final probablic distribution over target vocabulary
        """

        B, T = x.size()

        assert T <= self.config.max_seq_len, f"Length of the current sequence ({T}) is Longer than the Maximum Sequence Length of the Model ({self.config.max_seq_len})"

        token_embedding = self.decoder_embedding(x)

        pos_embedding = self.position_encoder(token_embedding)

        input_x = self.embedding_dropout(pos_embedding)
        for block in self.decoder_layers:

            input_x = block(input_x)

        out = self.final_layer_norm(input_x)

        if y is not None: # While Training
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(out)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        else: # While Inference
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(out[:, [-1], :]) # note: using list [-1] to preserve the time dim 
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, beam_size=1):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # If the sequence context is growing too long, we must crop it at block_size
        idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

        generated_sequences = []

        for _ in range(max_new_tokens):
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:

                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < v[:, [-1]]] = -float('inf')

            # Optionally crop the logits using nucleus sampling (top_p)
            if top_p is not None:

                # Calculate cumulative probabilities
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Determine tokens to keep based on the threshold
                sorted_indices_to_remove = cumulative_probs >= top_p
                # To maintain diversity and avoid aggressive pruning, shift the mask to retain the first token exceeding the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                # Create a mask tensor to modify logits
                mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                # Apply the mask
                logits[mask] = -float('inf')

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Use beam search if beam_size is greater than 1
            if beam_size > 1:
                # Expand sequences
                next_indices = torch.multinomial(probs, num_samples=beam_size)
                idx_candidates = torch.cat([idx_cond.repeat(beam_size, 1), next_indices], dim=-1)

                # Calculate log probabilities of candidates
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.gather(dim=1, index=next_indices)

                # Calculate scores using length normalization
                scores = log_probs / (idx_candidates.size(-1) ** (1.0 / 2.0))

                # Choose the top-k candidates
                _, topk_indices = scores.topk(beam_size, dim=0, largest=True, sorted=True)
                chosen_indices = next_indices[topk_indices]

                # Update idx_cond for next iteration
                idx_cond = chosen_indices

            else:
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # Append sampled index to the running sequence and continue
                idx_cond = torch.cat((idx_cond, idx_next), dim=1)

            generated_sequences.extend(idx_cond)

        return generated_sequences
    
if __name__ == '__main__':
    config = GPTConfig(embedding_dim=768)
    model = GPT(config)
    x = torch.randint(0, 8000, (2, 1024), dtype=torch.int64)
    print(f"Input's Shape: {x.shape}")
    out, loss = model(x)
    print(f"Out: {out}\nOut Shape: {out.shape}")
