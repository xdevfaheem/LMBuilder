from dataclasses import dataclass


@dataclass
class LLMConfig:
    
    """
    vocab_size : Vocab Size of the model
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
