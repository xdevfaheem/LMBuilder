import torch
import torch.nn as nn


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
