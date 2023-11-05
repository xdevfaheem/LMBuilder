import torch
from torch import nn
from llm_config import LLMConfig
from embedding_layer import EmbeddingLayer, VannilaPositionalEncoding
from layernorm import LayerNormWithBias
from llm_block import RegressiveBlock
from llm_builder.utils import number_to_words
import copy
import math
from torchinfo import summary
from prettytable import PrettyTable

class LLM(nn.Module):

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

        self.final_layer_norm = LayerNormWithBias(embedding_dim, config.bias)

        self.decoder_layers = nn.ModuleList([RegressiveBlock(
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
        print(summary(model, input_data=[review_batch, mask_batch)]))


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
    config = LLMConfig(embedding_dim=768)
    model = LLM(config)
    x = torch.randint(0, 8000, (2, 1024), dtype=torch.int64)
    print(f"Model Input: {x}\nModel Input's Shape: {x.shape}")
    out, _ = model(x)
    print(f"Model's Output: {out}\nModel Output's Shape: {out.shape}")
