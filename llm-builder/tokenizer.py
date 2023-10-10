import os
from pathlib import Path
from typing import Optional, List
import torch
import tiktoken
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
import youtokentome as yttm
import multiprocessing

class Tokenizer:

    def __init__(self,
                 model_path: Path = None,
                 youtokenizer=False,
                 tiktokenizer=True,
                 sp_tokenizer=False,
                 tik_encoding="cl100k_base"
                ) -> None:

        """
        Tokenizer class for text tokenization using various tokenization backends.
    
        Args:
            model_path (Path, optional): Path to the pre-trained tokenizer model.
            youtokenizer (bool): Use YouTokenToMe tokenizer if True.
            tiktokenizer (bool): Use Tiktoken tokenizer if True.
            sp_tokenizer (bool): Use SentencePiece tokenizer if True.
            tik_encoding (str): Tiktoken encoding mode (default: "cl100k_base").
    
        Attributes:
            _processor : Instance of the tokenizer class from the backend lib.
            backend (str): Tokenizer backend ("youtoken", "sentencepiece", or "tiktoken").
        """
        
        # Initialize the tokenizer based on the selected backend.
        if model_path is not None and youtokenizer:
            self._processor = yttm.BPE(model=str(model_path), n_threads=os.cpu_count())
            self.backend = "youtoken"
        elif model_path is not None and sp_tokenizer:
            self._processor = SentencePieceProcessor(model_file=str(model_path))
            self.backend = "sentencepiece"
            self.bos_id = self._processor.bos_id()
            self.eos_id = self._processor.eos_id()
        elif model_path is None and tiktokenizer is True:
            self.backend = "tiktoken"
            cl100k_base = tiktoken.get_encoding(tik_encoding)
            self._processor = tiktoken.Encoding(
                        name="cl100k_im",
                        pat_str=cl100k_base._pat_str,
                        mergeable_ranks=cl100k_base._mergeable_ranks,
                        special_tokens={
                            **cl100k_base._special_tokens,
                            "": 100261,
                        }
                    )
            self.bos_id = self.token_to_id("")
            self.eos_id = self._processor.eot_token
        else:
            raise NotImplementedError("No Other Backend Implemented Now!")

    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size of the tokenizer.

        Returns:
            int: Vocabulary size.
        """
        if self.backend == "youtoken":
            return self._processor.vocab_size()
        elif self.backend == "sentencepiece":
            return self._processor.get_piece_size()
        elif self.backend == "tiktoken":
            return self._processor.max_token_value

    def token_to_id(self, token: str) -> int:
        """
        Convert a token to its corresponding ID.

        Args:
            token (str): Token to convert.

        Returns:
            int: Token ID.
        """
        if self.backend == "sentencepiece":
            id_ = self._processor.piece_to_id(token)
        elif self.backend == "youtoken":
            id_ = self._processor.subword_to_id(token)
        elif self.backend == "tiktoken":
            id_ = self._processor.encode_single_token(token)
        else:
            raise RuntimeError("Unsupported Backend!")
        return id_

    def id_to_token(self, id: int) -> str:
        """
        Convert an ID to its corresponding token.

        Args:
            id (int): Token ID.

        Returns:
            str: Token.
        """
        if self.backend == "sentencepiece":
            tok_ = self._processor.id_to_piece(id)
        elif self.backend == "youtoken":
            tok_ = self._processor.id_to_subword(id)
        elif self.backend == "tiktoken":
            tok_ = self._processor.decode_single_token_bytes(id)
            tok_.decode("utf-8")
        else:
            raise RuntimeError("Unsupported Backend!")
        if tok_ is None:
            raise ValueError(f"id {id!r} not found in the vocab.")
        return tok_

    def encode(
        self,
        string: str,
        bos: bool = True,
        eos: bool = True,
        max_length: int = -1,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode a text string into a sequence of token IDs.

        Args:
            string (str): Input text string.
            bos (bool): Whether to prepend a beginning-of-sequence token.
            eos (bool): Whether to append an end-of-sequence token.
            max_length (int): Maximum sequence length (excluding special tokens).
            device (Optional[torch.device]): Device to store the resulting tensor.

        Returns:
            torch.Tensor: Encoded token IDs as a tensor.
        """
        if self.backend == "sentencepiece":
            tokens = self._processor.encode(string, out_type=int)
            if max_length > 0:
                tokens = tokens[:max_length-(2 if bos and eos else 1 if bos or eos else 0)]
            if bos and eos:
                tokens = [self.bos_id] + tokens + [self.eos_id]
            elif eos:
                tokens = tokens + [self.eos_id]
            elif bos:
                tokens = [self.bos_id] + tokens
        elif self.backend == "youtoken":
            if bos and eos:
                tokens = self._processor.encode(string, output_type=yttm.OutputType.ID, bos=True, eos=True)
            elif bos:
                tokens = self._processor.encode(string, output_type=yttm.OutputType.ID, bos=True, eos=False)
            elif eos:
                tokens = self._processor.encode(string, output_type=yttm.OutputType.ID, bos=False, eos=True)
            if max_length > 0:
                tokens = [*([tokens[0]] if bos else [])] + tokens[(1 if bos else 0):(-1 if eos else None)][:max_length-(2 if bos and eos else 1 if bos or eos else 0)] + [*([tokens[-1]] if eos else [])]
        elif self.backend == "tiktoken":
            tokens = self._processor.encode_ordinary(string)
            if max_length > 0:
                tokens = tokens[:max_length-(2 if bos and eos else 1 if bos or eos else 0)]
            if bos and eos:
                tokens = [self.bos_id] + tokens + [self.eos_id]
            elif bos:
                tokens = [self.bos_id] + tokens
            elif eos:
                tokens = tokens + [self.eos_id]
        else:
            raise RuntimeError("Unsupported Backend!")
        return torch.tensor(tokens, dtype=torch.int32, device= "cpu" if device is None else device)

    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode a sequence of token IDs into a text string.

        Args:
            tokens (torch.Tensor): Input token IDs as a tensor.

        Returns:
            str: Decoded text string.
        """
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self._processor.decode(tokens)

    @staticmethod
    def train(input_txt_path: str, destination: str, youtokenizer=True, sp_tokenizer=False, vocab_file_prefix='vocab', vocab_size=16000, pad_id=3, unk_id=2, eos_id=1, bos_id=0) -> None:
        """
        Train a tokenizer model from a text file using YouTokenToMe or SentencePiece.

        Args:
            input_txt_path (str): Path to the input text file.
            destination (str): Directory where the tokenizer model will be saved.
            youtokenizer (bool): Use YouTokenToMe tokenizer if True.
            sp_tokenizer (bool): Use SentencePiece tokenizer if True.
            vocab_file_prefix (str): Prefix for the generated vocabulary model files.
            vocab_size (int): Vocabulary size for the tokenizer.
            pad_id (int): ID for padding token.
            unk_id (int): ID for unknown token.
            eos_id (int): ID for end-of-sequence token.
            bos_id (int): ID for beginning-of-sequence token.
        """
        if sp_tokenizer:
            vocab_model_prefix = os.path.join(destination, f"{vocab_file_prefix}_{vocab_size}")
            SentencePieceTrainer.Train(input=input_txt_path, model_prefix=vocab_model_prefix, vocab_size=vocab_size, model_type="bpe", pad_id=pad_id, eos_id=eos_id, unk_id=unk_id, bos_id=bos_id)
        elif youtokenizer:
            vocab_model_prefix = os.path.join(destination, f"{vocab_file_prefix}_{vocab_size}.model")
            _ = yttm.BPE.train(data=input_txt_path, model=vocab_model_prefix, n_threads=multiprocessing.cpu_count(), vocab_size=vocab_size, coverage=0.9999, pad_id=pad_id, eos_id=eos_id, unk_id=unk_id, bos_id=bos_id)
            return None
