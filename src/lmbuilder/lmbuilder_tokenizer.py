import os
import json
from typing import Generator
from tokenizers import Tokenizer
from lmbuilder_logger import LMBuilderLogger
from utils import yaml_to_dict

# predefined tokenizer class
from tokenizers.implementations import (
    BaseTokenizer,
    BertWordPieceTokenizer,
    ByteLevelBPETokenizer,
    CharBPETokenizer,
    SentencePieceBPETokenizer,
    SentencePieceUnigramTokenizer
)

# prebuilt tokenizing models
from tokenizers.models import (
    Model,
    BPE,
    Unigram,
    WordLevel,
    WordPiece
)

# predefined normalizer
from tokenizers.normalizers import (
    Normalizer,
    BertNormalizer,
    NFC,
    NFD,
    NFKC,
    NFKD,
    Nmt,
    Sequence,
    Lowercase,
    Strip,
    StripAccents,
    Prepend,
    Precompiled,
    Replace,
)

# predefined pretokenizers
from tokenizers.pre_tokenizers import (
    PreTokenizer,
    BertPreTokenizer,
    ByteLevel,
    CharDelimiterSplit,
    Digits,
    Metaspace,
    Punctuation,
    Sequence,
    Split,
    UnicodeScripts,
    Whitespace,
    WhitespaceSplit
)

# predefined text processors
from tokenizers.processors import (
    PostProcessor,
    BertProcessing,
    ByteLevel,
    RobertaProcessing,
    Sequence,
    TemplateProcessing
)

# predefined tokenizer trainer
from tokenizers.trainers import (
    Trainer,
    BpeTrainer,
    WordLevelTrainer,
    UnigramTrainer,
    WordPieceTrainer
)

# prebuilt decoders
from tokenizers.decoders import (
    Decoder,
    ByteLevel,
    Replace,
    WordPiece,
    ByteFallback,
    Fuse,
    Strip,
    Metaspace,
    BPEDecoder,
    CTC,
    Sequence
)

TOKENIZERS = {
    "wordpiece":BertWordPieceTokenizer,
    "byte":ByteLevelBPETokenizer,
    "char":CharBPETokenizer,
    "sp_bpe":SentencePieceBPETokenizer,
    "sp_uni":SentencePieceUnigramTokenizer
}

TRAINERS = {
    "bpe":BpeTrainer,
    "wordlevel":WordLevelTrainer,
    "uni":UnigramTrainer,
    "wordpiece":WordPieceTrainer
}

MODELS = {
    "bpe": BPE,
    "uni": Unigram,
    "wordlevel": WordLevel,
    "wordpiece": WordPiece
    }

NORMALIZERS = {
    "base": Normalizer,
    "bert_norm": BertNormalizer,
    "nfc": NFC,
    "nfd": NFD,
    "nfkc":NFKC,
    "nfkd":NFKD,
    "nmt":Nmt,
    "seq":Sequence,
    "lowc":Lowercase,
    "strip":Strip,
    "strip_acc":StripAccents,
    "prpnd":Prepend,
    "pre_comp":Precompiled,
    "repl":Replace,
    }

PRETOKENIZERS = {
    "base":PreTokenizer,
    "bert":BertPreTokenizer,
    "byte":ByteLevel,
    "char":CharDelimiterSplit,
    "digit":Digits,
    "space":Metaspace,
    "punc":Punctuation,
    "seq":Sequence,
    "split":Split,
    "unicode":UnicodeScripts,
    "wspace":Whitespace,
    "wssplit":WhitespaceSplit,
}

PROCESSORS = {
    "post":PostProcessor,
    "bert":BertProcessing,
    "byte":ByteLevel,
    "roberta":RobertaProcessing,
    "seq":Sequence,
    "template":TemplateProcessing
}

DECODERS = {
    "base":Decoder,
    "byte":ByteLevel,
    "repl":Replace,
    "wordpiece":WordPiece,
    "bytefb":ByteFallback,
    "fuse":Fuse,
    "strip":Strip,
    "space":Metaspace,
    "bpe":BPEDecoder,
    "ctc":CTC,
    "seq":Sequence
}

class LMBuilderTokenizerConfigurator:
    """
    LMBuilderTokenizerConfigurator class for handling tokenizer configuration.
    """

    @staticmethod
    def load_config(config_prefix):
        """
        Load tokenizer configuration from a YAML file.

        Args:
             config_prefix  (`str`): Prefix for the configuration file.

        Returns:
            `dict`: Configuration dictionary.
        """
        config_path = os.path.join("./configs/tokenizer_configs", f"{config_prefix}.yaml")
        return yaml_to_dict(config_path)


class LMBuilderTokenizerLoader:
    """
    LMBuilderTokenizerLoader class for loading a tokenizer from a file.
    """

    @staticmethod
    def is_valid_json(file_path):
        """
        Check if a JSON file is valid.

        Args:
             file_path (`str`): Path to the JSON file.

        Returns:
            bool: True if the JSON is valid, False otherwise.
        """
        try:
            with open(file_path, 'r') as file:
                json.load(file)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def load_tokenizer(tokenizer_path):
        """
        Load a tokenizer from a JSON file.

        Args:
             tokenizer_path (`str`): Path to the tokenizer file.

        Returns:
            Tokenizer: Loaded Tokenizer instance.
        """
        try:
            if LMBuilderTokenizerLoader.is_valid_json(tokenizer_path):
                return Tokenizer.from_file(tokenizer_path)
        except FileNotFoundError:
            raise FileNotFoundError("Tokenizer file is not found in the given directory. Train the tokenizer first or check the given path is correct!")
        except Exception as e:
            print(f"An Error occurred: {e}")


class LMBuilderTokenizerTrainer:
    """
    LMBuilderTokenizerTrainer class for training a tokenizer.
    """

    @staticmethod
    def train_tokenizer(tokenizer, texts, trainer, length=None):

        """
        Train a tokenizer using the provided texts.

        Args:
             tokenizer  (Tokenizer): Tokenizer instance.
             texts  (`str`, `list`, or `Generator`): Texts for training the tokenizer.
             trainer : Trainer instance for training the tokenizer.
             length  (`int`, optional): Total number of sequence in the iterator if already known.
        
        Raises:
            ValueError: if type texts is not any one of (str, list, Generator)
        """

        if isinstance(texts, str):
            tokenizer.train([texts], trainer=trainer)
        
        elif isinstance(texts, list):
            if str(texts[0]).endswith((".txt", ".raw")):
                tokenizer.train(texts, trainer=trainer)  # if the list contains file names
            else:
                tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))  # if the list contains texts
        
        elif isinstance(texts, Generator):
            tokenizer.train_from_iterator(texts, trainer=trainer, length=length)
        else:
            raise ValueError("Train method only accepts any one of these (filepath: str, list of filepaths, list of sequence, generator which yields text sequence)")

        return tokenizer

class LMBuilderTokenizer:
    
    """
    LMBuilderTokenizer class acts as a coordinator, which orchestrates the interactions between other specialized classes for building and training tokenizers.

    Attributes:
        _tokenizer: Tokenizer instance for tokenization operations.
    """

    def __init__(self, tokenizer_path=None, vocab=None, merge=None, config_prefix=""):
        
        """
        Initialize the LMBuilderTokenizer.

        Args:
             tokenizer_path  (`str`, optional): Path to an existing or previously saved tokenizer file.
             vocab  (`str`, optional): vocab stored in-memory which is returned by LMBuilderTokenizer.from_files() method.
             merge  (`str`, optional): merge stored in-memory which is returned by LMBuilderTokenizer.from_files() method.
             config_prefix  (`str`, optional): Prefix for the configuration file.
                If provided, loads tokenizer configuration from the specified file.

        Raises:
            ValueError: if both tokenizer_path and config_prefix is None or if there's error while decoding json file or if only vocab file is given and tokenizer_type is not 'wordpiece'
            FileNotFoundError: if given tokenizer_path doesn't exist.
            Exception: if any other exception thrown.
        """
        
        if tokenizer_path is None and config_prefix == "":
            raise ValueError("Both tokenizer_path and config_prefix cannot be None. If tokenizer_path is None, then config_prefix must be passed and vice versa")

        if config_prefix is not None:
            config = LMBuilderTokenizerConfigurator.load_config(config_prefix)
            tokenizer_type = config["tokenizer_type"]
            tokenizer_kwargs = config["tokenizer_kwargs"]

        if tokenizer_path is not None:
            self._tokenizer = LMBuilderTokenizerLoader.load_tokenizer(tokenizer_path)
        
        elif (vocab is not None and merge is None):
            if not tokenizer_type == "wordpiece":
                raise ValueError("Only vocab file is passed, also only predefined class which accepts vocab only is Wordpiece. Make sure your config_prefix is correct or you passed both vocab_filename and merge_filename to 'from_files' method")
            tokenizer = TOKENIZERS[tokenizer_type](vocab, **tokenizer_kwargs)
            self._tokenizer = self.setup_tokenizer_pipeline(tokenizer, config)
        
        elif (vocab is not None and merge is not None):
            if (tokenizer_type != "custom" and tokenizer_type != "wordpiece"):
                tokenizer = TOKENIZERS[tokenizer_type](vocab, merge, **tokenizer_kwargs)
                self._tokenizer = self.setup_tokenizer_pipeline(tokenizer, config)
            else:
                raise ValueError("tokenizer_type cannot be 'custom' or 'wordpiece' when both vocab and merge are passed.")

    @classmethod
    def from_files(cls, vocab_filename=None, merge_filename=None):
        """
        Create an LMBuilderTokenizer instance from vocabulary and merge files.

        Args:
             vocab_filename (`str`): Path to the vocabulary file.
             merge_filename (`str`): Path to the merge file.

        Returns:
            LMBuilderTokenizer: An instance of LMBuilderTokenizer.

        Raises:
            ValueError: If both vocab_filename and merge_filename are None.
        """
        if (vocab_filename is not None and merge_filename is None):
            vocab = WordPiece.read_file(vocab_filename)
            return cls(tokenizer_path=None, vocab=vocab, merge=merge_filename)
        elif (vocab_filename is not None and merge_filename is not None):
            vocab, merge = BPE.read_file(vocab_filename, merge_filename)
            return cls(tokenizer_path=None, vocab=vocab, merge=merge)
        else:
            raise ValueError("Both vocab_filename and merge_filename can not be None to instantiate a tokenizer.")

    def setup_tokenizer_pipeline(self, tokenizer: Tokenizer, config):
        
        """
        Set up the tokenizer pipeline based on the provided configuration.

        Args:
             tokenizer (Tokenizer): Tokenizer instance.
             config (`dict`): Configuration dictionary.

        Returns:
            Tokenizer: Updated Tokenizer instance.
        """
        
        normalizers = config["normalizers"]
        pretokenizer = config["pretokenizer"]
        post_processor = config["post_processor"]
        decoder = config["decoder"]
        norm_kwargs = config["norm_kwargs"]
        pretok_kwargs = config["pretok_kwargs"]
        pprocessor_kwargs = config["pprocessor_kwargs"]
        decoder_kwargs = config["decoder_kwargs"]

        # setting up normalizer
        if normalizers is not None:
            if isinstance(normalizers, list):
                tok_normalizers = []
                for i, normalizer in enumerate(normalizers):
                    for k, v in NORMALIZERS.items():
                        if k == normalizer:
                            tok_normalizer = NORMALIZERS[v](**norm_kwargs[i])
                            tok_normalizers.append(tok_normalizer)
                tokenizer.normalizer = Sequence(tok_normalizers)
            elif isinstance(normalizers, str):
                if normalizers in NORMALIZERS.keys():
                    tokenizer.normalizer = NORMALIZERS[normalizers](**norm_kwargs[0])
                else:
                    raise ValueError("Wrong normalizer str. Must be one of", list(NORMALIZERS.keys()))
            else:
                raise ValueError("normalizers can only be either a list or single str")

        # setting up pretokenizer
        if pretokenizer is not None:
            tokenizer.pre_tokenizer = PRETOKENIZERS[pretokenizer](**pretok_kwargs)

        # setting up tokenizer decoder
        if decoder is not None:
            tokenizer.decoder = DECODERS[decoder](**decoder_kwargs)

        if post_processor is not None:
            tokenizer.post_processor = PROCESSORS[post_processor](**pprocessor_kwargs)

        return tokenizer

    @staticmethod
    def build_tokenizer_from_config(
            tokenizer_type,
            tokenizer_kwargs={},
            model=None,
            normalizers=None,
            pretokenizer=None,
            post_processor=None,
            decoder=None,
            model_kwargs={},
            norm_kwargs=[{}],
            pretok_kwargs={},
            processor_kwargs={},
            trainer_kwargs={},
            decoder_kwargs={},
    ):
        """
        Build a tokenizer based on the provided configuration.

        Args:
             tokenizer_type  (`str`): Type of tokenizer (e.g., 'wordpiece', 'byte', 'char', 'sp_bpe', 'sp_uni', 'custom').
             tokenizer_kwargs  (`dict`): Keyword arguments for your desired tokenizer class initialization. {} if tokenizer_type is 'custom'
             model  (`str`):  core tokenizer model to use.
             normalizers  (`list` or str): List of normalizers or a single normalizer.
             pretokenizer  (`str`): Type of pretokenizer to use.
             post_processor  (`str`): Type of post-processor to use.
             decoder  (`str`): Type of decoder to use.
             model_kwargs  (`dict`): Keyword arguments for model initialization.
             norm_kwargs  (`list`): List of keyword arguments for normalizers.
             pretok_kwargs  (`dict`): Keyword arguments for pretokenizer.
             processor_kwargs  (`dict`): Keyword arguments for post-processor.
             trainer_kwargs  (`dict`): Keyword arguments for trainer.
             decoder_kwargs  (`dict`): Keyword arguments for decoder.

        Returns:
            Tuple[Tokenizer, Trainer]: Tokenizer and Trainer instances.

        Raises:
            NotImplementedError: If the tokenizer type is not supported.
        """

        if tokenizer_type in TOKENIZERS.keys():
            if isinstance(tokenizer_kwargs, dict):
                tokenizer = TOKENIZERS[tokenizer_type](**tokenizer_kwargs)
            else:
                raise ValueError("'tokenizer_kwargs' must be a dictionary type.")
        elif tokenizer_type == "custom":
            # setting up model
            if model is not None:  # kwargs can be an empty dict to use default class args
                tokenizer = Tokenizer(MODELS[model](**model_kwargs))
            else:
                raise ValueError(f"model cannot be None for a custom tokenizer type. it must be one of {list(TOKENIZERS.keys())}")

            # setting up normalizer
            if normalizers is not None:
                if isinstance(normalizers, list):
                    tok_normalizers = []
                    for i, normalizer in enumerate(normalizers):
                        for k, v in NORMALIZERS.items():
                            if k == normalizer:
                                tok_normalizer = NORMALIZERS[v](**norm_kwargs[i])
                                tok_normalizers.append(tok_normalizer)
                    tokenizer.normalizer = Sequence(tok_normalizers)
                elif isinstance(normalizers, str):
                    if normalizers in [NORMALIZERS.keys()]:
                        tokenizer.normalizer = NORMALIZERS[normalizers](**norm_kwargs[0])
                    else:
                        raise ValueError("Wrong normalizer str. Must be one of", list(NORMALIZERS.keys()))
                else:
                    raise ValueError("normalizers can only be either a list or single str")

            # setting up pretokenizer
            if pretokenizer is not None:
                tokenizer.pre_tokenizer = PRETOKENIZERS[pretokenizer](**pretok_kwargs)

            # setting up tokenizer decoder
            if decoder is not None:
                tokenizer.decoder = DECODERS[decoder](**decoder_kwargs)

            if post_processor is not None:
                tokenizer.post_processor = PROCESSORS[post_processor](**processor_kwargs)

            trainer = TRAINERS[model](**trainer_kwargs)

        else:
            raise NotImplementedError("No other methods for initiating a tokenizer are currently implemented. it must be either predefined or custom.")

        return tokenizer, trainer

    @staticmethod
    def train_and_save_tokenizer(
            texts,
            ckpt_dir: str,
            config_prefix: str = "custom",
            tokenizer_prefix="lmbuilder_tokenizer",
            length=None,
            save_model=False,
        ):

        """
        Train a tokenizer using the provided texts and save it to the `ckpt_dir` directory with `config_prefix` as filename.

        Args:
            texts  (`str`, `list`, or `Generator`): Texts for training the tokenizer.
            ckpt_dir  (`str`): Directory to save the trained tokenizer.
            config_prefix  (`str`): Prefix for the configuration file in tokenizer_configs directory.
            tokenizer_prefix  (`str`): Prefix for the tokenizer file.
            length  (`int`, optional): Total number of Sequence in the iterator if already known.
            save_model (bool): Whether to save model files (vocab.json and merge.txt)
             
        Raises:
            ValueError: If the configuration file is not found or if there's an issue with the tokenizer training.
        """

        # build the tokenizer from config
        config = LMBuilderTokenizerConfigurator.load_config(config_prefix)
        tokenizer, tok_trainer = LMBuilderTokenizer.build_tokenizer_from_config(**config)

        # train the tokenizer
        tokenizer = LMBuilderTokenizerTrainer.train_tokenizer(tokenizer, texts, tok_trainer, length)

        # save the trained tokenizer
        if save_model:
            tokenizer.model.save(ckpt_dir, tokenizer_prefix)
        else:
            tokenizer.save(os.path.join(ckpt_dir, f"{tokenizer_prefix}.json"), pretty=True)
        

    def tokenize(self, text): 
        """
        Tokenize the given text using the configured tokenizer.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[str]: List of tokens.
        """
        return self._tokenizer.encode(text).tokens
