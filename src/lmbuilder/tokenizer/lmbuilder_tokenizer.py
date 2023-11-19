import os
import json
from typing import Type, Generator, Optional, Dict, Tuple, List, Union, Iterator
from torch import Tensor
from tokenizers import AddedToken, Encoding, Tokenizer
from tokenizers.implementations import BaseTokenizer
from tokenizers.models import Model, BPE, WordPiece
from tokenizers.normalizers import Normalizer, Sequence
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
from tokenizers.trainers import Trainer
from tokenizers.decoders import Decoder
import custom_comps
from lmbuilder.utils import yaml_to_dict   
from lmbuilder.tokenizer.utils import TOKENIZERS, TRAINERS, MODELS, NORMALIZERS, PRETOKENIZERS, PROCESSORS, DECODERS


class LMBuilderTokenizerIO:
    """
    LMBuilderTokenizerIO class for saving/loading a tokenizer and it's configuration to/from a file.
    """

    @staticmethod
    def is_valid_json(file_path: str):
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
    def load_config(config_prefix: str):
        """
        Load tokenizer configuration from a YAML file.

        Args:
             config_prefix  (`str`): Prefix for the configuration file.

        Returns:
            `dict`: Configuration dictionary.
        """
        config_path = os.path.join("./configs/tokenizer_configs", f"{config_prefix}.yaml")
        return yaml_to_dict(config_path)

    @staticmethod
    def load_tokenizer(tokenizer_path: str) -> Type[Tokenizer]:
        """
        Load a tokenizer from a JSON file.

        Args:
            tokenizer_path (`str`):
                Path to the tokenizer file.

        Returns:
            Tokenizer: Loaded Tokenizer instance.
        """
        try:
            if LMBuilderTokenizerIO.is_valid_json(tokenizer_path):
                tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)
                return tokenizer
        except FileNotFoundError:
            raise FileNotFoundError("Tokenizer file is not found in the given directory. Train the tokenizer first or check the given path is correct!")
        except Exception as e:
            print(f"An Error occurred: {e}")

    @staticmethod
    def save_model(tokenizer: Tokenizer, directory: str, prefix: Optional[str] = None) -> List[str]:
        """Save the current model to the given directory

        Args:
            directory: str:
                A path to the destination directory

            prefix: (Optional) str:
                An optional prefix, used to prefix each file name
        """
        return tokenizer.model.save(directory, prefix=prefix)

    @staticmethod
    def tokenizer_to_str(tokenizer: Tokenizer, pretty: Optional[bool] = False)-> str:
        """Get a serialized JSON version of the Tokenizer as a str

        Args:
            pretty: bool:
                Whether the JSON string should be prettified

        Returns:
            str
        """
        return tokenizer.to_str(pretty)
    
    @staticmethod
    def save_tokenizer(tokenizer: Tokenizer, path: str, pretty: Optional[bool] = True) -> None:
        """Save the current Tokenizer at the given path

        Args:
            path: str:
                A path to the destination Tokenizer file
        """
        return tokenizer.save(path, pretty)
    
    @staticmethod
    def save(
            tokenizer: Tokenizer,
            ckpt_dir: str, 
            tokenizer_prefix: Optional[str] = "lnbuilder_tokenizer",
            pretify: Optional[bool] = True, 
            save_method: Optional[str]="file",
        ) -> Union[str, List[str], None]:

        """Save the current Tokenizer at the given path

        Args:
            tokenizer: Tokenizer: 
                Tokenizer object to be saved.
            ckpt_dir: `str`:
                Path to the destination checkpoint directory where tokenizer files will be saved.
            tokenizer_prefix: `str`:
                Prefix for tokenizer files
            save_method: str
                Any one of the ('model', 'string' and 'file')
                `model`: to save tokenizer's core model as two files (vocab, merge)
                `string`: to serialize the JSON version of tokenizer as string.
                `file`: to save entire tokenizer and it's configuration into one file.
            `prefix`: bool
                whether the tokenizer file should be prettified.
        """
        
        if save_method == "model":
            return LMBuilderTokenizerIO.save_model(tokenizer, ckpt_dir, prefix=tokenizer_prefix)
        
        elif save_method == "string":
            return LMBuilderTokenizerIO.tokenizer_to_str(tokenizer, pretty=pretify)
        
        elif save_method == "file":
            path = os.path.join(ckpt_dir, f"{tokenizer_prefix}.json")
            tokenizer.save(path, pretty=pretify)
            return path
        else:
            raise ValueError("`save_method` can only be any one of ('model', 'string' and 'file')")

class LMBuilderTokenizerTrainer:
    """
    LMBuilderTokenizerTrainer class for training a tokenizer.
    """

    @staticmethod
    def train_tokenizer(
            tokenizer: Tokenizer,
            texts: Union[str, List[str], Iterator[str]],
            trainer: Trainer,
            length: Optional[int]=None
        ):

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


class LMBuilderTokenizerBuilder:
    """
    LMBuilderTokenizerBuilder class for building and setting up tokenizers. It supports both predefined tokenizers and custom tokenizers
    """

    @staticmethod
    def build_tokenizer_from_config(
            tokenizer_type,
            tokenizer_kwargs: Optional[Dict]={},
            model: Optional[str]=None,
            normalizers: Optional[str]=None,
            pretokenizer: Optional[str]=None,
            post_processor: Optional[str]=None,
            decoder: Optional[str]=None,
            model_kwargs: Optional[Dict]={},
            norm_kwargs: List[Optional[Dict]]=[{}],
            pretok_kwargs: Optional[Dict]={},
            processor_kwargs: Optional[Dict]={},
            trainer_kwargs: Optional[Dict]={},
            decoder_kwargs: Optional[Dict]={},
    ) -> Tuple[Type[Tokenizer], Type[Trainer]]:
        
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
                tokenizer: BaseTokenizer = TOKENIZERS[tokenizer_type](**tokenizer_kwargs)
            else:
                raise ValueError("'tokenizer_kwargs' must be a dictionary type.")
        
        elif tokenizer_type == "custom":
            # setting up model
            if model is not None:  # kwargs can be an empty dict to use default class args
                tokenizer_model: Model = MODELS[model](**model_kwargs)
                tokenizer = Tokenizer(tokenizer_model)
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
                    tokenizer.normalizer: Normalizer = Sequence(tok_normalizers)
                
                elif isinstance(normalizers, str):
                    
                    if normalizers in [NORMALIZERS.keys()]:
                        tokenizer.normalizer: Normalizer = NORMALIZERS[normalizers](**norm_kwargs[0])
                    
                    elif (normalizer_cls := getattr(custom_comps, normalizers)) is not None:
                        tokenizer.normalizer: Normalizer = Normalizer.custom(normalizer_cls(**norm_kwargs[0]))
                    
                    else:
                        raise ValueError(f"Given 'normalizer' string is neither one of {list(NORMALIZERS.keys())} nor a custom class present in `custom_comps.py`")
            
                else:
                    raise ValueError("normalizers can only be either a list or single str")

            # setting up pretokenizer
            if (pretokenizer is not None and isinstance(pretokenizer, str)):
                if pretokenizer in [PRETOKENIZERS.keys()]:
                    tokenizer.pre_tokenizer: PreTokenizer = PRETOKENIZERS[pretokenizer](**pretok_kwargs)
                elif (pretokenizer_cls := getattr(custom_comps, pretokenizer)) is not None:
                    tokenizer.pre_tokenizer: PreTokenizer = PreTokenizer.custom(pretokenizer_cls(**pretok_kwargs))
                    

            # setting up tokenizer decoder
            if (decoder is not None and isinstance(decoder, str)):
                if decoder in [DECODERS.keys()]:
                    tokenizer.decoder: Decoder = DECODERS[decoder](**decoder_kwargs)
                elif (decoder_cls := getattr(custom_comps, decoder)) is not None:
                    tokenizer.decoder: Decoder = Decoder.custom(decoder_cls(**decoder_kwargs))

            
            if (post_processor is not None and isinstance(post_processor, str)):
                if post_processor in [PROCESSORS.keys()]:
                    tokenizer.post_processor: PostProcessor = PROCESSORS[post_processor](**processor_kwargs)
                elif (processor_cls := getattr(custom_comps, post_processor)) is not None:
                    tokenizer.post_processor: PostProcessor = PostProcessor.custom(processor_cls(**processor_kwargs))

            trainer: Trainer = TRAINERS[model](**trainer_kwargs)

        else:
            raise NotImplementedError("No other methods for initiating a tokenizer are currently implemented. it must be either predefined or custom.")

        return tokenizer, trainer

    @staticmethod
    def setup_tokenizer_pipeline(tokenizer: Tokenizer, config: Dict) -> Type[Tokenizer]:
        
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
                tokenizer.normalizer: Normalizer = Sequence(tok_normalizers)
            
            elif isinstance(normalizers, str):
                
                if normalizers in [NORMALIZERS.keys()]:
                    tokenizer.normalizer: Normalizer = NORMALIZERS[normalizers](**norm_kwargs[0])
                
                elif (normalizer_cls := getattr(custom_comps, normalizers)) is not None:
                    tokenizer.normalizer: Normalizer = Normalizer.custom(normalizer_cls(**norm_kwargs[0]))
                
                else:
                    raise ValueError(f"Given 'normalizer' string is neither one of {list(NORMALIZERS.keys())} nor a custom class present in `custom_comps.py`")
        
            else:
                raise ValueError("normalizers can only be either a list or single str")

        # setting up pretokenizer
        if (pretokenizer is not None and isinstance(pretokenizer, str)):
            if pretokenizer in [PRETOKENIZERS.keys()]:
                tokenizer.pre_tokenizer: PreTokenizer = PRETOKENIZERS[pretokenizer](**pretok_kwargs)
            elif (pretokenizer_cls := getattr(custom_comps, pretokenizer)) is not None:
                tokenizer.pre_tokenizer: PreTokenizer = PreTokenizer.custom(pretokenizer_cls(**pretok_kwargs))
                

        # setting up tokenizer decoder
        if (decoder is not None and isinstance(decoder, str)):
            if decoder in [DECODERS.keys()]:
                tokenizer.decoder: Decoder = DECODERS[decoder](**decoder_kwargs)
            elif (decoder_cls := getattr(custom_comps, decoder)) is not None:
                tokenizer.decoder: Decoder = Decoder.custom(decoder_cls(**decoder_kwargs))
                
        
        if (post_processor is not None and isinstance(post_processor, str)):
            if post_processor in [PROCESSORS.keys()]:
                tokenizer.post_processor: PostProcessor = PROCESSORS[post_processor](**pprocessor_kwargs)
            elif (processor_cls := getattr(custom_comps, post_processor)) is not None:
                tokenizer.post_processor: PostProcessor = PostProcessor.custom(processor_cls(**pprocessor_kwargs))

        return tokenizer


class LMBuilderTokenizer:
    
    """
    LMBuilderTokenizer class acts as a coordinator, which orchestrates the interactions between other specialized classes for building and training tokenizers.

    Attributes:
        _tokenizer: Tokenizer instance for tokenization operations.
    """

    def __init__(
            self,
            config_prefix: str,
            tokenizer_path: Optional[str] = None,
            vocab: Optional[Union[str, Dict[str, int]]] = None,
            merge: Optional[Union[str, Dict[Tuple[int, int], Tuple[int, int]]]] = None
        ) -> object:
        
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

        config = LMBuilderTokenizerIO.load_config(config_prefix)
        tokenizer_type = config["tokenizer_type"]
        tokenizer_kwargs = config["tokenizer_kwargs"]

        if tokenizer_path is not None:
            self._tokenizer: Tokenizer = LMBuilderTokenizerIO.load_tokenizer(tokenizer_path)
        
        elif (vocab is not None and merge is None):
            if not tokenizer_type == "wordpiece":
                raise ValueError("Only vocab file is passed, also only predefined class which accepts vocab only is Wordpiece. Make sure your config_prefix is correct or you passed both vocab_filename and merge_filename to 'from_files' method")
            tokenizer = TOKENIZERS[tokenizer_type](vocab, **tokenizer_kwargs)
            self._tokenizer: Tokenizer = LMBuilderTokenizerBuilder.setup_tokenizer_pipeline(tokenizer, config)
        
        elif (vocab is not None and merge is not None):
            if (tokenizer_type != "custom" and tokenizer_type != "wordpiece"):
                tokenizer = TOKENIZERS[tokenizer_type](vocab, merge, **tokenizer_kwargs)
                self._tokenizer: Tokenizer = LMBuilderTokenizerBuilder.setup_tokenizer_pipeline(tokenizer, config)
            else:
                raise ValueError("tokenizer_type cannot be 'custom' or 'wordpiece' when both vocab and merge are passed.")
        
        # setting up global padding params
        if config["pad"]:
            self.enable_padding(**config["padding_args"])
        else:
            self.disable_padding()

        # NOTE: You can always enable/disable padding/truncation after initialization. that's why it's defined as this class's method.

        # setting up global truncate params
        if config["truncate"]:
            self.enable_truncation(**config["trucate_args"])
        else:
            self.disable_truncation()

        # Let the tokenizer know about special tokens if they are part of the vocab
        for spl_token in config["special_tokens"]:
            if self.token_to_id(str(spl_token)) is not None:
                self.add_special_tokens([str(spl_token)])

    @classmethod
    def from_files(cls, config_prefix: str, vocab_filename: Optional[str]=None, merge_filename: Optional[str]=None) -> Type["LMBuilderTokenizer"]:
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
            return cls(config_prefix, vocab=vocab, merge=merge_filename)
        
        elif (vocab_filename is not None and merge_filename is not None):
            vocab, merge = BPE.read_file(vocab_filename, merges=merge_filename)
            return cls(config_prefix, vocab=vocab, merge=merge)
        
        else:
            raise ValueError("Both vocab_filename and merge_filename can not be None to instantiate a tokenizer.")

    @staticmethod
    def train_and_save_tokenizer(
            texts: Union[str, List[str], Iterator[str]],
            ckpt_dir: str,
            config_prefix: Optional[str] = "custom",
            length: Optional[int]=None,
        ) -> None:

        """
        Train a tokenizer using the provided texts and save it to the `ckpt_dir` directory with `config_prefix` as filename.

        Args:
            texts  (`str`, `list`, or `Generator`): Texts for training the tokenizer.
            ckpt_dir  (`str`): Directory to save the trained tokenizer.
            config_prefix  (`str`): Prefix for the configuration file in tokenizer_configs directory.
            tokenizer_prefix  (`str`): Prefix for the tokenizer file.
            length  (`int`, optional): Total number of Sequence in the iterator if already known.
             
        Raises:
            ValueError: If the configuration file is not found or if there's an issue with the tokenizer training.
        """
        try:
            # build the tokenizer from config
            config = LMBuilderTokenizerIO.load_config(config_prefix)
            tokenizer, tok_trainer = LMBuilderTokenizerBuilder.build_tokenizer_from_config(**config)

            # train the tokenizer
            tokenizer = LMBuilderTokenizerTrainer.train_tokenizer(tokenizer, texts, tok_trainer, length)
        
        except ValueError as e:
            print("Error while training: ", e)

        except NotImplementedError as e:
            print("Unsupported `tokenizer_type`: ", e)

        # save the trained tokenizer
        LMBuilderTokenizerIO.save(tokenizer, ckpt_dir, config["tokenizer_prefix"], prettify=True, save_method=config["save_method"])

    def enable_padding(
        self,
        direction: Optional[str] = "right",
        pad_to_multiple_of: Optional[int] = None,
        pad_id: Optional[int] = 0,
        pad_type_id: Optional[int] = 0,
        pad_token: Optional[str] = "[PAD]",
        length: Optional[int] = None,
    )-> None:
        """Change the padding strategy

        Args:
            direction: (`optional`) str:
                Can be one of: `right` or `left`

            pad_to_multiple_of: (`optional`) unsigned int:
                If specified, the padding length should always snap to the next multiple of
                the given value. For example if we were going to pad with a length of 250 but
                `pad_to_multiple_of=8` then we will pad to 256.

            pad_id: (`optional`) unsigned int:
                The indice to be used when padding

            pad_type_id: (`optional`) unsigned int:
                The type indice to be used when padding

            pad_token: (`optional`) str:
                The pad token to be used when padding

            length: (`optional`) unsigned int:
                If specified, the length at which to pad. If not specified
                we pad using the size of the longest sequence in a batch
        """
        return self._tokenizer.enable_padding(
            direction=direction,
            pad_to_multiple_of=pad_to_multiple_of,
            pad_id=pad_id,
            pad_type_id=pad_type_id,
            pad_token=pad_token,
            length=length,
        )

    def disable_padding(self) -> None:
        
        """Disable padding"""
        return self._tokenizer.no_padding()

    def enable_truncation(
            self,
            max_length: int,
            stride: Optional[int] = 0,
            strategy: Optional[str] = "longest_first"
        ):
        """Change the truncation options

        Args:
            max_length: unsigned int:
                The maximum length at which to truncate

            stride: (`optional`) unsigned int:
                The length of the previous first sequence to be included
                in the overflowing sequence

            strategy: (`optional`) str:
                Can be one of `longest_first`, `only_first` or `only_second`
        """
        return self._tokenizer.enable_truncation(max_length, stride=stride, strategy=strategy)

    def disable_truncation(self):
        
        """Disable truncation"""
        return self._tokenizer.no_truncation()
    
    def add_special_tokens(self, special_tokens: List[Union[str, AddedToken]]) -> int:
        """Add the given special tokens to the vocabulary, and treat them as special tokens.

        The special tokens will never be processed by the model, and will be
        removed while decoding.

        Args:
            tokens: List[Union[str, AddedToken]]:
                A list of special tokens to add to the vocabulary. Each token can either be
                a string, or an instance of AddedToken

        Returns:
            The number of tokens that were added to the vocabulary
        """
        return self._tokenizer.add_special_tokens(special_tokens)
    
    def normalize(self, sequence: str) -> str:
        """Normalize the given sequence

        Args:
            sequence: str:
                The sequence to normalize

        Returns:
            The normalized string
        """
        return self._tokenizer.normalize(sequence)
    
    def post_process(
        self,
        encoding: Encoding,
        pair: Optional[Encoding] = None,
        add_special_tokens: Optional[bool] = True
    ) -> Type[Encoding]:

        
        """Apply all the post-processing steps to the given encodings.

        The various steps are:
            1. Truncate according to global params (provided to `enable_truncation`)
            2. Apply the PostProcessor
            3. Pad according to global params. (provided to `enable_padding`)

        Args:
            encoding: Encoding:
                The main Encoding to post process

            pair: Optional[Encoding]:
                An optional pair Encoding

            add_special_tokens: bool:
                Whether to add special tokens

        Returns:
            The resulting Encoding
        """
        return self._tokenizer.post_process(encoding, pair, add_special_tokens)

    def encode(
        self,
        sequence: str,
        pair: Optional[str] = None,
        is_pretokenized: Optional[bool] = False,
        add_special_tokens: Optional[bool] = True,
    ) -> Type[Encoding]:
        
        """
        Encode the given sequence or pair. This method can process raw text sequences as well
        as already pre-tokenized sequences.

        Args:
            sequence: InputSequence:
                The sequence we want to encode. This sequence can be either raw text or
                pre-tokenized, according to the `is_pretokenized` argument:

                - If `is_pretokenized=False`: `InputSequence` is expected to be `str`
                - If `is_pretokenized=True`: `InputSequence` is expected to be
                    `Union[List[str], Tuple[str]]`

            is_pretokenized: bool:
                Whether the input is already pre-tokenized.

            add_special_tokens: bool:
                Whether to add the special tokens while encoding.

        Returns:
            An Encoding
        """
        if sequence is None:
            raise ValueError("encode: `sequence` can't be `None`")

        return self._tokenizer.encode(sequence, pair, is_pretokenized, add_special_tokens)

    def encode_batch(
        self,
        inputs: List[List[str]],
        is_pretokenized: Optional[bool] = False,
        add_special_tokens: Optional[bool] = True,
    ) -> List[Type[Encoding]]:
        """Encode the given inputs. This method accept both raw text sequences as well as already
        pre-tokenized sequences.

        Args:
            inputs: List[EncodeInput]:
                A list of single sequences or pair sequences to encode. Each `EncodeInput` is
                expected to be of the following form:
                    `Union[InputSequence, Tuple[InputSequence, InputSequence]]`

                Each `InputSequence` can either be raw text or pre-tokenized,
                according to the `is_pretokenized` argument:

                - If `is_pretokenized=False`: `InputSequence` is expected to be `str`
                - If `is_pretokenized=True`: `InputSequence` is expected to be
                    `Union[List[str], Tuple[str]]`

            is_pretokenized: bool:
                Whether the input is already pre-tokenized.

            add_special_tokens: bool:
                Whether to add the special tokens while encoding.

        Returns:
            A list of Encoding
        """

        if inputs is None:
            raise ValueError("encode_batch: `inputs` can't be `None`")

        return self._tokenizer.encode_batch(inputs, is_pretokenized, add_special_tokens)

    def decode(self, ids: Tensor, skip_special_tokens: Optional[bool] = True) -> str:
        """Decode the given list of ids to a string sequence

        Args:
            ids: Tensor:
                A list of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output string

        Returns:
            The decoded string
        """
        if ids is None:
            raise ValueError("None input is not valid. Should be a list of integers.")

        tokens = [ids.item()] if ids.ndim == 0 else ids.tolist()

        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, sequences: List[List[Tensor]], skip_special_tokens: Optional[bool] = True) -> str:
        """Decode the list of sequences to a list of string sequences

        Args:
            sequences: List[List[unsigned int]]:
                A list of sequence of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output strings

        Returns:
            A list of decoded strings
        """
        if sequences is None:
            raise ValueError("None input is not valid. Should be list of list of integers.")
        
        sequences = [[ids.item()] if ids.ndim == 0 else ids.tolist() for ids in sequences]

        return self._tokenizer.decode_batch(sequences, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert the given token to its corresponding id

        Args:
            token: str:
                The token to convert

        Returns:
            The corresponding id if it exists, None otherwise
        """
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        """Convert the given token id to its corresponding string

        Args:
            token: id:
                The token id to convert

        Returns:
            The corresponding string if it exists, None otherwise
        """
        return self._tokenizer.id_to_token(id)
    
    @property
    def vocab_size(self) -> int:
        """Return the size of vocabulary, with added tokens"""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def padding(self) -> Optional[dict]:
        """Get the current padding parameters

        Returns:
            None if padding is disabled, a dict with the currently set parameters
            if the padding is enabled.
        """
        return self._tokenizer.padding

    @property
    def truncation(self) -> Optional[dict]:
        """Get the current truncation parameters

        Returns:
            None if truncation is disabled, a dict with the current truncation parameters if
            truncation is enabled
        """
        return self._tokenizer.truncation
    
    @property
    def model(self) -> Model:
        return self._tokenizer.model
    
    @property
    def normalizer(self) -> Normalizer:
        return self._tokenizer.normalizer

    @property
    def pre_tokenizer(self) -> PreTokenizer:
        return self._tokenizer.pre_tokenizer

    @property
    def post_processor(self) -> PostProcessor:
        return self._tokenizer.post_processor

    @property
    def decoder(self) -> Decoder:
        return self._tokenizer.decoder