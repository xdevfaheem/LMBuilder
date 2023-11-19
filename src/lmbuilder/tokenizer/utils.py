# Utility module which contain utility dictionary which maps custom strings to specialized classes

# predefined tokenizer class
from tokenizers.implementations import (
    BertWordPieceTokenizer,
    ByteLevelBPETokenizer,
    CharBPETokenizer,
    SentencePieceBPETokenizer,
    SentencePieceUnigramTokenizer
)

# prebuilt tokenizing models
from tokenizers.models import (
    BPE,
    Unigram,
    WordLevel,
    WordPiece
)

# predefined normalizer
from tokenizers.normalizers import (
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
    "bert_norm": BertNormalizer,
    "nfc": NFC,
    "nfd": NFD,
    "nfkc":NFKC,
    "nfkd":NFKD,
    "nmt":Nmt,
    "lowc":Lowercase,
    "strip":Strip,
    "strip_acc":StripAccents,
    "prpnd":Prepend,
    "pre_comp":Precompiled,
    "repl":Replace,
    }

PRETOKENIZERS = {
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