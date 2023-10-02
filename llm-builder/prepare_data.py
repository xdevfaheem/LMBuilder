
# Importing the Necessary Libraries
import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset # huggingface datasets
import sentencepiece as spm
import tiktoken

# an utility func for loading the dataset
def load_ds(name, build_vocab=True):

    dataset = load_dataset(name,num_proc=round(os.cpu_count() * 0.75))
    train_eval_set = dataset["train"].train_test_split(test_size=0.00006, seed=2354, shuffle=True)
    train_eval_set['val'] = train_eval_set.pop('test') # rename the test split to val
    if build_vocab:
        return dataset, train_eval_set
    else:
        return train_eval_set

# an utilty function for writing the dataset into a txt file for training sentencepiece vocabulary
def write_to_txt(file_path, sentences):

    with open(file_path, 'w') as f:
        for sentence in tqdm(sentences, total=len(sentences), desc="Writing dataset to txt"):
            f.write(sentence["text"]+"\n")
    return file_path

# an utility function to train vocabulary for the dataset using sentencepiece trainer
def build_vocab(dataset_folder: str, dataset, vocab_file_prefix='vocab', vocab_size=8000, model_type='bpe'):

        print("Checking whether the vocabulary file already exist...")
        if not os.path.isfile(dataset_folder+vocab_file_prefix+'.model'):
            print("Going to create a vocabulary using sentencepiece for this dataset as it doesn't already exist \nChecking whether the dataset txt file exist to build vocab file")
            if not os.path.isfile(dataset_folder+'openwebtext_20p.txt'):
                print("Writing Dataset to a txt file as it doesn't already exist")
                data_txt_path = write_to_txt(dataset_folder+'openwebtext_20p.txt', dataset)
            else:
                print("Not going to write the dataset to a txt file as it already exist")
                data_txt_path = dataset_folder+'openwebtext_20p.txt'

            os.chdir(dataset_folder)

            print(f"Training the Sentencepiece Vocab with: \nFile Prefix: {vocab_file_prefix} \nVocab Size: {vocab_size} \nModel Type: {model_type} \nPAD ID: {None} \nEOS ID: {int(1)} \nUNK ID: {int(2)} \nBOS ID: {None} \n", flush=True)
            try:
                spm.SentencePieceTrainer.train(input=data_txt_path, model_prefix=vocab_file_prefix, vocab_size=vocab_size, model_type=model_type, pad_id=-1, eos_id=1, unk_id=2, bos_id=-1, eos_piece='<EOS>', unk_piece='<UNK>')
                print("Vocabulary Created Successfully!")
            except Exception as e:
                print("An error occurred during SentencePiece training:", e)

            return dataset_folder + vocab_file_prefix+'.model'
        
        else:
            print("Vocabulary File Already Exist! Won't Train an other.")
            return dataset_folder + vocab_file_prefix +'.model'

# function for tokenizing the dataset using our trained sentencepiece vocabulary
def vocab_process(example, vocab=True, bpe_tt=False):
    ids = sp_vocab.encode(example["text"], out_type=int)
    ids.append(sp_vocab.eos_id())
    return {"ids":ids, "len": len(ids)}

# function for tokenizing the dataset using openai's tiktoken tokenizer
def tt_process(example):
    ids = tt_vocab.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(tt_vocab.eot_token) # add the end of text token
    return {'ids': ids, 'len': len(ids)}

# preapare the memmory maaped binary file dataset
def prepare_memmap(dataset_folder, build_vocab=True, dataset="Bingsu/openwebtext_20p"):
    
    if build_vocab:
        full_dataset, splitted_dataset = load_ds(dataset, build_vocab=build_vocab)
        # vocabulary model path of the trained vocab
        vocab_path = build_vocab(dataset_folder, full_dataset['train'], vocab_file_prefix="openwebtext_20p", vocab_size=16000)
        # loading the vocab for encoding process
        sp_vocab = spm.SentencePieceProcessor(model_file=vocab_path)
        #mapping the dataset to the `vocab_process` function to tokenize them using the trained vocab
        tokenized_dataset = splitted_dataset.map(
                            vocab_process,
                            remove_columns=['text'],
                            desc="Tokenizing the Dataset",
                            num_proc=(os.cpu_count() // 2),)
    else: 
        splitted_dataset = load_ds(dataset, build_vocab=build_vocab)
        # loading openai's tiktoken bpe tokenizer
        tt_vocab = tiktoken.get_encoding("gpt2")
        #mapping the dataset to the `tt_process` function to tokenize them using tiktoken tokenizer
        tokenized_dataset = splitted_dataset.map(
                            tt_process,
                            remove_columns=['text'],
                            desc="Tokenizing the Dataset",
                            num_proc=(os.cpu_count() // 2),
                        )

    # for a quick reference about the memmory mapped file, A memory mapped file is a file that is physically present on disk in a way that the
    # correlation between the file and the memory space permits applications to treat the mapped portions as if it were primary memory, allowing very fast I/O!
    
    # here we are creating mmap array with same shape, if you wish to save img and text which has different shape and axes as a memmory map array which cannot be done by np.memmap,
    # there's a cool library for it. checkout it out -> https://github.com/hristo-vrigazov/mmap.ninja
    
    # combine all the tokenized ids of the train and validation dataset into a single memmory mapped binary file for training
    for split, dset in tokenized_dataset.items():
        arr_len = np.sum(dset['len'], dtype=np.uint32) # unint64 is passed because arr_len for train (1,898,295,682) and val (1,15,272) < 2**32 (4,294,967,296)
        filename = os.path.join(dataset_folder, f'openwebtext_20p_{split}.bin')
        dtype = np.uint16 # unsigned integer 16 dtype chosed here, since vocab.get_piece_size() == 8000 is < 2**16 to improve efficiency and reduce momory wastage
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024 # you can decrease this if you want for faster writing based on you I/O speed
    
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}', unit=" batch"):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush() # flush the mmap array to offload the file to disk
    
    return os.path.join(dataset_folder, 'openwebtext_20p_train.bin'), os.path.join(dataset_folder, 'openwebtext_20p_val.bin')

# train.bin is ~4GB and vali.bin is ~250KB
# training dataset has ~1.9B (1,898,295,682) tokens.
# validation dataset has ~0.12M (1,15,272) tokens.

# it can be loaded later using numpy like this
# train_data = np.memmap(file_path, dtype=np.uint16, mode="r")
# it can be intrepreted as regular numpy array
