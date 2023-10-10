# Importing the Necessary Libraries
import os
import time
import sys
from tqdm.auto import tqdm
import numpy as np
from multiprocessing import Process, cpu_count, current_process, Pool, Lock
import datasets # huggingface datasets
from datasets import get_dataset_split_names, load_dataset, load_from_disk, Dataset, DatasetDict
import sentencepiece as spm
import tiktoken
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class PrepareDataset:
                    
    def __init__(self,
                 dataset_folder,
                 hf_dataset=None,
                 dataset_files=None,
                 from_disk: bool = False,
                 local_dataset_path=None,
                 train_dset_list=None,
                 val_dset_list=None,
                 train_data_percentage=0.8,
                 dset_prefix="tinystories",
                 build_vocab=False,
                 vocab_size=8000,
                 vocab_type="yt",
                 eos=1,
                 bos=0,
                 pad=2,
                 unk=3,
                 **kwargs
                ):
        
        self.dataset_folder = dataset_folder
        self.dataset_files_folder = os.path.join(self.dataset_folder, "dataset_files")
        os.makedirs(self.dataset_files_folder, exist_ok=True)
        self.data_txt_path = os.path.join(dataset_folder, f'{dset_prefix}.txt')
        self.dset_prefix = dset_prefix

        if from_disk:
            if not os.path.exists(local_dataset_path):
                assert (train_dset_list), "atleast train dataset `text` list should be passed" # val_dset_list is optional as validation set will be splitted from train set 
                self.create_custom_dataset(train_dset_list, local_dataset_path, val_list=val_dset_list)
        
        if build_vocab:
            self.splitted_dataset, self.dataset_splits = self.load_dsets(hf_dset_name=hf_dataset, data_files=dataset_files, from_disk=from_disk, dataset_path=local_dataset_path, train_p=train_data_percentage)
            print("\nDataset: ",self.splitted_dataset, flush=True)
            # vocabulary model path of the trained vocab
            vocab_path = self.build_vocabulary(self.splitted_dataset, vocab_prefix=self.dset_prefix, vocab_size=vocab_size, vocab_model_type=vocab_type, eos_id=eos, bos_id=bos, pad_id=pad, unk_id=unk)
            # choosing the tokenizer backend
            sp = True if vocab_type=="sp" else False
            yt = True if vocab_type=="yt" else False
            # loading the tokenizer for encoding process
            self.tokenizer = Tokenizer(model_path=vocab_path, youtokenizer=yt, sp_tokenizer=sp)
            if self.tokenizer.backend == "youtoken":
                setattr(self.tokenizer, "eos_id", eos)
                setattr(self.tokenizer, "bos_id", bos)
                
        else:
            self.splitted_dataset, self.dataset_splits = self.load_dsets(hf_dset_name=hf_dataset, data_files=dataset_files, from_disk=from_disk, dataset_path=local_dataset_path, train_p=train_data_percentage)
            # loading openai's tiktoken bpe tokenizer
            self.tokenizer = Tokenizer(model_path=None, tiktokenizer=True)
        
    # an utility function for loading the hugging face dataset
    def load_dsets(self, hf_dset_name: str = None, from_disk=False, data_files: dict=None, dataset_path=None, train_p: float = 0.9):

        assert ((from_disk and dataset_path) or hf_dset_name is not None), "the Dataset Should be Either loaded from local path or from hugging face dataset repo"
        
        if from_disk:
            print("\nLoading dataset from the disk...")
            assert dataset_path is not None, "Pass the path for local dataset which was saved to disk earlier."
            dataset_dict = load_from_disk(dataset_path)
            
        else:
            print("\nLoading dataset from huggingface...")
            p = train_p*100
            dataset_dict = {}
            splits = get_dataset_split_names(hf_dset_name)
            for split in splits:
                if split=="train":
                    dataset = datasets.load_dataset(hf_dset_name, data_files=data_files, split=datasets.ReadInstruction(split, from_=0, to=p, unit='%'), num_proc=int(os.cpu_count())//2)
                else:
                    dataset = datasets.load_dataset(hf_dset_name, data_files=data_files, split=split, num_proc=int(os.cpu_count())//2)
                dataset_dict[split] = Dataset.from_dict({'text': dataset["text"]})

        if len(dataset_dict) == 1:
            dataset_dict.update(dataset_dict[list(dataset_dict.keys())[0]].train_test_split(test_size=0.0001, seed=2354, shuffle=True))
        
        assert len(dataset_dict) == 2, "Dataset Should have only 2 splits"
        
        split_keys = list(dataset_dict.keys())
        # key much be 'train' and 'val' to match up in data loading in GPTBuilder
        dataset_dict.update({"train":dataset_dict.pop(split_keys[0])})
        dataset_dict.update({"val":dataset_dict.pop(split_keys[1])})
        
        assert dataset_dict.keys()[0] == "train", "The key of the first split much be 'train'"
        assert dataset_dict.keys()[1] == "val", "The key of the second split much be 'val'"
        
        final_splits = list(dataset_dict.keys())
        if from_disk:
            return dataset_dict, final_splits
        
        else:
            return DatasetDict(dataset_dict), final_splits

    def create_custom_dataset(self, train_list, save_path, val_list=None):
        
        if val_list is not None:
            dataset = DatasetDict({
                'train': Dataset.from_dict({'text': train_list }),
                'validation': Dataset.from_dict({'text': val_list}),
            })
        else:
            dataset = DatasetDict({
                'train': Dataset.from_dict({'text': train_list }),
            })
        dataset.save_to_disk(save_path)

    # an utilty function for writing the dataset into a txt file for training sentencepiece vocabulary
    def write_to_txt(self, file_path: str, dataset: dict):
        with open(file_path, 'w') as f:
            for split in self.dataset_splits:
                for sentence in tqdm(dataset[split]["text"], total=len(dataset[split]["text"]), desc=f"Writing {split} dataset to txt file"):
                    f.write(sentence+"\n")
        return file_path

    # an utility function to train vocabulary for the dataset using sentencepiece trainer
    def build_vocabulary(self, dataset: datasets.DatasetDict, vocab_prefix: str = 'vocab', vocab_size: int = 8000, vocab_model_type="yt", pad_id: int = 3, unk_id: int = 2, eos_id: int = 1, bos_id: int = 0):
            
            sp = True if vocab_model_type=="sp" else False
            yt = True if vocab_model_type=="yt" else False
            
            print("\nChecking whether the vocabulary file already exist...")
            if not os.path.isfile(os.path.join(self.dataset_folder, f"{vocab_prefix}_{vocab_size}.model")):
                
                print(f"Going to create a vocabulary using `{'SentencePiece' if sp else 'Youtokentome'}` for this dataset as it doesn't already exist \nChecking whether the dataset txt file exist to build vocab file")
                if not os.path.isfile(self.data_txt_path):
                    print("Writing Dataset to a txt file as it doesn't already exist")
                    data_txt_path = self.write_to_txt(self.data_txt_path, dataset)
                    
                else:
                    print("Not going to write the dataset to a txt file as it already exist")
                    data_txt_path = self.data_txt_path
                    
                try:
                    print(f"Training the `{'SentencePiece' if sp else 'Youtokentome'}` Vocab with: \nFile Prefix: {vocab_prefix} \nVocab Size: {vocab_size} \nPAD ID: {pad_id} \nEOS ID: {eos_id} \nUNK ID: {unk_id} \nBOS ID: {bos_id} \n", flush=True)
                    Tokenizer.train(data_txt_path, self.dataset_folder, youtokenizer=yt, sp_tokenizer=sp, vocab_file_prefix=vocab_prefix, vocab_size=vocab_size, pad_id=pad_id, unk_id=unk_id, eos_id=eos_id, bos_id=bos_id)
                    print("Vocabulary Created Successfully!")
                
                except Exception as e:
                    print("An error occurred during vocabulary training:", e)

                return os.path.join(self.dataset_folder, f"{vocab_prefix}_{vocab_size}.model")

            else:
                print("Vocabulary File Already Exist! Won't Train an other.\n")
                return os.path.join(self.dataset_folder, f"{vocab_prefix}_{vocab_size}.model")
            
    # a function for preparing the dateset by preparing a shard of the dataset and tracking time
    def prepare_dataset(self, split: str, dataset: datasets.Dataset, max_length: int, num_blocks: int, i: int, mode="process"):
        
        current = current_process()
        if mode=="process":
            pos = current._identity[0]-1
        elif mode=="thread":
            pos=None
            
        builder = PackedDatasetBuilder(
                        outdir=os.path.join(self.dataset_files_folder, split),
                        prefix=self.dset_prefix,
                        chunk_size=((max_length+1)*num_blocks),
                        sep_token=self.tokenizer.eos_id,
                        dtype="auto",
                        vocab_size=self.tokenizer.vocab_size,
                    )
        with tqdm(total=len(dataset), mininterval=0, unit=" examples", desc=f"Preparing dataset buffers #{i}", position=pos, leave=True) as pbar:
            for example in dataset["text"]:
                ids = self.tokenizer.encode(example, bos=False, eos=True, max_length=max_length)
                # where_eos = [x for x in arr if x == self.tokenizer.eos_id]
                # we expect two EOS tokens, one per file
                #assert len(where_eos) == 1
                #assert ids[-1] == self.tokenizer.eos_id, "Last token of the tokenized id is not equal to tokenizer's eos id"
                builder.add_array(np.array(ids, dtype=builder.dtype))
                pbar.update(1)
            
        builder.write_reminder()
        dataset = None # to free up memory
        
        """
        we have used num_block=1024, max_len=1024, so chunk_size=(1024+1)*1024=1,049,600. as itemsize of data type is 2. so each id occupy 2 bytes.
        so 1,049,600 ids occupy 2,099,200 bytes that ~2.1MB binary file. we got around 63 files so total bytes of data is 132,249,600. so if 1 id occupy two bytes then total tokens in the dataset much be ~66,124,800.
        
        Calculate the total tokens using this:
        from tqdm.auto import tqdm
        tokenizer = Tokenizer(model_path="./tinystories_8000.model", youtokenizer=False, sp_tokenizer=True)
        print(train_dataset)
        total_tokens=0
        for datum in tqdm(train_dataset["text"], total=len(train_dataset["text"])):
            total_tokens += len(tokenizer.encode(datum, bos=False, eos=True, max_length=1024))
        print(f"Total tokens in the train set: {total_tokens}")
        """
        
    # prepare the memmory maped binary file dataset
    def prepare(self, max_length: int = 512, num_blocks: int = 1024):
        
        data_dirs = []
        
        writer_lock = Lock()
        global prepare_shard #prevent pickling error
        
        def prepare_shard(args):
            split, n_process, max_len, num_block, process_id, mode = args
            print(f"Process {process_id} started!\n", flush=True)
            start_time = time.time()
            shard_dset = self.splitted_dataset[split].shard(n_process, index=process_id) #.to_iterable_dataset()
            self.prepare_dataset(split, shard_dset, max_len, num_block, process_id, mode=mode)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Process {process_id} finished in {elapsed_time:.2f} seconds", flush=True)
        
        for split in self.dataset_splits:
            data_dirs.append(str(os.path.join(self.dataset_files_folder, split)))
            num_process = (cpu_count()*2) if split == "train" else (cpu_count())
            # Create a Pool and map the function to the range of processes \ , initializer=tqdm.set_lock, initargs=(writer_lock,)
            with ThreadPoolExecutor(max_workers=num_process, initializer=tqdm.set_lock, initargs=(writer_lock,)) as executor: # , initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)
                if isinstance(executor, ProcessPoolExecutor):
                    mode="process"
                elif isinstance(executor, ThreadPoolExecutor):
                    mode="thread"
                print(f"\nPreparing `{split.capitalize()}` Set with \nNumber of process: {num_process} \nMaximum Length: {max_length} \nNumber of Blocks: {num_blocks} \nMode: {mode}\n", flush=True)
                list(executor.map(prepare_shard, [(split, num_process, max_length, num_blocks, process_id, mode) for process_id in range(num_process)], chunksize=1))
        
        return data_dirs
    
#if __name__ == '__main__':
    # dset_files = {'train':'TinyStoriesV2-GPT4-train.txt', 'validation':'TinyStoriesV2-GPT4-valid.txt'}
    # dataset_preparer = PrepareDataset("./tinystories", hf_dataset="roneneldan/TinyStories", from_disk=True, local_dataset_path="./tinystories/hf_dataset", dataset_files=dset_files, train_data_percentage=0.7, build_vocab=True, vocab_type="sp")
    # dataset_preparer.prepare(max_length=1024)
