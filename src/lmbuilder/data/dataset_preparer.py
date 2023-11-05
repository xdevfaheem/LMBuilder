# Importing the Necessary Libraries
import os
import time
import sys
import glob
import types
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from multiprocessing import Process, cpu_count, current_process, Pool, Lock
# from datasets import get_dataset_split_names, load_dataset, load_from_disk, Dataset, DatasetDict
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from lmbuilder.lmlogger import LMBuilderLogger
from tokenizer import Tokenizer
from memap_dataset import MemMappedDatasetBuilder


class LMBuilderDatasetPreparer:
    
    """
    Prepares language model training datasets by processing source files and creating memmory-mapped packed datasets.

    This class streamlines the preparation of training and validation datasets for language model training. It processes
    source text files, tokenizes the text, and constructs memap datasets for efficient data loading.

    Args:
        source_path (str): The directory containing the source dataset files.
        tokenizer_path (str): The path to the tokenizer model used for text tokenization.
        destination_path (str): The directory to save the prepared datasets.
        log_dir (str): The directory for storing log files.
        filenames_set (dict): A dictionary with dataset prefixes and corresponding subdir file patterns.
        exclude_sets (list, optional): A list of sets to exclude from dataset preparation. Defaults to None.
        train_percentage (float, optional): The percentage of data to use for training (0.0 to 1.0). Defaults to 1.0.

    Methods:
        prepare:
            Prepares training and validation datasets based on the provided configurations.

        load_data:
            Loads and yields text data from the specified list of filenames.

        prepare_dataset:
            Prepares a dataset by tokenizing text data and saving it to the specified output path.

    Attributes:
        source_path (str): The path to the source dataset files.
        destination_path (str): The directory for saving prepared datasets.
        tokenizer (Tokenizer): The tokenizer used for text encoding.
        logger (Logger): A configured logger for logging messages.

    Usage Example:
        source_path = "data/your_dataset"
        tokenizer_path = "checkpoints/your_tokenizer.model"
        destination_path = "data/your_destination"
        max_length = 2048
        blocks = 1024
        file_set = {"arxiv_set", "arxiv/*"}
        exclude_sets = ["excluded_set1", "excluded_set2"]

        dataset_preparer = LMBuilderDatasetPreparer(
            source_path, tokenizer_path, destination_path, "logs", file_set, exclude_sets, train_percentage=0.95)
        dataset_preparer.prepare(mode="process", max_length, blocks)
    """
    
    def __init__(self,
                source_path,
                tokenizer_path,
                destination_path,
                log_dir,
                filenames_set: dict,
                exclude_sets: list = None,
                train_percentage: float = 1.0
                ):
        
        
        # source_path is the path to the dataset files
        self.source_path = source_path
        # creating the out directory if it's not already exist 
        self.destination_path = destination_path
        os.makedirs(self.destination_path, exist_ok=True)
        # initialising the tokenizer
        self.tokenizer_path = tokenizer_path
        self.tokenizer = Tokenizer(self.tokenizer_path)
        # configuring logger
        self.logger_inst = LMBuilderLogger()
        self.logger = self.logger_inst.configure_logging(log_dir, "dataset_preparation")
        # cofiguring checkpointing
        self.ckpt_file = os.path.join(log_dir, "data_prep_ckpt.json")
        # dict containing the prefix with dataset subdirectory
        self.filenames_set = filenames_set
        # files to exclude in dataset preparation
        self.exclude_sets = exclude_sets or []
        # train and val %
        self.train_percentage = train_percentage
        
        
    def save_checkpoint(self, pattern, file_idx, split):
        
        # Save the current state to the checkpoint file
        checkpoint_data = {
            split: {
                pattern : {
                    "current_file_idx": file_idx
                    }
                }
            }
        
        with open(self.ckpt_file, "w") as checkpoint_file:
            json.dump(checkpoint_data, checkpoint_file)

    def load_checkpoint(self, split):
        
        # Load the checkpoint if it exists
        if os.path.exists(self.ckpt_file):
            try:
                with open(self.ckpt_file, "r") as checkpoint_file:
                    checkpoint_data = json.load(checkpoint_file)[split]
                    pattern = list(checkpoint_data.keys())[0]
                    curr_file_idx = checkpoint_data[pattern].get("current_file_idx")
            except: 
                # to prevent key error if val split is not processed yet
                pattern, curr_file_idx = "", 0
        
        else:
            pattern, curr_file_idx = "", 0 # using "" insead None to prevent Nonetype error when comparing path_pattern with None
            
        return pattern, curr_file_idx


    def load_data(self, filenames):
        
        for file_path in filenames:

            if file_path.endswith(".json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    yield json.load(f)["text"]
            
            elif file_path.endswith(".jsonl"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    yield (json.loads(line)["text"] for line in lines)
            
            elif file_path.endswith(".parquet"):
                contents = pd.read_parquet(file_path, engine='pyarrow')['content']
                yield (text for text in contents)
                 
            else:
                raise ValueError(f"Unsupported file format detected!: {file_path}")


    def prepare_dataset(self, out_path, pattern, prefix, file_names, chunk_size, process_id, split="train"):
        
        curr_file_idx = 0

        builder = MemMappedDatasetBuilder(
            outdir=out_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=self.tokenizer.bos_id,
            dtype="auto",
            vocab_size=self.tokenizer.vocab_size,
        )
        try:
            for data, i in enumerate(tqdm(iterable=self.load_data(file_names), desc=f"#{process_id}", leave=True, mininterval=0.0, unit=" texts")):
                
                if isinstance(data, types.GeneratorType):
                    for text in data:
                        text_ids = self.tokenizer.encode(text)
                        builder.add_array(np.array(text_ids, dtype=builder.dtype))     
                
                else:
                    text_ids = self.tokenizer.encode(data)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
                
                curr_file_idx += i

            builder.write_reminder()
            
        except KeyboardInterrupt:
            self.logger.error("Data files preparation stopped due to intentional keyboard interruption.")
            self.logger.info(f"Saving Checkpoint to resume later in {self.ckpt_file}")
            self.save_checkpoint(pattern, curr_file_idx, split)
            
        except Exception as e:
            self.logger.error(f"Error processing {file_names}: {str(e)}")
            self.logger.info(f"Saving Checkpoint to resume later in {self.ckpt_file}")
            self.save_checkpoint(pattern, curr_file_idx, split)
            

    def prepare(self, num_workers, prl_pool="thread", max_length: int = 512, num_blocks: int = 1024):
        
        global concurrent_dataset_preparation # prevents pickling error
        def concurrent_dataset_preparation(arg):
            # unload arguments
            dest_path, prefix, file_names, chunk_size, split, process_id, num_process = arg
            print(f"Process {process_id} started!\n", flush=True)
            start_time = time.perf_counter()
            filenames = np.array_split(file_names, num_process)[process_id]
            self.prepare_dataset(dest_path, prefix, filenames, chunk_size, process_id, split)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Process {process_id} finished in {elapsed_time:.2f} seconds", flush=True)
        
        prl_ctx_mngr = ThreadPoolExecutor if prl_pool=="thread" else ProcessPoolExecutor if prl_pool=="process" else None
        assert prl_ctx_mngr is not None, "specify prl_pool to be either 'thread' or 'process'"
            
        # resume where we left
        curr_train_dir, curr_train_idx = self.load_checkpoint(split="train")
        curr_val_dir, curr_val_idx = self.load_checkpoint(split="val")
        from_scratch = curr_train_dir == "" and curr_train_idx == 0
        
        # calculating chunk size, +1 bcoz we want next token also. so that we can just x = [:-1], y = [1:] 
        chunk_size = ((max_length+1)*num_blocks)
        
        if from_scratch:
            filenames_sets = self.filenames_set.items()
        else:
            self.logger.info(f"Resuming data preparation from {curr_train_dir}")
            filenames_sets = {k: v for k, v in self.filenames_set.items() if v >= curr_train_dir}
        
        for dset_name, path_pattern in filenames_sets.items():

            if os.path.isfile(path_pattern):
                self.logger_inst.log_timestamp()
                self.logger.info(f"Processing file {path_pattern}")
                train_filenames = [path_pattern]
                val_filenames = None
                train_start_idx = 0

            else:
                train_start_idx = curr_train_idx if curr_train_dir == path_pattern else 0
                val_start_idx = curr_val_idx if curr_val_dir == path_pattern else 0

                self.logger_inst.log_timestamp()
                self.logger.info(f"Processing {path_pattern}")
                train_filenames = glob.glob(os.path.join(self.source_path, path_pattern), recursive=True)
                val_filenames = None
                if self.exclude_sets:
                    train_filenames = [filename for filename in train_filenames if not any(exclude in filename for exclude in self.exclude_sets)]

                if self.train_percentage < 1.0:
                    if train_filenames<3:
                        raise ValueError("A subdirectory or a path pattern should atleast have 3 files inside it!")
                    num_files = int(len(train_filenames) * self.train_percentage)
                    train_filenames = train_filenames[:num_files]
                    val_filenames = train_filenames[num_files:] # if the len(train_filenames) == num_files then val_filenames = []
                    
            with prl_ctx_mngr(max_workers=(num_workers), initializer=tqdm.set_lock, initargs=(Lock(),)) as executor:
                list(executor.map(
                    concurrent_dataset_preparation,
                    [
                        (os.path.join(self.destination_path, "train"), dset_name, train_filenames[train_start_idx:len(train_filenames)], chunk_size, "train", process_id, num_workers) 
                        for process_id in range(num_workers)
                    ],
                    chunksize=1
                ))
                if val_filenames is not None or []:
                    list(executor.map(
                        concurrent_dataset_preparation,
                        [
                            (os.path.join(self.destination_path, "val"), dset_name, val_filenames[val_start_idx:len(val_filenames)], chunk_size, "val", process_id, num_workers) 
                            for process_id in range(num_workers)
                        ],
                        chunksize=1
                    ))


if __name__ == "__main__":
    
    # Example usage:
    source_path = "data/your_dataset"
    tokenizer_path = "checkpoints/your_tokenizer.model"
    destination_path = "data/your_destination"
    max_length = 2048
    blocks = 1024
    file_set = {"arxiv_set", "arxiv/*"}
    exclude_sets = ["excluded_set1", "excluded_set2"]
    
    dataset_preparer = LMBuilderDatasetPreparer(source_path, tokenizer_path, destination_path, file_set, exclude_sets, percentage=0.95)
    dataset_preparer.prepare(os.cpu_count(), prl_pool="process", max_length=max_length, num_blocks=blocks)
    
    lm = LMBuilderDatasetPreparer()