# Neccessary packages
import os
import struct
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from utils import find_data_type_code, HEADER_MAGIC, HEADER_SIZE


# IterableDataset for distributed sampling as iterable-style dataset don't support sampler for DDP.
class MemMappedDataset(IterableDataset):
    
    def __init__(self,
                 file_paths,
                 num_chunks,
                 block_size,
                 seed=12345,
                 shuffle=True,
                 wrap=False,
                 num_processes=1,
                 process_rank=0
                ):

        """
        Initializes an IterableDataset for distributed sampling.

        Args:
            file_paths (list): List of file paths.
            num_chunks (int): Number of data chunks to load.
            block_size (int): Size of each data block.
            seed (int): Random seed for shuffling.
            shuffle (bool): Whether to shuffle the data chunks.
            wrap (bool): Whether to wrap around the dataset when reaching the end.
            num_processes (int): Number of processes for distributed sampling.
            process_rank (int): Rank of the current process for distributed sampling.
        """
        
        self._file_paths = file_paths
        self._num_chunks = num_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __iter__(self):

        """
        Initialize a MemMappedDataIterator (Iterable-type Dataset) with a set of distinctive memory-mapped files based on the no. of processes and rank of the process.
        """
        
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._file_paths) // num_shards * num_shards
        file_paths = self._file_paths[shard_id : max_num_files : num_shards]

        return MemMappedDataIterator(
            file_paths=file_paths,
            num_chunks=self._num_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )
        
        
class MemMappedDataIterator:
    
    def __init__(self, file_paths, num_chunks, block_size, seed, shuffle, wrap):
        
        """
        Initializes an iterator for MemMappedData.

        Args:
            file_paths (list): List of file paths.
            num_chunks (int): Number of data chunks to load.
            block_size (int): Size of each data block.
            seed (int): Random seed for shuffling.
            shuffle (bool): Whether to shuffle the data chunks.
            wrap (bool): Whether to wrap around the dataset when reaching the end.
        """
        
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_indices = None

        self._wrap = wrap
        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.

        self._file_paths = file_paths
        self._file_index = 0

        self._num_chunks = num_chunks

        self._data_type = None
        self._block_size = block_size
        self._num_blocks = None

        self._mmaps = []
        self._buffers = []

        self._block_indices = []
        self._current_index = 0

        self._load_num_chunks()

    def _read_header(self, file_path):

        """
        Reads header information from a binary file.
        Args:
            file_path (str): File path to the binary file.
        Returns:
            tuple: Tuple containing data type and chunk size.
        """
        
        with open(file_path, "rb") as file:
            magic = file.read(len(HEADER_MAGIC))
            assert magic == HEADER_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", file.read(8))
            assert (1,) == version
            (data_type_code,) = struct.unpack("<B", file.read(1))
            data_type = find_data_type_code(data_type_code)
            (chunk_size,) = struct.unpack("<Q", file.read(8))
        return data_type, chunk_size

    def _close_mmaps(self):
        """
        Closes memory-mapped files.
        """
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_num_chunks(self):
        """
        Loads a new set of data chunks from files.
        """
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._num_chunks > len(self._file_paths[self._file_index:]):
            #if not self._wrap:
                #raise StopIteration
            self._file_index = 0

        for i in range(self._num_chunks):
            file_path = self._file_paths[self._file_index + i]
            if self._data_type is None:
                self._data_type, self._chunk_size = self._read_header(file_path)
                self._num_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            mmap = np.memmap(file_path, mode="r", order="C", offset=HEADER_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_index += self._num_chunks
        all_blocks = self._num_chunks * self._num_blocks

        self._block_indices = (
            self._rng.permutation(all_blocks)
            if self._shuffle
            else range(all_blocks)
        )

        self._current_index = 0

    def __del__(self):
        """
        Destructor to close resources.
        """
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        """
        Makes the this class's attributes accessible 
        """
        return self

    def __next__(self):
        """
        Returns the next data block as a PyTorch tensor (torch.Tensor).
        """
        # Check if all blocks in the current chunk have been used
        if self._current_index >= len(self._block_indices):
            # Load the next set of chunks from the dataset files
            self._load_num_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        # Get the index of the next block to retrieve
        block_index = self._block_indices[self._current_index]
        # Calculate the chunk ID corresponding to the block index
        chunk_id = block_index // self._num_blocks
        # Access the memory view (buffer) for the appropriate chunk
        buffer = self._buffers[chunk_id]
        # Calculate the element ID within the chunk where the data block starts
        element_id = (block_index % self._num_blocks) * self._block_size
        # Calculate the byte offset within the memory view for the data block
        # offset is the the number of bytes from the beginning of the memory region to the starting point of the data you want to access.
        offset = np.dtype(self._data_type).itemsize * element_id
        # Read the data block from the memory view
        array = np.frombuffer(
            buffer,
            dtype=self._data_type,
            count=self._block_size,
            offset=offset
        )
        self._current_index += 1
        return torch.from_numpy(array.astype(np.int64))