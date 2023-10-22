# Neccessary packages
import os
import struct
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

# Dictionary to map data types to codes
dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}

# Function to find the code for a given data type
def code(dtype):
    
    """Finds the code for a given data type.
    Args:
        dtype: Data type to find the code for.
    Returns:
        int: Code corresponding to the data type.
    """
    
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

# Header constants
HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes

# IterableDataset for distributed sampling as iterable-style dataset don't support sampler for DDP.
class PackedDataset(IterableDataset):
    
    def __init__(self,
                 filenames,
                 n_chunks,
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
            filenames (list): List of filenames.
            n_chunks (int): Number of data chunks to load.
            block_size (int): Size of each data block.
            seed (int): Random seed for shuffling.
            shuffle (bool): Whether to shuffle the data chunks.
            wrap (bool): Whether to wrap around the dataset when reaching the end.
            num_processes (int): Number of processes for distributed sampling.
            process_rank (int): Rank of the current process for distributed sampling.
        """
        
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __iter__(self):

        """
        Returns:
            Initialize an PackedDatasetIterator (Iterable type Dataset) with set of distinctive memorymap file based on the num_processes and process_rank
        """
        
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id : max_num_files : num_shards]

        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )


class PackedDatasetBuilder(object):
    
    """
    for a quick reference about the memmory mapped file, A memory mapped file is a file that is physically present on disk in a way that the
    correlation between the file and the memory space permits applications to treat the mapped portions as if it were primary memory, allowing very fast I/O!
    
    here we are creating mmap array with same shape, if you wish to save img and text which has different shape and axes as a memmory map array which cannot be done by np.memmap,
    there's a cool library for it. checkout out -> https://github.com/hristo-vrigazov/mmap.ninja
    """
    def __init__(
        self,
        outdir,
        prefix,
        chunk_size,
        sep_token,
        dtype="auto",
        vocab_size=None,
    ):

        """
        Initializes a class to build PackedDataset.

        Args:
            outdir (str): Output directory for storing dataset files.
            prefix (str): Prefix for dataset file names.
            chunk_size (int): Size of each data chunk.
            sep_token (int): Separator token to fill empty space.
            dtype (str): Data type of the dataset (default is "auto").
            vocab_size (int): Vocabulary size (required when dtype is "auto").
        """
        
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        os.makedirs(outdir, exist_ok=True)
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []

    def _write_chunk(self):
        """
        Writes a data chunk to a binary file.
        """
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        """
        Returns the data type of the dataset.
        """
        return self._dtype

    @property
    def filenames(self):
        """
        Returns a list of dataset file names.
        """
        return self._filenames.copy()

    def add_array(self, arr):
        """
        Adds a numpy array to the PackedDataset.

        Args:
            arr (np.ndarray): Numpy array to add to the dataset.
        """
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self):
        """
        Writes any remaining data to chunks.
        """
        self._write_chunk()


class PackedDatasetIterator:
    
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap):
        
        """
        Initializes an iterator for PackedDataset.

        Args:
            filenames (list): List of filenames.
            n_chunks (int): Number of data chunks to load.
            block_size (int): Size of each data block.
            seed (int): Random seed for shuffling.
            shuffle (bool): Whether to shuffle the data chunks.
            wrap (bool): Whether to wrap around the dataset when reaching the end.
        """
        
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0

        self._load_n_chunks()

    def _read_header(self, path):

        """
        Reads header information from a binary file.
        Args:
            path (str): File path to the binary file.
        Returns:
            tuple: Tuple containing data type and chunk size.
        """
        
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert (1,) == version
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        """
        Closes memory-mapped files.
        """
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self):
        """
        Loads a new set of data chunks from files.
        """
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx:]):
            #if not self._wrap:
                #raise StopIteration
            self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(
                    filename
                )
                self._n_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = (
            self._rng.permutation(n_all_blocks)
            if self._shuffle
            else range(n_all_blocks)
        )

        self._curr_idx = 0

    def __del__(self):
        """
        Destructor to close resources.
        """
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        """
        Makes the PackedDatasetIterator's attributes accessible 
        """
        return self

    def __next__(self):
        """
        Returns the next data block as a PyTorch tensor (torch.Tensor).
        """
        # Check if all blocks in the current chunk have been used
        if self._curr_idx >= len(self._block_idxs):
             # Load the next set of chunks from the dataset files
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        # Get the index of the next block to retrieve
        block_idx = self._block_idxs[self._curr_idx]
        # Calculate the chunk ID corresponding to the block index
        chunk_id = block_idx // self._n_blocks
        # Access the memory view (buffer) for the appropriate chunk
        buffer = self._buffers[chunk_id]
        # Calculate the element ID within the chunk where the data block starts
        elem_id = (block_idx % self._n_blocks) * self._block_size #the starting element index within the chunk.
        # Calculate the byte offset within the memory view for the data block
        # offset is the the number of bytes from the beginning of the memory region to the starting point of the data you want to access.
        offset = np.dtype(self._dtype).itemsize * elem_id # Each id occupy (np.dtype(self._dtype).itemsize) amount of bytes. 
        # Read the data block from the memory view
        arr = np.frombuffer(
            buffer, dtype=self._dtype, count=self._block_size, offset=offset
        )
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64))
    

class CombinedDataset(IterableDataset):
    
    def __init__(self, datasets, seed, weights=None):
        """
        Initializes an IterableDataset for combining multiple datasets.
        
        Args:
            datasets (list): List of datasets build by PackedDatasetBuilder.
            seed (int): Random seed for dataset selection.
            weights (list): List of weights for dataset selection (default is None).
        """
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        """
        Initialize the CombinedDatasetIterator class with gotten args.
        """
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    
    def __init__(self, datasets, seed, weights):
        """
        Initializes an iterator for CombinedDataset.

        Args:
            datasets (list): List of datasets to combine.
            seed (int): Random seed for dataset selection.
            weights (list): List of weights for dataset selection.
        """
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        """
        Returns the next element from one of the datasets as a PyTorch tensor.
        """
        dataset, = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)
