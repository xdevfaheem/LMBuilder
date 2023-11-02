# Necessary packages
import os
import struct
import numpy as np
from utils import find_data_type_code, HEADER_MAGIC


class MemMappedDatasetBuilder(object):
    
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
        self._data_array = np.zeros(self._chunk_size, dtype=self._dtype)
        self._data_array.fill(self._sep_token)
        self._data_index = 0
        self._version = 1
        self._file_paths = []

    def _write_chunk(self):
        """
        Writes a data chunk to a binary file.
        """
        file_name = f"{self._prefix}_{self._counter:010d}.bin"
        file_name = os.path.join(self._outdir, file_name)

        with open(file_name, "wb") as file:
            file.write(HEADER_MAGIC)
            file.write(struct.pack("<Q", self._version))
            file.write(struct.pack("<B", find_data_type_code(self._dtype)))
            file.write(struct.pack("<Q", self._chunk_size))
            file.write(self._data_array.tobytes(order="C"))

        self._file_paths.append(file_name)
        self._counter += 1
        self._data_array.fill(self._sep_token)
        self._data_index = 0

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
        return self._file_paths.copy()

    def add_data_array(self, array: np.array):
        """
        Adds a numpy array to a memory mapped binary file.

        Args:
            array (np.ndarray): Numpy array to add to the dataset.
        """
        while self._data_index + array.shape[0] > self._chunk_size:
            part_length = self._chunk_size - self._data_index
            self._data_array[self._data_index : self._data_index + part_length] = array[:part_length]
            self._write_chunk()
            array = array[part_length:]

        array_length = array.shape[0]
        self._data_array[self._data_index : self._data_index + array_length] = array
        self._data_index += array_length

    def write_reminder(self):
        """
        Writes any remaining data to chunks.
        """
        self._write_chunk()