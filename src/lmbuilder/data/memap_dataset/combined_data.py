# Neccessary packages
import random
from torch.utils.data import IterableDataset


class CombinedMemMappedDataset(IterableDataset):
    
    def __init__(self, datasets, seed, weights=None):
        
        """
        A class which inherits from IterableDataset, which accepts the neccessary arguments and also intialize data chunk iterator when wrapped with iter function
        
        Args:
            datasets (list): List of datasets built by MemMappedDataBuilder.
            seed (int): Random seed for dataset selection.
            weights (list): List of weights for dataset selection (default is None).
        """
        
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        num_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / num_datasets] * num_datasets

    def __iter__(self):
        """
        initializes the CombinedMemMappedDataIterator class for controlled iteration over multiple dataset.
        """
        return CombinedMemMappedDataIterator(self._datasets, self._seed, self._weights)

# Iterator which 
class CombinedMemMappedDataIterator:
    
    def __init__(self, datasets, seed, weights):
        """
        A class which combine multiple datasets and return the next data chunk from the specific dataset based on it's weight

        Args:
            datasets (list): List of datasets to combine.
            seed (int): Random seed for dataset selection.
            weights (list): List of weights for dataset selection.
        """
        self._datasets = [iter(dataset) for dataset in datasets]
        self._weights = weights
        self._random_generator = random.Random(seed)

    def __next__(self):
        """
        Returns the next element from one of the datasets as a PyTorch tensor.
        """
        selected_dataset = self._random_generator.choices(self._datasets, weights=self._weights, k=1)
        return next(selected_dataset)