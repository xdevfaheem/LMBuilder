import numpy as np

# Header constants
HEADER_MAGIC = b"LITPKDS"
HEADER_SIZE = 24  # bytes

# Dictionary to map data types to codes
data_type_codes = {
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
def find_data_type_code(data_type):
    """
    Finds the code for a given data type.
    Args:
        data_type: Data type to find the code for.
    Returns:
        int: Code corresponding to the data type.
    """
    for code, dtype in data_type_codes.items():
        if dtype == data_type:
            return code
    raise ValueError(data_type)