from typing import Union, Dict, Any, Optional, Mapping, Tuple
from pathlib import Path
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem
from lmbuilder.utils import is_package_available


def get_filesystem(path: Path, **kwargs: Any) -> AbstractFileSystem:
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs

def _is_object_storage(fs: AbstractFileSystem) -> bool:
    if is_package_available("adlfs"):
        from adlfs import AzureBlobFileSystem

        if isinstance(fs, AzureBlobFileSystem):
            return True

    if is_package_available("gcsfs"):
        from gcsfs import GCSFileSystem

        if isinstance(fs, GCSFileSystem):
            return True

    if is_package_available("s3fs"):
        from s3fs import S3FileSystem

        if isinstance(fs, S3FileSystem):
            return True

    return False


def _is_dir(fs: AbstractFileSystem, path: Union[str, Path], strict: bool = False) -> bool:
    """Check if a path is directory-like.

    This function determines if a given path is considered directory-like, taking into account the behavior
    specific to object storage platforms. For other filesystems, it behaves similarly to the standard `fs.isdir`
    method.

    Args:
        fs: The filesystem to check the path against.
        path: The path or URL to be checked.
        strict: A flag specific to Object Storage platforms. If set to ``False``, any non-existing path is considered
            as a valid directory-like path. In such cases, the directory (and any non-existing parent directories)
            will be created on the fly. Defaults to False.

    """
    # Object storage fsspec's are inconsistent with other file systems because they do not have real directories,
    # see for instance https://gcsfs.readthedocs.io/en/latest/api.html?highlight=makedirs#gcsfs.core.GCSFileSystem.mkdir
    # In particular, `fs.makedirs` is a no-op so we use `strict=False` to consider any path as valid, except if the
    # path already exists but is a file
    if _is_object_storage(fs):
        if strict:
            return fs.isdir(path)

        # Check if the path is not already taken by a file. If not, it is considered a valid directory-like path
        # because the directory (and all non-existing parent directories) will be created on the fly.
        return not fs.isfile(path)

    return fs.isdir(path)
