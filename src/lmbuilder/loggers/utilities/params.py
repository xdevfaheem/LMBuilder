from typing import Union, Dict, Any, Optional, Mapping, Tuple
from torch import Tensor
from argparse import Namespace


def _convert_params(params: Optional[Union[Dict[str, Any], Namespace]]) -> Dict[str, Any]:
    """Ensure parameters are a dict or convert to dict if necessary.

    Args:
        params: Target to be converted to a dictionary

    Returns:
        params as a dictionary

    """
    # in case converting from namespace
    if isinstance(params, Namespace):
        params = vars(params)

    if params is None:
        params = {}

    return params

def _add_prefix(
        metrics: Mapping[str, Union[Tensor, float]],
        prefix: str = None,
        separator: str = "."
    ) -> Mapping[str, Union[Tensor, float]]:

    """Insert prefix before each key in a dict, separated by the separator.

    Args:
        metrics: Dictionary with metric names as keys and measured quantities as values
        prefix: Prefix to insert before each key
        separator: Separates prefix and original key name

    Returns:
        Dictionary with prefix and separator inserted before each key

    """
    if not prefix:
        return metrics
    return {f"{prefix}{separator}{k}": v for k, v in metrics.items()}


def _sanitize_callable_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize callable params dict, e.g. ``{'a': <function_**** at 0x****>} -> {'a': 'function_****'}``.

    Args:
        params: Dictionary containing the hyperparameters

    Returns:
        dictionary with all callables sanitized

    """

    def _sanitize_callable(val: Any) -> Any:
        # Give them one chance to return a value. Don't go rabbit hole of recursive call
        if callable(val):
            try:
                _val = val()
                if callable(_val):
                    return val.__name__
                return _val
            # todo: specify the possible exception
            except Exception:
                return getattr(val, "__name__", None)
        return val

    return {key: _sanitize_callable(val) for key, val in params.items()}