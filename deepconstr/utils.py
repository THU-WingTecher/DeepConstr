import copy
import yaml
from typing import Dict, Any
SPLITER="\n"

def formatted_dict(dict : Dict[Any, Any], sep=" : ", split= "\n") -> str:
    return split.join([f"{str(k)}{sep}{str(v)}" for k, v in dict.items() if v is not None])

def flatten_list(nested_list):
    """
    Flattens a nested list structure into a single list.

    Args:
    nested_list (list): A nested list that may contain other lists.

    Returns:
    list: A single, flattened list containing all elements.
    """
    flat_list = []

    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                flatten(item)
            else:
                flat_list.append(item)

    flatten(nested_list)
    return flat_list