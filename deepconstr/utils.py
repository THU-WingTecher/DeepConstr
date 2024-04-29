import copy
import yaml
from typing import Dict, Any
SPLITER="\n"
def load_yaml(path) :
    with open(path, 'r') as f:
        return yaml.safe_load(f)

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


def save_record(record: dict, path: str) -> None:
    """
    Save a given record dictionary to a YAML file, ensuring that the dictionary is
    transformed back to its original expected format before saving.

    Args:
        record (dict): The record dictionary to be saved.
        path (str): The file path where the record should be saved.

    Raises:
        FileNotFoundError: If the directory specified in the path does not exist.
        Exception: For any unexpected errors during the save operation.
    """
    # Transform the record back to the expected format
    record = copy.deepcopy(record)
    record = transform_record_for_saving(record)

    # Ensure the directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        # raise FileNotFoundError(f"The directory {directory} does not exist.")
    
    with open(path, 'w') as file:
        yaml.dump(record, file)

    TRAIN_LOG.debug(f"Record saved successfully to {path}.")