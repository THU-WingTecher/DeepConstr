import yaml
from typing import Dict, Any
SPLITER="\n"
def load_yaml(path) :
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def formatted_dict(dict : Dict[Any, Any], sep=" : ", split= "\n") -> str:
    return split.join([f"{str(k)}{sep}{str(v)}" for k, v in dict.items() if v is not None])