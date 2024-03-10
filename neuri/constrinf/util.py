import yaml
from typing import Dict, Any
SPLITER="\n"
def load_yaml(path) :
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def formatted_dict(dict : Dict[Any, Any]) -> str:
    return "\n".join([f"{str(k)} : {str(v)}" for k, v in dict.items() if v is not None])