import copy
import itertools
import os
import re
from deepconstr.grammar.dtype import DType, materalize_dtypes
import yaml
from logger import TRAIN_LOG
from deepconstr.train.inferencer import Inferencer
from typing import *
from deepconstr.gen.record import save_record, transform_record_for_saving
from deepconstr.train.prompter import torch_type_hint_infer_prompts, numpy_type_hint_infer_prompts, tf_type_hint_infer_prompts
from deepconstr.grammar import MAX_ARR_LEN

DEFAULT_RULES = [
    {'cot' : 'default',
    'length': lambda arg_names : len(arg_names),
    'txt' : lambda arg_names : ' and '.join([f'all(i >= 0 for i in {name}.shape)' for name in arg_names]),
    'msg' : 'negative dimensions are not allowed'},
    {'cot' : 'default',
    'length': lambda arg_names : len(arg_names),
    'txt' : lambda arg_names : ' and '.join([f'{name}.rank <= {MAX_ARR_LEN}' for name in arg_names]),
    'msg' : 'Too large tensor shape'},
]

def get_tensor_args(record) :
    from deepconstr.grammar.dtype import AbsVector
    return [name for name, dtype in zip(record['args']['name'], record['args']['dtype_obj']) if isinstance(dtype[0], AbsVector)]

def gen_default_rule(arg_names, record, rule) :

    return {
        'cot' : rule['cot'],
        'length' : rule['length'](arg_names),
        'txt' : rule['txt'](arg_names),
        'target' : {
            'choosen_dtype' : {name : dtype for name, dtype in zip(record['args']['name'], record['args']['dtype']) if dtype is not None},
            'msg' : rule['msg'],
            'package' : record['package']
        }
    }

def add_default_rules(record) : 
    if "rules" not in record.keys() :
        record['rules'] = []
    for rule in DEFAULT_RULES :
        for ori_rule in record['rules'] :
            if ori_rule[0]['target']['msg'] == rule['msg'] :
                continue
        arg_names = get_tensor_args(record)
        constr = gen_default_rule(arg_names, record, rule)
        scores = {"f1_score" : -1, "overall_score" : -1, "precision" : -1, "recall" : -1}

        record['rules'].append([constr, scores]) 
    
    return record

def gen_dtype_info(save_dir, func_name, package, inferencer) :

    res = []
    if package == "torch" :
        res = torch_load_from_doc(save_dir, api_name=func_name)
    if not res :
        type_gen = TypeGenerator(save_dir, func_name, package, inferencer)
        if type_gen.generated :
            res.append(type_gen.save_path())
    
    return res

def torch_load_from_doc(save_dir, api_name = None):
    import torch.jit.supported_ops
    results = []
    legal = 0
    illegal = 0
    ops_doc_lines = torch.jit.supported_ops.__doc__.splitlines()
    """
    1. tensor_methods: `Tensor.method_name(...`
    2. torch_funcs: `torch.func_name(...`
    """
    skip_startwith_kws = []
    skip_in_kws = [
        ".detach",
        ".save",
        ".item",
        ".dim",
        ".to",
        ".set",
        ".clone",
        ".device",
        ".cpu",
        ".cuda",
        ".tensor",
    ]
    tensor_methods = {}
    torch_funcs = {}
    tot_doc_lines = len(ops_doc_lines)
    i_line = 0
    n_ = 0
    while i_line < tot_doc_lines:
        line = ops_doc_lines[i_line].strip()
        if not line:
            i_line += 1
            continue

        is_tensor_method = line.startswith("Tensor.")
        is_torch_func = line.startswith("torch.")
        # i_line += 1
        # read wanted lines
        if any(
            [line.startswith(skip_kw) for skip_kw in skip_startwith_kws]
        ) or any([skip_kw in line for skip_kw in skip_in_kws]):
            i_line += 1
            continue
        if is_tensor_method or is_torch_func:
            n_+=1
            # get specification of a whole op/func
            op_spec = []
            while i_line < tot_doc_lines:
                line = ops_doc_lines[i_line].strip()
                op_spec.append(line)
                if "->" in line:
                    break
                i_line += 1
            op_spec = " ".join(op_spec)
            # parse op spec
            op_name = op_spec.split("(")[0]
            if api_name is not None and op_name != api_name : 
                i_line += 1
                continue
            if op_name in tensor_methods or op_name in torch_funcs:
                cnt+=1 
            else :
                cnt = 0
            if is_tensor_method:
                op_name = f"torch.{op_name}"
            obj = eval(op_name)
            if callable(obj):
                op_args = op_spec.split("(")[1].split(")")[0]
                op_rets = op_spec.split("-> ")[1]
                if not (
                    ("Tensor" in op_args or is_tensor_method)
                    # and "Tensor" in op_rets
                    # and "out : Tensor" not in op_args
                ):
                    # both inputs and outputs have tensors
                    i_line += 1
                    continue
                if is_torch_func:
                    torch_funcs[op_name] = obj
                else:
                    tensor_methods[op_name] = obj
                save_path = os.path.join(save_dir, f"{op_name.replace('.', '/')}-{cnt}.yaml")
                results.append(save_path)
                if os.path.exists(save_path):
                    i_line += 1
                    continue
                # TRAIN_LOG.info(f"deal with {save_path}")
                record = gen_record_for_operator(op_name, op_args, is_tensor_method)
                save_record(transform_record_for_saving(record), save_path)
            else:
                TRAIN_LOG.info(f"  (Ignored: {obj = } from {op_name = } is not callable")
        i_line += 1
    return results


def gen_record_for_operator(op_name, args_str, is_tensor_method, package="torch") :
    """
    Used for the generate records by cli/prepare.py
    {
        'name': 'torch.add',
        'args': {
            'name': ['arg_0', 'arg_1', 'alpha'],
            'dtype' : [number, List[int], Optional[str], ...]
            'is_pos': [True, True, False],
            'required': [True, True, False],
            'value' : [None, None, None],
        },
        'package': 'torch',
        'pass_rate': 0, 
        'rules' : {}

    }
    """
    special_kws = [ # kwarg name -> we don't support this object
        "dtype",
        "memory_format",
        "layout"
    ]
    is_pos_kwargs = [ "self", "tensors"]
    # args_str --> name : dtype=1, Optional
    records = {
        'name': op_name,
        "args": {
            "name": [],
            "dtype" : [],
            "is_pos": [],
            "required": [],
            "value": [],
        },
        "package": package,
        "pass_rate": 0,
        "rules" : {}
    }
    names = []
    dtypes = []
    is_pos = []
    required_list = []
    if is_tensor_method :
        names.append("self")
        dtypes.append("Tensor")
        is_pos.append(True)
        required_list.append(True)
    for arg_str in custom_split(args_str) :
        if arg_str :
            arg_name = arg_str.split(':')[0].strip()
            arg_dtype = arg_str.split(':')[1].split('=')[0].strip()
            if arg_name in special_kws : # dtype arg name related data type is all wrong
                arg_dtype = "None"
            if 'Optional' in arg_str or "=" in arg_str :
                required = False
            else :
                required = True
            names.append(arg_name)
            dtypes.append(arg_dtype)
            is_pos.append(True if arg_name in is_pos_kwargs else False)
            required_list.append(required)
    records['args']['name'] = names
    records['args']['dtype'] = dtypes
    records['args']['is_pos'] = is_pos
    records['args']['required'] = required_list
    records['args']['value'] = [None] * len(names)
    return records


def custom_split(input_string):
    # Stores the parts of the string split by commas outside brackets
    parts = []
    current_part = []  # Stores characters for the current part being processed
    bracket_stack = []  # Keeps track of open brackets to ensure matching pairs
    
    # Mapping of closing and opening brackets
    brackets = {']': '[', '}': '{', ')': '('}
    
    for char in input_string:
        if char in "[{(":
            bracket_stack.append(char)
        elif char in "]})" and bracket_stack:
            if bracket_stack[-1] == brackets[char]:
                bracket_stack.pop()
            else:
                # Handle mismatched bracket scenario
                raise ValueError("Mismatched brackets in input string")
        
        if char == ',' and not bracket_stack:
            # If we're not inside brackets, split here
            parts.append(''.join(current_part))
            current_part = []
        else:
            current_part.append(char)
    
    # Add the last part if there's any
    if current_part:
        parts.append(''.join(current_part))
    
    return parts

def process_speical_case(record : dict) -> dict :
    """
    If the record is a special case, it will be processed here.
    """
    banned_values = ["signature", "dtype", "memory_format", "layout"]
    idx_to_del = []
    for i, key in enumerate(record.keys()) :
        if key in banned_values :
            idx_to_del.append(key)
    for key in idx_to_del :
        record.pop(key)
    return record

def transfer_older_record_to_newer(record: dict, func_name, package) -> dict:
    all_results = []
    new = {
        "args" : { 
            "is_pos" : [],
            "dtype" : [],
            "name" : [],
            "required" : [],
        },
        "name" : None,
        "package" : None,
        "pass_rate" : 0,
    }
    if package == "tf" :
        new["package"] = "tensorflow"
    record = process_speical_case(record)
    new["name"] = func_name
    new["args"]["name"] = list(record.keys())
    new["args"]["required"] = list([a["required"] for a in record.values()])
    new["args"]["is_pos"] = [a.get("is_pos", False) for a in record.values()]
    new["args"]["value"] = [None for _ in range(len(new["args"]["name"]))]
    dtypes_comb = list([a["dtype"] for a in record.values()])
    combinations = list(itertools.product(*dtypes_comb))
    for comb in combinations :
        new["args"]["dtype"] = list(comb)
        all_results.append(copy.deepcopy(new))

    return all_results

def materalize_func(func_name : str, package : Literal['torch', 'tensorflow'] = 'torch') -> Union[Callable, None] :
    """
    Generate function with given name
    """
    print("materallize : ", package)
    if package == 'torch' :
        import torch 
    elif package == 'tensorflow' :
        import tensorflow as tf 
    elif package == 'numpy' :
        import numpy 
    else :
        pass
    function_str = func_name
    function_parts = function_str.split('.')
    if len(function_parts) < 2:
        return eval(function_str)
    module_name = '.'.join(function_parts[:-1])
    function_name = function_parts[-1]

    if hasattr(eval(module_name), function_name):
        return getattr(eval(module_name), function_name)
        
    else:
        raise AttributeError(f"Function '{function_str}' does not exist.")

    
def add_required_info(data, func_name, package) :
    data['title'] = func_name
    data['pass_rate'] = 0.0
    data['rules'] = {}
    data['package'] = package

def gen_cfg(cfg_path : str, add_info : Optional[Dict[str,Any]] = None) -> str :
    """
    workspace : /home/ 
    func_name : a.b.c 
    ==> generate /home/a/b/c.yaml file and return file_path. 
    return workspace, last func name part.
    """
    info={}
    paths = cfg_path.split('.')[0] # get rid of .yaml
    paths = paths.split('/')
    for i, path in enumerate(paths) : 
        if path == 'tf' or path == 'torch' : 
            break
    full_func_name = '.'.join(paths[i:])
    info = add_required_info(full_func_name)
    with open(cfg_path, 'w') as outfile:
        yaml.dump(info, outfile)

class TypeGenerator() :
    def __init__(self, 
                 save_dir,
                 func_name,
                 package,
                 inferencer : Inferencer) : # # API names that we should extract(every functions in tf.keras)
        self.save_dir = save_dir
        self.func_name = func_name
        self.package = package
        self.func = materalize_func(self.func_name, package)
        # self.find_aliases()
        self.inferencer = inferencer
        self.args_info = {}
        # self.args_info = add_required_info(self.func_name, package)
        self.only_forward=None
        self.def_args_info = {}
        self.forward_args_info = {}
        self.validated = True
        self.infered = None 
        self.prompts = None 
        self.new = False
        self.undefined_tag = None
        self.inferencer.change_to_gpt4()
        self.generated = False
        if len(self.args_info) == 0 :
            self.init()
            self.args_info.update(self.def_args_info)
            self.args_info.update(self.forward_args_info) 
            if self.is_extractable() :
                self.gen()
                self.new = True
                # if self._precheck() : 
                #     self.gen()
        else : 
            self.generated = True 
        if self.generated :
            self.give_pos()
            generated_results = transfer_older_record_to_newer(self.args_info, func_name, package)
            self.dump(generated_results)
        else :
            TRAIN_LOG.error(f"{self.func_name} Inferencing failed")

    def dump(self, records) : 
        for idx, record in enumerate(records) :
            converted = transform_record_for_saving(record)
            for i in range(len(converted['args']['dtype'])) :
                for i, dtype in enumerate(converted['args']['dtype']) : 
                    if isinstance(dtype, str) :
                        converted['args']['dtype'][i] = dtype 
                    elif isinstance(dtype, DType) :
                        converted['args']['dtype'][i] = dtype.to_abs().to_str() 
                    else :
                        converted['args']['dtype'][i] = dtype.to_str() 

            save_record(converted, self.save_path(idx))

    def init(self) : 
        self.def_args_info = self.look_up_sig(self.func)
        self.check_class_call()
    def is_extractable(self) :
        if len(self.args_info) == 0 :
            if not hasattr(self.func, '__doc__') or \
                self.func.__doc__ is None or \
                len(self.func.__doc__) == 0 :
                return False
        return True
    def give_pos(self) : 
        for key in self.args_info.keys() : 
            if self.args_info[key]['init'] :
                self.def_args_info.update({key : self.args_info[key]}) 
            else :
                self.forward_args_info.update({key : self.args_info[key]})

    def get_forward_keys(self) : 
        return list(self.forward_args_info.keys())
    def get_def_keys(self) : 
        return list(self.def_args_info.keys())
    def gen(self, num_try=3) : 
        cnt=0
        while cnt <num_try and self._gen()==False : 
            cnt+=1
            self.inferencer.change_to_gpt4()
        
    def mark_pos(self, res) :
        if self.only_forward :
            for key in res.keys() :
                res[key]['init'] = False
        else :
            if not len(self.forward_args_info) == 0 :
                for key in self.forward_args_info.keys() :
                    res[key]['init'] = False
            if not len(self.def_args_info) == 0 :
                for key in self.def_args_info.keys() :
                    res[key]['init'] = True
            for key in res.keys() : 
                if not 'init' in res[key].keys() :
                    res[key]['init'] = False
        return res       
    def _gen(self) : 
        try :
            self.run_llm()
            res = self.interpret_res(self.infered)
            res = self.materalize(res)
            res = self.mark_pos(res) 
            self.update(self.args_info, res)
            if self.check() :
                self.generated = True 
                return True 
            else : 
                return False
        except : 
            return False

    def save_path(self, idx=0) :
        return os.path.join(self.save_dir, f"{self.func_name.replace('.', '/')}-{idx}.yaml")
    
    def update(self, target, new) : 
        if len(target) == 0 :
            target.update(new)
        for key in target.keys() :
            if isinstance(target[key], dict) :
                self.update(target[key], new[key])
            else :
                if key in new.keys() : 
                    target[key] = new[key]
        for key in new.keys() :
            if key not in target.keys() :
                target[key] = new[key]

    def __call__(self) : 
        return self.args_info

    def materalize(self, types_dict) : 
        for key in types_dict.keys() :
            for attr in types_dict[key].keys() :
                if attr == "dtype" :
                    materalized = materalize_dtypes(types_dict[key][attr])
                    if materalized is None :
                        materalized = self.undefined_tag
                    types_dict[key][attr] = materalized
                elif attr == "required" :
                    if type(types_dict[key][attr]) == bool : 
                        pass
                    elif types_dict[key][attr].lower() == "true" :
                        types_dict[key][attr] = True 
                    elif types_dict[key][attr].lower() == "false" :
                        types_dict[key][attr] = False  
                    else :
                        raise UnboundLocalError()
        return types_dict
    def _precheck(self) : 
        # type check failed -> it is required, and no default value -> cannot gen. regen.
        changed = False 
        for key in self.args_info.keys() :
            if not self.type_check(self.args_info[key]["dtype"]) :
                self.args_info[key]["dtype"] = None
                changed = True
        return changed
    def check(self) : 
        # type check failed -> it is required, and no default value -> cannot gen. regen.
        rm_list = []
        for key in self.args_info.keys() :
            if not self.type_check(self.args_info[key]["dtype"]) :
                if self.args_info[key]['required'] :
                    TRAIN_LOG.error(f"The type of {key}(={self.args_info[key]['dtype']})(REQUIRED) cannot utilized, dtype infer failed")
                    return False
                else :
                    rm_list.append(key)
        for key in rm_list :
            if key in self.args_info.keys() :
                TRAIN_LOG.error(f"The type of {key}(={self.args_info[key]['dtype']})cannot utilized, delete this param(={key})")
                self.args_info.pop(key)
        return True
    
    def type_check(self, dtype) -> bool :
        return dtype is not None 
        # return RandomGenerator.check(dtype)     

    def look_up_sig(self, func) : 
        import inspect
        dtypes_info = {}
        try :
            sig = {name : parameter
                            for name, parameter in 
                            inspect.signature(func).parameters.items() if name not in ['self', 'args', 'kwargs']}  
        except :
            return dtypes_info
        
        for name, parameters in sig.items() :
            dtypes_info[name] = {}
            if parameters.default == inspect._empty :
                dtypes_info[name]['default'] = self.undefined_tag
                dtypes_info[name]['required'] =True
            else : 
                dtypes_info[name]['default'] = parameters.default 
                dtypes_info[name]['required'] = False      
                   
            if parameters.annotation == inspect._empty :
                dtypes_info[name]['dtype'] = self.undefined_tag
            else :
                dtypes_info[name]['dtype'] = parameters.annotation
        return dtypes_info
    def check_class_call(self) : 
        """ 
        Set self.forward_args_info and self.def_args_info
        """
        if hasattr(self.func, 'forward') :
            self.only_forward = False
            self.call_func_name = 'forward'
            self.forward_args_info = self.look_up_sig(getattr(self.func, self.call_func_name))
        elif hasattr(self.func, 'call') :
            self.only_forward = False
            self.call_func_name = 'call'
            self.forward_args_info = self.look_up_sig(getattr(self.func, self.call_func_name))
        else :
            self.only_forward = True
            self.forward_args_info = copy.deepcopy(self.def_args_info)
            self.def_args_info = {}

    def run_llm(self) :
        self.prompts = self._gen_prompts()
        TRAIN_LOG.info(f"type_gen prompts :\n{self.prompts}")
        self.infered = self.inferencer.inference(self.prompts)
        TRAIN_LOG.info(f"type_gen results :\n{self.infered}")

    def _gen_prompts(self) :
        if self.package == 'torch' :
            return torch_type_hint_infer_prompts(self.func, self.args_info, undefined_tag=self.undefined_tag)
        elif self.package == 'numpy' :
            return numpy_type_hint_infer_prompts(self.func, self.args_info, undefined_tag=self.undefined_tag)
        elif self.package == 'tensorflow' :
            return tf_type_hint_infer_prompts(self.func, self.args_info, undefined_tag=self.undefined_tag)
        else :
            raise NotImplementedError(f"Package {self.package} is not supported")
    
    def preprocess(self, res) :
        return res.replace("False","false").replace("True","true").replace("None","null").replace(',}','}')
    def interpret_res(self,res) :
        import json
        if res is not None :
            res = self.preprocess(res)
            res = self.extract_json(res)
            res = json.loads(res)
            TRAIN_LOG.debug(f"interpret_res : {res}")
            for key, value in res.items():
                if value is None:
                    res[key] = None
        return res
    def extract_json(self, res : str) :
        json_start = res.index('{')  # Find the start of the JSON object
        json_end = res.rindex('}')  # Find the end of the JSON object
        res = res[json_start:json_end + 1]  # Extract the JSON string including braces
        return res
    def _extract_matched_part(self, res : str) : 
        import re
        pattern = r"\{[^}]+\}"

        matches = re.findall(pattern, res) # Find all occurrences of the pattern in the string
        if matches : return matches[0]
        else : return None

    def convert(self, res : Dict[str, Any]) : 
        converted = {}
        for required, item in res.items() :
            for key, info in item.items() :
                converted[key] = info 
                if required=="required" : 
                    converted[key]["required"] = True 
                else : 
                    converted[key]["required"] = False
        return converted