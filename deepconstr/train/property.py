def materalize_func(func_name : str, package : Literal['torch', 'tf'] = 'torch') -> Union[Callable, None] :
    """
    Generate function with given name
    """
    print("materallize : ", package)
    if package == 'torch' :
        import torch 
    elif package == 'tf' :
        import tensorflow as tf 
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

    
def add_required_info(func_name) :
    data = {}
    data['title'] = func_name
    data['pass_rate'] = 0.0
    data['rules'] = {}
    return data
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

def tf_get_func_doc(func) :
    raw_doc="" 
    if hasattr(func, '__doc__') :
        raw_doc = getattr(func, '__doc__')
        if raw_doc is None : return ""
        if len(raw_doc) == 0 : 
            return ""
        else : 
            if "Args" in raw_doc :
                doc = "Args" + ''.join(raw_doc.split("Args")[1:])
            else :
                doc = raw_doc
            if "Returns" in doc : 
                doc = ''.join(doc.split("Returns")[:-1])
    return doc


def tf_type_hint_infer_prompts(func, dtypes=None, undefined_tag="undefined") :
    prompts=f"After read the {func.__name__} doc {tf_get_func_doc(func)}\n"

    if len(dtypes)==0 :
        prompts+=f"""Analyze the api parameter information  follwoing below steps, only output final json result. Do not explain.
your output format should be matched with input form of json.loads.\n"""
        step_discription="""First, list every input parameters that the api has, 
Second, collect type and default value of input parameters.
Finally, print the final results as dictionary value to the key of input parameter name.
"""
        prompts+=step_discription
        Q=""

    else :

        for arg_name, info in dtypes.items() :
            for key, item in info.items() : 
                if not isinstance(dtypes[arg_name][key], str) : 
                    dtypes[arg_name][key] =  '"'+ str(dtypes[arg_name][key]) + '"'

        serizlized = str(dtypes).replace('\'','"').replace('""', '"')
        prompts+=f"""fill the given api parameters infomation by replace the "{undefined_tag}". You can only replace the "{undefined_tag}" value with JSON-serializable object """
        Q=f"\nQ : {serizlized}"

    examples="""
ex 1 : {{"input" : {{"default" : None, "required" :true, "dtype" : "tf.tensor",}}, "ksize" : {{"default" : None, "required" :true, "dtype" : "tf.tensor",}}, "strides" : {{"default" : None, "required" :true, "dtype" : "int, List[int]",}}, "padding" : {{"default" : None, "required" :true, "dtype" : "valid", "same"}}}}
ex 2 : {{"axis" : {{"default" : -1, "required" : false, "dtype" : "int",}}, "x" : {{"default" : None, "required" : false, "dtype" : "tensor",}}, "center" : {{"default" : true, "required" : true, "dtype" : "tensor",}}, "beta_initializer : {{"default" : "zeros", "required" :true, "dtype" : "zeros",}}, "beta_regularizer" : {{"default" : "None", "required" : false, "dtype" :"not defined"}}}}
"""        
    return prompts+examples+Q

def torch_doc_filter(doc) :
    
    function_name_pattern = r"\w+\(.*\)"
    args_content_pattern = r"Args:\n(.+?)(\n\n|$)"
    function_name = re.findall(function_name_pattern, doc)
    args_content = re.findall(args_content_pattern, doc, re.DOTALL)
    if function_name : 
        function_name = function_name[0].strip()
    else :
        function_name = ""
    
    if args_content and args_content[0] :
        args_content = args_content[0][0].strip()
    else :
        args_content = ""
    

    # Regular expression to find content after "Args:"
    # args_content = re.findall(args_content_pattern, doc, re.DOTALL)[0][0].strip()

    return '\n'.join([function_name, args_content])

def torch_get_func_doc(func) :
    raw_doc="" 
    if hasattr(func, '__doc__') :
        doc = func.__doc__
        return torch_doc_filter(doc)
    else :
        return raw_doc

def torch_type_hint_infer_prompts(func, dtypes=None, undefined_tag="undefined") :
    prompts=f"After read the {func.__name__} doc. ```{torch_get_func_doc(func)}```\n"

    if len(dtypes)==0 :
        prompts+=f"""Analyze the api parameter information  follwoing below steps, only output final json result. Do not explain.
your output format should be matched with input form of json.loads.\n"""
        step_discription="""First, list every input parameters that the api has by checking parameter between "()"
, 
Second, collect type and default value of input parameters.
Finally, print the final results as dictionary value to the key of input parameter name.
"""
        prompts+=step_discription
        Q=""

    else :
        for arg_name, info in dtypes.items() :
            for key, item in info.items() : 
                if not isinstance(dtypes[arg_name][key], str) : 
                    dtypes[arg_name][key] =  '"'+ str(dtypes[arg_name][key]) + '"'

        serizlized = str(dtypes).replace('\'','"').replace('""', '"')
        prompts+=f"""fill the given api parameters infomation by replace the "{undefined_tag}". You can only replace the "{undefined_tag}" value with JSON-serializable object """
        Q=f"\nQ : {serizlized}"

    examples="""
ex 1 : {{"input" : {{"default" : None, "required" :true, "dtype" : "tensor",}}, "ksize" : {{"default" : None, "required" :true, "dtype" : "tensor",}}, "strides" : {{"default" : None, "required" :true, "dtype" : "int, List[int]",}}, "padding" : {{"default" : None, "required" :true, "dtype" : "valid", "same"}}}}
ex 2 : {{"axis" : {{"default" : -1, "required" : false, "dtype" : "int",}}, "x" : {{"default" : None, "required" : false, "dtype" : "tensor",}}, "center" : {{"default" : true, "required" : true, "dtype" : "tensor",}}, "beta_initializer : {{"default" : "zeros", "required" :true, "dtype" : "zeros",}}, "beta_regularizer" : {{"default" : "None", "required" : false, "dtype" :"not defined"}}}}
"""        
    return prompts+examples+Q
class TypeGenerator() :
    def __init__(self, 
                 cfg : Dict[str, Any], 
                 inferencer : Inferencer) : # # API names that we should extract(every functions in tf.keras)
        self.cfg = cfg
        self.func_name = cfg['title']
        self.func = materalize_func(self.func_name, cfg['package'])
        self.find_aliases()
        self.inferencer = inferencer
        self.tensor_type_keys = []
        self.args_info = {}
        self.only_forward=None
        self.def_args_info = {}
        self.forward_args_info = {}
        self.validated = True
        self.infered = None 
        self.prompts = None 
        self.new = False
        self.undefined_tag = "not defined"
        self.args_info = self.load_from_cfg()
        self.inferencer.change_to_gpt4()
        self.generated = False
        if len(self.args_info) == 0 :
            self.init()
            self.args_info.update(self.def_args_info)
            self.args_info.update(self.forward_args_info) 
            if self.is_extractable() :
                self.gen()
                self.new = True
                if self._precheck() : 
                    self.gen()
        else : 
            self.generated = True 
        if self.generated :
            if self.check() :
                self.give_pos()
            else :
                self.cfg['skipped'] = 'uncompatiable type'
                self.generated=False
        else :
            LOGGER.error(f"{self.func_name} Inferencing failed")

    def find_aliases(self) :
        found = False 
        cur_func = self.func
        new = True 
        cnt=0
        while new :
            if hasattr(cur_func, '__doc__') :
                if find_aliases := self._find_aliases(cur_func.__doc__) :
                    cnt+=1 
                    if cnt > 3 : break
                    self.cfg['alias'] = find_aliases[0]
                    cur_func = materalize_func(find_aliases[0], self.cfg['package'])
                    new = True 
                    continue
            new = False 

        self.func = cur_func

    def _find_aliases(self, doc_string):
        if doc_string is None : return None 
        alias_patterns = [
            r'See :func:`([\w.]+)`',  # Pattern like "See :func:`torch.gt`"
            r'In-place version of :meth:`([\w.]+)`',  # Pattern like "In-place version of :meth:`~Tensor.unsqueeze`"
            r'Alias for :func:`([\w.]+)`',  # Pattern like "Alias for :func:`torch.add`"
            r'See :class:`([\w.]+)`'
        ]
        
        aliases = []
        
        for pattern in alias_patterns:
            aliases.extend(re.findall(pattern, doc_string))
        aliases = [aliase.replace('~','torch.') for aliase in aliases if not aliase[1:].startswith('torch.')]
        if aliases : LOGGER.info("found aliases : "+str(aliases[0]))
        return aliases

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
    
    def load_from_cfg(self) -> Dict[str, Any] : ## will change
        ## cfg <- yaml file
        types_dict={}
        res={}
        if 'constraints' in self.cfg.keys() :
            # types_dict = load_types_dict(self.cfg, self.package)
            types_dict = copy.deepcopy(self.cfg['constraints'])
            for key, value in types_dict.items():
                if value is None:
                    types_dict[key] = None
        if len(types_dict) == 0 : return {} 

        for name, params in types_dict.items() :
            types_dict[name]['dtype'] = materalize_dtypes(types_dict[name]['dtype'])
        return types_dict 
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
            self.generated = True 
            return True 
        except : 
            return False

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
        "str->typing"
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
                self.args_info[key]["dtype"] = 'not defined'
                changed = True
        return changed
    def check(self) : 
        # type check failed -> it is required, and no default value -> cannot gen. regen.
        rm_list = []
        for key in self.args_info.keys() :
            if not self.type_check(self.args_info[key]["dtype"]) :
                if self.args_info[key]['required'] :
                    LOGGER.error(f"The type of {key}(={self.args_info[key]['dtype']})(REQUIRED) cannot utilized, dtype infer failed")
                    return False
                else :
                    rm_list.append(key)
        for key in rm_list :
            if key in self.args_info.keys() :
                LOGGER.error(f"The type of {key}(={self.args_info[key]['dtype']})cannot utilized, delete this param(={key})")
                self.args_info.pop(key)
        return True
    
    def type_check(self, dtype) -> bool :
        return RandomGenerator.check(dtype)     

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
        LOGGER.info(f"type_gen prompts :\n{self.prompts}")
        self.infered = self.inferencer.inference(self.prompts)
        LOGGER.info(f"type_gen results :\n{self.infered}")

    def _gen_prompts(self) :
        from train.prompt import torch_type_hint_infer_prompts, tf_type_hint_infer_prompts
        if self.cfg['package'] == 'torch' :
            return torch_type_hint_infer_prompts(self.func, self.args_info, undefined_tag=self.undefined_tag)
        else :
            return tf_type_hint_infer_prompts(self.func, self.args_info, undefined_tag=self.undefined_tag)
    
    def preprocess(self, res) :
        return res.replace("False","false").replace("True","true").replace("None","null").replace(',}','}')
    def interpret_res(self,res) :
        import json
        if res is not None :
            res = self.preprocess(res)
            res = self.extract_json(res)
            res = json.loads(res)
            LOGGER.debug(f"interpret_res : {res}")
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


api_name = ""
gen_cfg(cfg_save_path)