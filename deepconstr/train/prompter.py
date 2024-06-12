import re 
from typing import Dict, Any, Optional, List
import random
from deepconstr.train.constr import Constraint
from deepconstr.train.errmsg import ErrorMessage, sort_sentences_by_similarity
from deepconstr.utils import formatted_dict

PROMPT_EXAMPLE_PATH = "data/prompts/cot.json"

def load_json(path) :
    import json
    with open(path, 'r') as f :
        data = json.load(f)
    return data

class Prompter() :
    def __init__(self, record : Dict[str, Any]) :
        self.func = record['name']
        self.err_db = load_json(PROMPT_EXAMPLE_PATH)
        self.args_names = record['args']['name']
        self.keys_print = ' | '.join([f"{key}" for key in self.args_names])
        self.cases : Dict[str, List[Dict[str, Any]]] = {'FN' : [], 'FP' : []}
        self.collect_switch : Dict[str, bool] = {'FN' : False, 'FP' : False}
        self.asset = 3
        self.errmsgs_of_FP_cases = []
        self.Q_history = []
        self.msg_pool = set()
    def FP_history(self, prev_answer : Constraint, FP_example, target) -> None :
        """create history prompts with False Positive examples """
        txt = prev_answer.txt
        ex = [f"{key}:{str(vals)}" for key, vals in FP_example.items()]
        template = f"""Your previous answer({txt}) is wrong. there are cases like {', '.join(ex)} that still trigger the error while satisfying the previous answer. 
propose other constraints that would not trigger the below error.
Q :relate this values({FP_example}) to generate constraints that do not trigger -> {target} \nAnswers :"""  
        return template
    def FN_history(self, prev_answer : Constraint, FN_example) -> None :
        """create history prompts with False Negative examples """
        txt = prev_answer.txt
        correct_discription = "complete"
        ex = [f"{key}:{str(vals)}" for key, vals in FN_example.items()]
        template = f"""Your previous answer({txt}) was partly correct. There are cases like {', '.join(ex)} would not trigger the error even though it doesn't satisfy your answer.
Q : try to make constraints {correct_discription} so that it can make the whole input space not to trigger {prev_answer.target}.\nA : """
        return template
    def gen_history(self, prev_answer : Constraint) :
        prompts = ""
        if "FP" in self.cases.keys() and len(self.cases['FP']) > 0 and len(self.cases['FP'][0]) > 0 : # "FP" is more important in general 
            idx =0
            target = self.errmsgs_of_FP_cases[idx]
            prompts+=self.FP_history(prev_answer, self.cases['FP'][idx], target = target) 
        else :
            if "FN" in self.cases.keys() and len(self.cases['FN']) > 0 and len(self.cases['FN'][0]) > 0 :
                prompts+=self.FN_history(prev_answer, self.cases['FN'][0])
        return prompts
    def clear_saved_errmsgs(self) -> None :
        self.initialize_collected_error_messages()
        
    def initialize_collected_error_messages(self) -> None :
        self.collected_error_messages = ''     

    def dynamic_template(self, err_msg, cot, answers) : 
       return f"""Q : {err_msg}\nAnswers : {cot}\n```{answers}```"""
    def task_introduce(self, func_name) :
        return f"""You developed the function {func_name}, and know everything about it. Now, infer the root cause of given error messages with its runtime information, and then formulate the condition that make the error disappear. Think step-by-step as below examples."""
    def Constraint_grammar(self) : 
        return f"""<symbol> ::= {self.keys_print} | type(<symbol>) | len(<symbol>) | <symbol>[<int>] | <symbol>.dim | <symbol>.shape
<ops> ::= + | - | * | / | == | != | > | >= | < | <= | in | not in"""
# <comparator> ::= dtype | integer | tensor.dtype | List[int]"""
    def get_closet_examples(self, err_msg, num_of_ex=2) : 
        sorted_li =[ele[0] for ele in sort_sentences_by_similarity(err_msg.get_core_msg(), list(self.err_db.keys()))[:num_of_ex]]
        sorted_li = sorted_li[::-1]
        examples = '\n'.join([self.dynamic_template(errmsg, self.err_db[errmsg]['cot'], self.err_db[errmsg]['answers']) for errmsg in sorted_li])
        return examples
    
    def question(self, targets : List[ErrorMessage], synthesizer, func_name) :
        target_str = "Q : Based on the given runtime information, formulate constraint that prevent the error.\n" 
        new = None
        while targets and new is None :
            new = targets.pop(0)
            if new.get_core_msg() not in self.msg_pool :
                self.msg_pool.add(new.get_core_msg())
                break
            else :
                new = None

        if new is None :
            new = self.Q_history.pop(0)
        self.Q_history.append(new)
        synthesizer.set_target(new)
        # target_str+= f"""({func_name}({formatted_dict(new.get_values_map(), sep="=", split=", ")}) -> {new.get_core_msg()}"""
        target_str+= f"""({func_name}({formatted_dict(new.get_dtypes_map(), sep="=", split=", ")}) -> {new.get_core_msg()}"""
        target_str+= f"\nWrap the final formula with ```. Correctly match the variable names : {list(new.get_dtypes_map().keys())}\nAnswers :"
        return target_str
    
    def gen_infer_history(self, ans_history : str) : 
        if ans_history is None or len(ans_history) == 0 :
            return ""
        introduce = "avoid same answer with your prev answer : "
        return introduce + '```' + ans_history + '```'
    
    def gen(self, err_msgs : List[ErrorMessage], func_name, synthesizer, prev_answer : Optional[Constraint]=None) :
        prompts = ""
        history = ""
        contexts = ""
        task_introduce= self.task_introduce(func_name)
        # grammar = self.Constraint_grammar() if random.choice([0,1]) else ""
        examples = self.get_closet_examples(
                                            err_msgs[0] if err_msgs else self.Q_history[-1], 
                                            num_of_ex = random.randint(1,3)
                                            )
        # history = self.gen_history(prev_answer) if prev_answer is not None else ""
        # infer_history = self.gen_infer_history(ans_history)
        if len(history) == 0 : 
            contexts = '\n'.join([task_introduce, 
                                # grammar, 
                                examples])
            prompts = self.question(err_msgs, synthesizer, func_name)
        else : 
            contexts = '\n'.join([task_introduce, 
                                # grammar, 
                                examples])
            prompts = self.question(history, func_name)
        return contexts, prompts
    def _CoT_template(self) -> None :
        """
        Q : Errmsg with error triggering args values
        A : State the what cause error, 
        State the what is expected not to cause error(State the what is expected to cause error)
        'Let's see what the args were.' -> State related args.
        Therefore, Left =
        (if needs, state the chain of thought to find out the comparators)
        based on this, ops = 
        Right = 
        
        Answers : Left ops Right\n...
        """
        pass

    def collect_error_messages(self, 
                               error_messages : str, 
                               simple : bool = True,
                                func_argument_print=False,
                                input_argument_print=False
                               ) -> str :
        
        whole_error_messages = self.gen_err_msg(error_messages, 
                                                simple,
                                                func_argument_print,
                                                input_argument_print)
        if len(self.collected_error_messages) + len(whole_error_messages) <= self.max_error_message_tokens :
            self.collected_error_messages += whole_error_messages+'\n'
    def get_last_err_msg(self) -> str :
        return self.collected_error_messages.strip().split('\n')[-1]

    def generate_func_doc(self) -> str :
        doc_str = self.func.__doc__
        return doc_str

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

def numpy_type_hint_infer_prompts(func, dtypes=None, undefined_tag="undefined") :
    prompts=f"After read the {func.__name__} doc. ```{tf_get_func_doc(func)}```\n"

    if len(dtypes)==0 :
        prompts+=f"""Analyze the api parameter information  follwoing below steps, only output final json result. Do not explain.
your output format should be matched with input form of json.loads.\n"""
        step_discription="""First, list every input parameters that the api has by checking parameter between "()", if keyward is not specified, is_pos should be set to true .
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
ex 1 : {{"x1" : {{"default" : None, "required" :true, "dtype" : "array", "is_pos" : True}}, "x2" : {{"default" : None, "required" :true, "dtype" : "array", "is_pos" : True}}, "where" : {{"default" : True, "required" : False, "dtype" : "array, None",}}, "padding" : {{"default" : None, "required" :true, "dtype" : "valid", "same"}}}}
ex 2 : {{"axis" : {{"default" : -1, "required" : false, "dtype" : "int",}}, "x" : {{"default" : None, "required" : false, "dtype" : "tensor",}}, "center" : {{"default" : true, "required" : true, "dtype" : "tensor",}}, "beta_regularizer" : {{"default" : "None", "required" : false, "dtype" :"not defined"}}}}
"""        
    return prompts+examples+Q