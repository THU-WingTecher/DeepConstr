
from concurrent.futures import ProcessPoolExecutor
import functools
from multiprocessing import Manager, Pool
import random
from typing import Any, Dict, List, Optional, Tuple
from neuri.autoinf.instrument.op import OpInstance
from neuri.constrinf import record_args_info
from neuri.specloader.smt import gen_val
from neuri.constrinf.errmsg import ErrorMessage

def contains_ctypes(obj, visited=None):
    import ctypes
    if visited is None:
        visited = set()
    
    # Avoid infinite recursion for circular references
    if id(obj) in visited:
        return False
    visited.add(id(obj))

    if isinstance(obj, ctypes._SimpleCData):
        return True

    if isinstance(obj, dict):
        return any(contains_ctypes(v, visited) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(contains_ctypes(item, visited) for item in obj)
    elif hasattr(obj, '__dict__'):
        return contains_ctypes(vars(obj), visited)
    
    return False

_gloabl_constraints = []
def set_global_constraints(constraints) :
    global _gloabl_constraints
    _gloabl_constraints = constraints

def worker(model, record, noise, allow_zero_length_rate, allow_zero_rate, num_of_try):
    chosen_dtype = {}
    for i_arg, arg_name in enumerate(record['args']['name']):
        if len(record['args']['dtype'][i_arg]) > 0:
            chosen_dtype[arg_name] = random.choice(record['args']['dtype'][i_arg])
        else:
            chosen_dtype[arg_name] = record['args']['dtype'][i_arg]
    
    values = gen_val(
                num_of_try,
                chosen_dtype, 
                _gloabl_constraints, # constraints
                noise_prob=noise,
                allow_zero_length_rate=allow_zero_length_rate,
                allow_zero_rate=allow_zero_rate,
                api_name=record['name']
            )
    if values is None : 
        return None
    record_args_info(record, values)
    inst = OpInstance(record)
    concretized_values = {
        key : values[key].concretize(inst.input_symb_2_value, only_shape=True) for key in values
    }
    try:
        # Assuming record_args_info is a function to log or record argument info
        # self.record_args_info(record, values)  # Placeholder for actual logging or recording
        res_or_bug = model.execute_op(inst)
        return True, ErrorMessage("no error", concretized_values, chosen_dtype)  # Assuming execution success
    except Exception as e:
        error_instance = ErrorMessage(str(e), concretized_values, chosen_dtype)
        return False, error_instance  # Return error state and message
class Executor:
    def __init__(self, model, parallel=8) :
        self.model = model
        self.parallel = parallel
    def _execute(self, ntimes, *args, **kwargs) -> Optional[List[Tuple[bool, ErrorMessage]]]:
        results = []
        unable_to_gen_tor = 4
        worker_fn = functools.partial(worker, self.model, *args, **kwargs)
        for _ in range(ntimes):
            res = worker_fn()
            if res is None :
                unable_to_gen_tor -= 1
                if unable_to_gen_tor == 0 :
                    return None
            else :
                unable_to_gen_tor = 4
                results.append(res)
        return results
    def execute(self, ntimes, constraints, *args, **kwargs) -> Optional[List[Tuple[bool, ErrorMessage]]]:
        """
        ctypes pickling problem.
        To support parallel execution, 
        Constr is converted to executable(Callable)
        dtypes are converted to z3 types
        Executes the given operation (opinstance) multiple times in parallel using multicore processing.
        Classifies the program result (success or error) and deals with error messages along with argument values.
        
        Args:
        - opinstance: The operation instance to be executed.
        - ntimes: The number of times to execute the operation.
        
        Returns:
        A tuple (success_rate: float, error_messages: dict) where success_rate is the ratio of successful executions
        and error_messages is a dictionary mapping error messages to their corresponding argument values.
        """
        # for key, item in kwargs.items() :
        #     if contains_ctypes(item) :
        #         for k, v in item.items() :
        #             if contains_ctypes(v) :
        #                 print(k, v[0][0])
 
        set_global_constraints(constraints) # to be used in worker(parallel execution)
        with ProcessPoolExecutor(max_workers=self.parallel) as executor:
            # Generate a list of future tasks
            worker_fn = functools.partial(worker, self.model, *args, **kwargs)
            futures = [executor.submit(worker_fn) for _ in range(ntimes)]
            results = [future.result() for future in futures]
        
        return results
        
