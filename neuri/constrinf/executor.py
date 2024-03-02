
from concurrent.futures import ProcessPoolExecutor
import functools
from multiprocessing import Manager, Pool
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple
from neuri.autoinf.instrument.op import OpInstance
from neuri.constrinf import record_args_info
from neuri.logger import TRAIN_LOG
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

def clear_global_constraints() :
    global _gloabl_constraints
    _gloabl_constraints = []

def worker(model, record, noise=0.8, allow_zero_length_rate=0.1, allow_zero_rate=0.1, num_of_try=30):
    chosen_dtype = {}
    concretized_values = {}
    for i_arg, arg_name in enumerate(record['args']['name']):
        if record['args']['dtype_obj'][i_arg] is None :
            chosen_dtype[arg_name] = None
            TRAIN_LOG.warning(f"Unidentiable dtype for {arg_name} : {record['args']['dtype'][i_arg]}")
        elif len(record['args']['dtype_obj'][i_arg]) > 0:
            chosen_dtype[arg_name] = random.choice(record['args']['dtype_obj'][i_arg])
        else:
            chosen_dtype[arg_name] = record['args']['dtype_obj'][i_arg]
    
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
    for key in values:
        if isinstance(values[key], list):
            concretized_values[key] = [v.concretize(inst.input_symb_2_value, only_shape=True) if hasattr(v, 'concretize') else v for v in values[key]]
        else :
            concretized_values[key] = values[key].concretize(inst.input_symb_2_value, only_shape=True) if hasattr(values[key], 'concretize') else values[key]
    TRAIN_LOG.debug(f"Concretized values of {record['name']}: {concretized_values}")
    try:
        # Assuming record_args_info is a function to log or record argument info
        # self.record_args_info(record, values)  # Placeholder for actual logging or recording
        res_or_bug, abs_ret_list = model.execute_op(inst)
        if res_or_bug == NotImplemented :
            err_instance = ErrorMessage("NotImplemented", "", concretized_values, chosen_dtype)
            err_instance.error_type = NotImplementedError
            return False, err_instance
        return True, ErrorMessage("no error", "", concretized_values, chosen_dtype)  # Assuming execution success
    except Exception as e:
        error_instance = ErrorMessage(e, traceback.format_exc(), concretized_values, chosen_dtype)
        assert isinstance(error_instance, ErrorMessage)
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
        try :
            TRAIN_LOG.info(f"Executing {ntimes} times")
            set_global_constraints(constraints) # to be used in worker(parallel execution)
            with ProcessPoolExecutor(max_workers=self.parallel) as executor:
                # Generate a list of future tasks
                worker_fn = functools.partial(worker, self.model, *args, **kwargs)
                futures = [executor.submit(worker_fn) for _ in range(ntimes)]
                results = [future.result() for future in futures]
            clear_global_constraints()
            return results
        except Exception as e:
            TRAIN_LOG.error(f"Error in execute: {e}, maybe child process core dumped")
            err_instance = ErrorMessage(MemoryError(), traceback.format_exc(), {}, {})
            return [[False, err_instance] for _ in range(ntimes)]

Exception