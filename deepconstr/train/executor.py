import functools
import multiprocessing
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple
from deepconstr.error import InternalError
from deepconstr.grammar.dtype import AbsVector
from deepconstr.grammar.op import OpPlaceholder
from deepconstr.grammar.base import DEFAULT_DTYPE_CONSTR
from deepconstr.gen.record import record_args_info
from deepconstr.logger import GEN_LOG
from deepconstr.gen.solve import gen_val
from deepconstr.train.errmsg import ErrorMessage

NOERR_MSG = "no error"

_gloabl_constraints = []
_dtype_constrs_executable = []
def set_global_constraints(constraints) :
    global _gloabl_constraints
    _gloabl_constraints = constraints

def clear_global_constraints() :
    global _gloabl_constraints
    _gloabl_constraints = []

def is_normal_error(result) :
    if result is None :
        return False
    _, error_instance = result
    return not isinstance(error_instance.error_type, (InternalError, TimeoutError))

def worker(model, record, noise=0.8, allow_zero_length_rate=0.1, allow_zero_rate=0.1, num_of_try=30):
    chosen_dtype = {}
    concretized_values = {}
    dtype_constrs = []
    for i_arg, arg_name in enumerate(record['args']['name']):
        if record['args']['dtype_obj'][i_arg] is None :
            chosen_dtype[arg_name] = None
            GEN_LOG.debug(f"Unidentiable dtype for {arg_name} : {record['args']['dtype'][i_arg]}")
        elif len(record['args']['dtype_obj'][i_arg]) > 0:
            chosen_dtype[arg_name] = random.choice(record['args']['dtype_obj'][i_arg])
        else:
            chosen_dtype[arg_name] = record['args']['dtype_obj'][i_arg]
        if isinstance(chosen_dtype[arg_name], AbsVector) :
            dtype_constrs.append(_dtype_constrs_executable(arg_name))

    values = gen_val(
                num_of_try,
                chosen_dtype, 
                _gloabl_constraints, # constraints
                noise_prob=noise,
                allow_zero_length_rate=allow_zero_length_rate,
                allow_zero_rate=allow_zero_rate,
                api_name=record['name'],
                dtype_constrs=dtype_constrs,
            )
    if values is None : 
        return None
    record_args_info(record, values)
    inst = OpPlaceholder(record)
    for key in values:
        if isinstance(values[key], list):
            concretized_values[key] = [v.concretize(inst.input_symb_2_value, only_shape=True) if hasattr(v, 'concretize') else v for v in values[key]]
        else :
            concretized_values[key] = values[key].concretize(inst.input_symb_2_value, only_shape=True) if hasattr(values[key], 'concretize') else values[key]
    GEN_LOG.debug(f"Concretized values of {record['name']}: {concretized_values}")
    try:
        res_or_bug, abs_ret_list = model.execute_op(inst)
        return True, ErrorMessage(NOERR_MSG, "", concretized_values, chosen_dtype, package=model.package)  # Assuming execution success
    except Exception as e:
        error_instance = ErrorMessage(e, traceback.format_exc(), concretized_values, chosen_dtype, package=model.package)
        return False, error_instance  # Return error state and message

def worker_wrapper(worker_fn, return_dict, task_chunk, *args, **kwargs):
    for task_id in task_chunk:
        try:
            result = worker_fn(*args, **kwargs)
        except Exception as e:
            err_instance = ErrorMessage(InternalError(), traceback.format_exc(), {}, {})
            GEN_LOG.error(f"Err execute: {e}, maybe child process core dumped")
            result = [False, err_instance]
        return_dict[task_id] = result

class Executor:
    def __init__(self, model, parallel=8) :
        global _dtype_constrs_executable
        self.model = model
        self.parallel = parallel
        _dtype_constrs_executable = DEFAULT_DTYPE_CONSTR.get(self.model.package)

    def execute(self, ntimes, constraints, *args, **kwargs) -> Optional[List[Tuple[bool, ErrorMessage]]]:
        GEN_LOG.info(f"Executing {ntimes} times")
        set_global_constraints(constraints) # to be used in worker(parallel execution)
        res = self.parallel_execute(ntimes, *args, **kwargs) \
            if self.parallel != 1 else self._execute(ntimes, *args, **kwargs)
        clear_global_constraints()
        return res
    def _execute(self, ntimes, *args, **kwargs) -> Optional[List[Tuple[bool, ErrorMessage]]]:
        results = []
        worker_fn = functools.partial(worker, self.model, *args, **kwargs)
        for _ in range(ntimes):
            res = worker_fn()
            if res is None :
                results.append(res)
            else :
                results.append(res)
        return results
    def parallel_execute(self, ntimes, *args, **kwargs) -> Optional[List[Tuple[bool, ErrorMessage]]]:
        # def execute_in_parallel(worker, self, ntimes, num_of_task_per_process, *args, **kwargs):
        num_of_task_per_process = ntimes // self.parallel
        manager = multiprocessing.Manager()
        timeout = num_of_task_per_process * 1.5
        return_dict = manager.dict()
        worker_fn = functools.partial(worker, self.model, *args, **kwargs)
        processes = []
        results = []
        tasks_per_process = max(ntimes // self.parallel, 1)  # Ensure at least one task per process
        for i in range(self.parallel):
            start_index = i * tasks_per_process
            end_index = start_index + tasks_per_process
            if i == self.parallel - 1:  # Ensure the last process gets any remaining tasks
                end_index = ntimes
            
            task_chunk = range(start_index, end_index)  # Assuming tasks can be identified by their index

            # Start the process with its chunk of tasks
            p = multiprocessing.Process(target=worker_wrapper, args=(worker_fn, return_dict, task_chunk) + args, kwargs=kwargs)
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout) 

        # Handle processes that did not finish in time
        for p in processes:
            if p.is_alive():
                p.terminate()  # Terminate process

        for i in range(len(return_dict)) :
            if return_dict.get(i, None) is None :
                pass 
            else :
                results.append(return_dict[i])
        return results