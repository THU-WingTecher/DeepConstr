
from multiprocessing import Pool
import random
from typing import List, Optional, Tuple
from neuri.autoinf.instrument.op import OpInstance
from neuri.constrinf import record_args_info
from neuri.specloader.smt import gen_val
from neuri.constrinf.errmsg import ErrorMessage


class Executor:
    def __init__(self, model, parallel=8) :
        self.model = model
        self.parallel = parallel
    def execute(self, record, constraints, ntimes, *args, **kwargs) -> Optional[List[Tuple[bool, ErrorMessage]]]:
        """
        Executes the given operation (opinstance) multiple times in parallel using multicore processing.
        Classifies the program result (success or error) and deals with error messages along with argument values.
        
        Args:
        - opinstance: The operation instance to be executed.
        - ntimes: The number of times to execute the operation.
        
        Returns:
        A tuple (success_rate: float, error_messages: dict) where success_rate is the ratio of successful executions
        and error_messages is a dictionary mapping error messages to their corresponding argument values.
        """
        # Function to wrap _execute call for multiprocessing, capturing all required arguments
        def worker(*args, **kwargs):
            return self._execute(record, constraints, *args, **kwargs)
        
        with Pool(processes=self.parallel) as pool:
            args_list = [(i,) for i in range(ntimes)]  # Example: args for each execution
            # Execute in parallel, note: Assuming `worker` adapts to _execute_with_error_instance
            results = pool.starmap(worker, args_list)
        
        return results
    def _execute(self, record, constraints, noise=0.1, allow_zero_length_rate=0.1, allow_zero_rate=0.1, num_of_try=30):
        """
        Execute an API operation based on the given record, considering argument data types,
        constraints, and other specified parameters.
        
        Args:
        - record (dict): The record containing API name, args, dtype, constraints, etc.
        - noise (float): Probability of introducing noise into argument values.
        - allow_zero_length_rate (float): Rate of allowing zero-length values.
        - allow_zero_rate (float): Rate of allowing zero values.
        
        Returns:
        Tuple of execution success (bool) and error message (str), if any.
        """
        # Step 1: Select argument data types
        chosen_dtype = {}
        for i_arg, arg_name in enumerate(record['args']['name']):
            if len(record['args']['dtype'][i_arg]) > 0:
                chosen_dtype[arg_name] = random.choice(record['args']['dtype'][i_arg])
            else:
                chosen_dtype[arg_name] = record['args']['dtype'][i_arg]
        
        # Step 2: Generate values based on constraints
        # Placeholder for constraint solving and value generation logic
        values = gen_val(chosen_dtype, 
                        constraints,
                        noise_prob=noise,
                        allow_zero_length_rate=allow_zero_length_rate,
                        allow_zero_rate=allow_zero_rate,
                        num_of_try=num_of_try,
                        api_name=record['name'])
        if values is None : 
            return None
        record_args_info(record, values)
        # Placeholder for argument recording and operation execution
        try:
            # Assuming record_args_info is a function to log or record argument info
            # self.record_args_info(record, values)  # Placeholder for actual logging or recording
            
            inst = OpInstance(record)
            res_or_bug = self.model.execute_op(inst)
            return True, ""  # Assuming execution success
        except Exception as e:
            error_instance = ErrorMessage(str(e), values, chosen_dtype)
            return False, error_instance  # Return error state and message
        
