from typing import Any, Dict

def assert_allclose(
    actual: Dict[str, Any],
    desired: Dict[str, Any],
    actual_name: str,
    oracle_name: str,
    equal_nan=False,
    rtol=1e-2,
    atol=1e-3,
):
    # Unified assert allclose -> choose function by the tensor type
    akeys = set(actual.keys())
    dkeys = set(desired.keys())
    if akeys != dkeys:
        raise KeyError(f"{actual_name}: {akeys} != {oracle_name}: {dkeys}")

    for key in akeys:
        lhs = actual[key]
        rhs = desired[key]

        # check if lhs is np.ndarray
        import numpy as np
        if isinstance(lhs, np.ndarray) and isinstance(rhs, np.ndarray):
            np.testing.assert_allclose(
                lhs,
                rhs,
                equal_nan=equal_nan,
                rtol=rtol,
                atol=atol,
                err_msg=f"{actual_name} != {oracle_name} at {key}",
            )
        else :
            import torch
            if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
                is_same = torch.allclose(
                    lhs,
                    rhs,
                    equal_nan=equal_nan,
                    rtol=rtol,
                    atol=atol,
                )
                if not is_same :
                    raise ValueError(f"{actual_name} != {oracle_name} at {key}")
            else :
                import tensorflow as tf 
                if isinstance(lhs, tf.Tensor) and isinstance(rhs, tf.Tensor):
                    tf.test.TestCase().assertAllClose(
                        lhs,
                        rhs,
                        equal_nan=equal_nan,
                        rtol=rtol,
                        atol=atol,
                        err_msg=f"{actual_name} != {oracle_name} at {key}",
                    )
                else :
                    raise NotImplementedError(f"Unknown type: {type(lhs)}")

