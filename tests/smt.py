
import hydra
from omegaconf import DictConfig
from deepconstr.grammar.dtype import AbsDType, AbsTensor
from nnsmith.logger import SMT_LOG
from deepconstr.gen.noise import gen_noise
from deepconstr.gen.solve import gen_val

_all_dtypes = [
    AbsDType.int,
    AbsDType.float,
    AbsDType.bool,
    AbsDType.str,
    AbsDType.complex,
    AbsTensor(),
    AbsDType.int.to_iter(),
    AbsDType.float.to_iter(),
    AbsDType.bool.to_iter(),
    AbsDType.str.to_iter(),
    AbsDType.complex.to_iter(),
    AbsTensor.to_iter(),
]

def test_gen_noise(name, dtype, length=2) :
    noises = gen_noise(name, dtype, length, 1)
    SMT_LOG.info(f"noises : {noises}")
    return noises

def test_smt(names, dtypes, constrs =[], noise_prob=1.0):

    args_types = {name : dtype for name, dtype in zip(names, dtypes)}
    values = gen_val(
                num_of_try = 10,
                args_types=args_types, 
                constrs=constrs, # constraints
                noise_prob=noise_prob,
                allow_zero_length_rate=0.2,
                allow_zero_rate=0.3,
                api_name="test"
            )
    SMT_LOG.info(f"values : {values}")
    return values

@hydra.main(version_base=None, config_path="../nnsmith/config/", config_name="main")
def main(cfg: DictConfig):
    dtypes = [
        AbsDType.int,
        AbsDType.float,
        AbsDType.bool,
        AbsDType.str,
        # AbsDType.complex,
        AbsTensor(),
        AbsDType.int.to_iter(),
        AbsDType.float.to_iter(),
        AbsDType.bool.to_iter(),
        AbsDType.str.to_iter(),
        # AbsDType.complex.to_iter(),
        AbsTensor.to_iter(),
    ]
    for _ in range(20) :
        test_smt([str(dtype) for dtype in dtypes], dtypes)


if __name__ == "__main__":
    main()