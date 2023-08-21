import pickle

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    gir_path = cfg["debug"]["gir_path"]
    with open(gir_path, "rb") as f:
        gir = pickle.load(f)
    gir.debug()


if __name__ == "__main__":
    main()
