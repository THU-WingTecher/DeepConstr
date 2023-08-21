import hydra
from omegaconf import DictConfig

from neuri.backends import BackendFactory
from neuri.materialize import Model
from neuri.narrow_spec import auto_opconfig


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    backend_cfg = cfg["backend"]
    if backend_cfg["type"] is not None:
        factory = BackendFactory.init(
            name=backend_cfg["type"],
            target=backend_cfg["target"],
            optmax=backend_cfg["optmax"],
        )
    else:
        factory = None
    model_type = Model.init(cfg["model"]["type"], backend_target=backend_cfg["target"])
    auto_opconfig(model_type, factory)


if __name__ == "__main__":
    main()
