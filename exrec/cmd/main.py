import argparse
import json
from pprint import pprint

import dacite
from exrec.cmd.app import provide_app
from exrec.cmd.config import (
    AppConfig,
    ExperimentConfig,
    base_type_hooks,
    provide_type_hooks,
)


def main():
    parser = argparse.ArgumentParser(
        description="An application that performs experiments for exrec systems based on the given configuration."
    )
    parser.add_argument(
        "--config", "-c", help="Please give the path to the configuration file"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        dict_cfg = json.load(f)
    pprint(dict_cfg)

    ex_cfg = dacite.from_dict(
        data_class=ExperimentConfig,
        data=dict_cfg["experiment_cfg"],
        config=dacite.Config(type_hooks=base_type_hooks),
    )
    pprint(f"ExperimentType: {ex_cfg}")

    type_hooks = provide_type_hooks(ex_cfg=ex_cfg)

    app_cfg = dacite.from_dict(
        data_class=AppConfig, data=dict_cfg, config=dacite.Config(type_hooks=type_hooks)
    )
    pprint(f"Configuration: {app_cfg}")

    app = provide_app(config=app_cfg)
    app.start()


if __name__ == "__main__":
    main()
