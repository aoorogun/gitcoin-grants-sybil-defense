from __future__ import annotations

import argparse
import json

from .allocation import allocate
from .config import load_config
from .data import generate
from .modeling import train
from .monitoring import monitor
from .scenario import run_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["generate", "train", "allocate", "monitor"]:
        commands.add_parser(name)
    scenario = commands.add_parser("scenario")
    scenario.add_argument("--file", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "generate":
        donors, contributions = generate(config)
        print({"donors": len(donors), "contributions": len(contributions), "simulated": True})
    elif args.command == "train":
        print(json.dumps(train(config), indent=2))
    elif args.command == "allocate":
        print(allocate(config).head(10).to_string(index=False))
    elif args.command == "scenario":
        print(json.dumps(run_scenario(config, args.file), indent=2))
    else:
        print(json.dumps(monitor(config), indent=2))


if __name__ == "__main__":
    main()
