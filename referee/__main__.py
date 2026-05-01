from __future__ import annotations

import argparse
import importlib
import sys
from typing import Optional

from .agent import Agent, RandomAgent
from .referee import Referee


def _load_agent(spec: str, default_name: str) -> Agent:
    """Load an agent from a module spec.

    The spec is a Python import path (e.g. ``agent_mcts`` or
    ``mypkg.mymod:MyAgent``). The target module must expose either an
    ``Agent`` attribute or the explicit class named after ``:``.
    """
    if ":" in spec:
        module_path, attr = spec.split(":", 1)
    else:
        module_path, attr = spec, "Agent"

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise SystemExit(f"Could not import agent module {module_path!r}: {exc}") from exc

    if not hasattr(module, attr):
        raise SystemExit(
            f"Module {module_path!r} does not expose attribute {attr!r}. "
            f"Define an `Agent` class (or use 'module:ClassName')."
        )

    cls = getattr(module, attr)
    if not (isinstance(cls, type) and issubclass(cls, Agent)):
        raise SystemExit(f"{module_path}:{attr} is not a subclass of referee.agent.Agent")

    return cls(name=default_name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m referee",
        description="Pit two tic-tac-toe agents against each other.",
    )
    parser.add_argument("agent_x", nargs="?", default=None,
                        help="Module path of the X agent (e.g. 'agent_mcts').")
    parser.add_argument("agent_o", nargs="?", default=None,
                        help="Module path of the O agent.")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-move board output.")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Optional per-move time limit in seconds.")
    parser.add_argument("--no-swap", action="store_true",
                        help="Do not swap sides between games in a series.")
    return parser


def main(argv: Optional[list] = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.agent_x is None and args.agent_o is None:
        agent_x: Agent = RandomAgent(name="Rand-X", seed=1)
        agent_o: Agent = RandomAgent(name="Rand-O", seed=2)
    else:
        if args.agent_x is None or args.agent_o is None:
            raise SystemExit("Please specify two agent modules (or none for the demo).")
        agent_x = _load_agent(args.agent_x, default_name=f"{args.agent_x}-X")
        agent_o = _load_agent(args.agent_o, default_name=f"{args.agent_o}-O")

    ref = Referee(agent_x, agent_o, verbose=not args.quiet,
                  move_time_limit=args.time_limit)

    if args.games <= 1:
        record = ref.play_match()
        print("\n" + record.summary())
        return

    records = ref.play_series(args.games, swap_sides=not args.no_swap)
    print("\n=== Series summary ===")
    wins = {agent_x.name: 0, agent_o.name: 0, "Draw": 0}
    for i, rec in enumerate(records, 1):
        print(f"Game {i}: {rec.summary()}")
        if rec.winner is None:
            wins["Draw"] += 1
        else:
            wins[rec.players[rec.winner]] += 1
    print("Totals:", ", ".join(f"{k}={v}" for k, v in wins.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
