"""Round-robin benchmark of all tic-tac-toe agents.

For every ordered pair of agents (X vs O) we play ``GAMES`` games and
record:

* win / draw / loss counts (from X's perspective)
* total wall-clock time spent thinking
* peak memory usage during the match (via :mod:`memory_profiler`)

Results are written to ``benchmarks/results.json`` and rendered as a
bar chart at ``benchmarks/benchmark.png``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402
from memory_profiler import memory_usage  # noqa: E402

from agent_astar import AStarAgent  # noqa: E402
from agent_mcts import MCTSAgent  # noqa: E402
from agent_minimax import MinimaxAgent  # noqa: E402
from referee import RandomAgent, Referee  # noqa: E402
from referee.agent import Agent  # noqa: E402

GAMES = 10
OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"
PLOT_PATH = OUT_DIR / "benchmark.png"


# Each entry is (label, factory). Factories are called fresh for every
# match so per-agent state (RNG, MCTS tree) doesn't leak between games.
AGENTS: List[Tuple[str, Callable[[], Agent]]] = [
    ("Random", lambda: RandomAgent(name="Random", seed=42)),
    ("MCTS", lambda: MCTSAgent(name="MCTS", iterations=400, seed=42)),
    ("Minimax", lambda: MinimaxAgent(name="Minimax", seed=42)),
    ("A*", lambda: AStarAgent(name="A*", seed=42)),
]


@dataclass
class PairResult:
    x: str
    o: str
    games: int
    x_wins: int
    o_wins: int
    draws: int
    seconds: float
    peak_mib: float


def _play_series(x_factory: Callable[[], Agent], o_factory: Callable[[], Agent]) -> Tuple[int, int, int, float]:
    x_wins = o_wins = draws = 0
    start = time.perf_counter()
    for _ in range(GAMES):
        ref = Referee(x_factory(), o_factory(), verbose=False)
        rec = ref.play_match()
        if rec.winner is None:
            draws += 1
        elif rec.winner.name == "X":
            x_wins += 1
        else:
            o_wins += 1
    return x_wins, o_wins, draws, time.perf_counter() - start


def _measure(x_factory: Callable[[], Agent], o_factory: Callable[[], Agent]) -> PairResult:
    holder: Dict[str, Tuple[int, int, int, float]] = {}

    def runner() -> None:
        holder["r"] = _play_series(x_factory, o_factory)

    # interval=0.05s sampling; include the child's own RSS, return max.
    mem_samples = memory_usage((runner, (), {}), interval=0.05, timeout=None, max_iterations=1)
    peak = max(mem_samples) if mem_samples else 0.0
    baseline = min(mem_samples) if mem_samples else 0.0
    x_wins, o_wins, draws, seconds = holder["r"]
    return PairResult(
        x="<set by caller>",
        o="<set by caller>",
        games=GAMES,
        x_wins=x_wins,
        o_wins=o_wins,
        draws=draws,
        seconds=seconds,
        peak_mib=round(peak - baseline, 3),
    )


def run_round_robin() -> List[PairResult]:
    results: List[PairResult] = []
    for x_label, x_factory in AGENTS:
        for o_label, o_factory in AGENTS:
            print(f"  {x_label:<7} (X) vs {o_label:<7} (O) ...", end="", flush=True)
            res = _measure(x_factory, o_factory)
            res.x, res.o = x_label, o_label
            results.append(res)
            print(
                f" X={res.x_wins} D={res.draws} O={res.o_wins}"
                f" | {res.seconds:.2f}s | +{res.peak_mib:.2f} MiB"
            )
    return results


def _make_plot(results: List[PairResult]) -> None:
    labels = [a[0] for a in AGENTS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Time matrix (X-agent rows, O-agent columns) ---
    n = len(labels)
    time_matrix = [[0.0] * n for _ in range(n)]
    mem_matrix = [[0.0] * n for _ in range(n)]
    for r in results:
        i = labels.index(r.x)
        j = labels.index(r.o)
        time_matrix[i][j] = r.seconds
        mem_matrix[i][j] = r.peak_mib

    def heatmap(ax, matrix, title, fmt, cmap):
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(n), labels)
        ax.set_yticks(range(n), labels)
        ax.set_xlabel("O agent")
        ax.set_ylabel("X agent")
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, fmt.format(matrix[i][j]), ha="center", va="center", fontsize=9, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    heatmap(axes[0], time_matrix, f"Wall-clock per series ({GAMES} games), seconds", "{:.2f}", "Blues")
    heatmap(axes[1], mem_matrix, f"Peak Δ memory per series ({GAMES} games), MiB", "{:.2f}", "Oranges")

    fig.suptitle("Tic-Tac-Toe Agents — Round-Robin Benchmark", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved plot to {PLOT_PATH}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Round-robin benchmark: {GAMES} games per ordered pair\n")
    results = run_round_robin()
    RESULTS_PATH.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nSaved raw results to {RESULTS_PATH}")
    _make_plot(results)


if __name__ == "__main__":
    main()
