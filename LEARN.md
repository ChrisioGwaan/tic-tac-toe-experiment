# Learning Guide — Tic-Tac-Toe AI Experiment

> A hands-on tour of four classic AI game-playing techniques, from
> naïve random play to optimal minimax search, on a board simple enough
> that you can hold the whole game tree in your head.

---

## Who Is This For?

Students and self-learners who want to understand how AI agents *decide what to do* in a
two-player game. No machine learning background is required. A basic grasp of Python and
recursion is enough to follow along.

---

## What Is This Project?

This repository pits four agents against each other on a 3×3 tic-tac-toe board:

| Agent | Core idea |
|---|---|
| **Random** | Picks a legal move at random — the baseline |
| **Minimax with Alpha-Beta Pruning** | Exhaustively searches the game tree; plays perfectly |
| **Monte Carlo Tree Search (MCTS)** | Samples random playouts to estimate move quality |
| **A\* Heuristic Search** | Treats the game as a shortest-path problem with a hand-crafted heuristic |

A referee module handles rules, move timing, and result recording. A benchmark runner
plays every agent against every other agent (round-robin, 10 games per ordered pair) and
saves timing and memory statistics.

---

## Concepts You Will Learn

### 1 — Game Trees and the Minimax Principle

Every two-player, zero-sum game can be modelled as a tree of states.  
- **Nodes** are board positions.  
- **Edges** are legal moves.  
- **Leaves** are terminal positions (win / draw / loss).

Minimax assigns a score to each leaf and propagates it upward:
the *maximising* player (you) picks the child with the **highest** score;
the *minimising* player (opponent) picks the child with the **lowest** score.

```
              root (MAX)
             /          \
          –1 (MIN)      +1 (MIN)
         /    \          /    \
       –1     0        +1    –1
```

Tic-tac-toe's tree has at most 9! ≈ 362 880 leaf nodes, which is tiny —
Minimax can solve it completely in milliseconds.

**Where to look:** [`agent_minimax/minimax_agent.py`](agent_minimax/minimax_agent.py)

---

### 2 — Alpha-Beta Pruning

Alpha-beta pruning is an optimisation on top of Minimax. It tracks two bounds:

- **α (alpha)** — the best score the maximiser is *already guaranteed*.
- **β (beta)** — the best score the minimiser is *already guaranteed*.

When `α ≥ β` the current branch cannot change the root's decision, so we
**prune** (skip) it entirely. On average, alpha-beta cuts the number of
nodes visited roughly in half — on a well-ordered tree it reduces
$O(b^d)$ to $O(b^{d/2})$, where $b$ is branching factor and $d$ is depth.

```python
# Simplified alpha-beta core (from minimax_agent.py)
if is_max:
    for move in legal:
        board.play(move)
        value = max(value, _minimax(board, alpha, beta, depth + 1))
        board.undo()
        if value >= beta:
            return value          # β-cutoff: minimiser would never allow this
        alpha = max(alpha, value)
```

**Key insight:** move ordering matters — exploring the best moves first
produces more cutoffs and speeds up the search significantly.

---

### 3 — Monte Carlo Tree Search (MCTS)

MCTS does *not* evaluate positions with a hand-crafted heuristic. Instead,
it learns move quality by **random simulation**:

1. **Select** — walk the existing tree using the UCB1 formula, which
   balances exploitation (high win-rate nodes) with exploration (rarely
   visited nodes).
2. **Expand** — add one new child node for an untried move.
3. **Simulate** — play randomly to a terminal state and record the result.
4. **Backpropagate** — update visit counts and win totals all the way up
   to the root.

After many iterations the most-visited child is chosen.

The UCB1 score used to select child nodes is:

$$\text{UCB1} = \frac{w_i}{n_i} + C \sqrt{\frac{\ln N}{n_i}}$$

where $w_i$ = wins at node $i$, $n_i$ = visits at node $i$, $N$ = visits
at the parent, and $C$ is an exploration constant (commonly $\sqrt{2}$).

**Where to look:** [`agent_mcts/mcts_agent.py`](agent_mcts/mcts_agent.py)

---

### 4 — A\* Heuristic Search

A\* is a classic single-agent pathfinding algorithm. Applying it to a
two-player game requires a modelling decision: this implementation treats
the opponent as a fixed **greedy responder** (part of the environment)
rather than an adversary.

Each node on the priority queue is scored by:

$$f = g + h$$

- $g$ = number of *our* moves played so far (path cost).
- $h$ = estimated moves still needed to win, plus a penalty for
  unblocked opponent two-in-a-rows (threat penalty).

Because the opponent model is greedy rather than minimax, A\* is **not
guaranteed to be optimal** against a perfect player — it beats random play
easily but typically draws or loses to MCTS and Minimax.

**Where to look:** [`agent_astar/astar_agent.py`](agent_astar/astar_agent.py)

---

### 5 — Why Tic-Tac-Toe Is "Solved"

Tic-tac-toe is a **solved game**: with perfect play from both sides the
result is always a draw. This means:

- Any two "strong" agents (Minimax, MCTS with enough iterations) will
  always draw each other — exactly what the benchmark shows.
- Weaker agents (Random, A\*) lose because they cannot find the optimal
  defence every time.

This property makes tic-tac-toe an ideal testbed: correctness is
verifiable without domain expertise.

---

### 6 — Benchmarking AI Agents

The benchmark ([`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py)) measures two things
that matter in practice:

| Metric | Why it matters |
|---|---|
| **Wall-clock time** | How fast can the agent make a decision? |
| **Peak memory (MiB)** | How much RAM does the agent's data structure need? |

Key findings from this experiment:

- **Minimax** is the slowest (~1.5 s / 10-game series) because it visits
  every node in the tree with no transposition table.
- **MCTS** uses the most memory (~1–2 MiB / series) because it allocates
  a fresh search tree each turn.
- **A\*** and **Random** are the fastest and most memory-efficient.

> The time and memory trade-offs you see here scale dramatically on
> larger games — this is exactly the motivation for combining search
> with learned value functions, as in AlphaGo/AlphaZero.

---

## Project Structure Walk-through

```
tic-tac-toe-experiment/
│
├── referee/              ← Game engine (rules, board, referee loop)
│   ├── board.py          ← Board state, legal moves, win detection
│   ├── agent.py          ← Agent base class + RandomAgent
│   ├── referee.py        ← Match orchestration, timing, result recording
│   └── __main__.py       ← CLI entry point (python -m referee)
│
├── agent_minimax/        ← Minimax + Alpha-Beta Pruning
├── agent_mcts/           ← Monte Carlo Tree Search
├── agent_astar/          ← A* Heuristic Search
│
└── benchmarks/
    ├── run_benchmark.py  ← Round-robin runner + chart generator
    └── results.json      ← Raw results from the last benchmark run
```

### The Agent Interface

Every agent inherits from `referee.agent.Agent` and implements one method:

```python
def select_move(self, board: Board, player: Player) -> Move:
    ...
```

The referee passes a **copy** of the board and expects a `(row, col)` tuple back.
This clean interface means you can drop in any new agent without touching
the game engine.

---

## Running the Experiments

### Play two agents against each other

```bash
# Minimax (X) vs MCTS (O), 1 game, verbose board output
python -m referee agent_minimax agent_mcts

# 5 games, suppress per-move output
python -m referee agent_minimax agent_mcts --games 5 --quiet
```

### Play as a human

```bash
# You are X, Minimax is O
python -m referee --games 1
# (follow the prompt — enter moves as "row col", e.g. "1 1" for centre)
```

### Reproduce the benchmark

```bash
pip install memory_profiler matplotlib
python -m benchmarks.run_benchmark
```

---

## Exercises and Extensions

These are open-ended ideas to deepen your understanding:

1. **Add a transposition table to Minimax.**  
   Many board positions are reachable via different move orders. Cache
   results by board state and count how many node evaluations you save.

2. **Tune MCTS iterations.**  
   Run the benchmark with `iterations=50`, `200`, `1000`, `5000`. At what
   point does MCTS consistently draw Minimax? Plot the result.

3. **Improve the A\* heuristic.**  
   Replace the greedy opponent model with a one-ply minimax look-ahead.
   Does A\* start drawing against MCTS?

4. **Scale up to 4×4 or 5×5.**  
   Minimax on a 5×5 board is intractable without depth limiting. Add a
   maximum depth and an evaluation function, then re-run the benchmark.

5. **Add iterative deepening.**  
   Combine depth-limited Minimax with iterative deepening (IDDFS) so the
   agent always returns the best move it has found within a time budget.

6. **Implement a neural network agent.**  
   Train a small policy/value network on self-play data from Minimax games,
   then use it inside MCTS (à la AlphaGo). Compare against the pure MCTS
   agent in the benchmark.

---

## Recommended Reading

| Resource | What you will get |
|---|---|
| *Artificial Intelligence: A Modern Approach* — Russell & Norvig, Ch. 5 | The canonical textbook treatment of Minimax and Alpha-Beta |
| [*A Survey of Monte Carlo Tree Search Methods*](http://www.cameronius.com/cv/mcts-survey-master.pdf) — Browne et al. | Comprehensive academic overview of MCTS |
| [Mastering the Game of Go — DeepMind / AlphaGo](https://www.nature.com/articles/nature16961) | The paper that inspired this project |
| [Wikipedia — Alpha-Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) | Illustrated walk-through with pseudocode |
| [Wikipedia — Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) | Covers UCB1 derivation and MCTS phases |

---

## Author

Built by [Chrisio Gwaan](https://github.com/ChrisioGwaan) as part of a university AI study.  
Contributions and questions are welcome — open an issue or a pull request!
