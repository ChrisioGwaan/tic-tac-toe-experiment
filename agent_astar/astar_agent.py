"""A* heuristic search agent for tic-tac-toe.

A* is a single-agent shortest-path algorithm, so to use it on a
two-player game we treat the opponent as part of the environment:
after every one of *our* moves the opponent immediately replies with
their heuristically best move (a fixed responder model). The resulting
state-transition function is deterministic, which is exactly what A*
expects.

State graph
-----------
* **Node**: a board where it is *our* turn (after the opponent's forced
  reply, or the initial root if we move first).
* **Edge cost** ``g``: number of our own moves played so far.
* **Heuristic** ``h``: estimate of additional moves we still need to
  win (>= 1 unless already won) plus a small penalty for unblocked
  opponent two-in-a-rows so defensively sound states are explored first.
* **Goal**: a state where we have already won.

Algorithm
---------
1. For each legal first move, play it (and the modelled opponent reply)
   and push the resulting state onto a priority queue keyed by
   ``f = g + h``.
2. Pop the most promising state. If it's a win, return the first move
   on its path. If it's a draw or loss, drop it.
3. Otherwise expand: try every one of *our* legal moves, append the
   opponent's modelled reply, push to the queue.
4. Stop when the queue is empty or a node budget is exhausted; fall
   back to the root move with the best post-reply heuristic.

Because the opponent model is greedy (not minimax), the agent is not
guaranteed to be optimal — strong opponents like the MCTS or Minimax
agents will usually draw it, but it crushes random play.
"""

from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from referee.agent import Agent
from referee.board import Board, Move, Player

_LINES: Tuple[Tuple[Move, Move, Move], ...] = (
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 2), (1, 2), (2, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 2), (1, 1), (2, 0)),
)


def _line_counts(board: Board, line, who: Player) -> Tuple[int, int]:
    mine = theirs = 0
    opp = who.opponent()
    for cell in line:
        v = board.cell(cell)
        if v is who:
            mine += 1
        elif v is opp:
            theirs += 1
    return mine, theirs


def _heuristic(board: Board, me: Player) -> int:
    """A* heuristic — moves still needed for ``me`` to win, plus threat penalty."""
    if board.winner() is me:
        return 0

    best = 99
    for line in _LINES:
        mine, theirs = _line_counts(board, line, me)
        if theirs == 0:
            best = min(best, 3 - mine)

    threat_penalty = 0
    for line in _LINES:
        mine, theirs = _line_counts(board, line, me.opponent())
        if mine == 0 and theirs == 2:
            threat_penalty += 1

    return max(1, best) + threat_penalty


def _opponent_reply(board: Board, opp: Player) -> Optional[Move]:
    """Greedy opponent: take a win, else block our win, else minimise our heuristic."""
    me = opp.opponent()
    legal = board.legal_moves()
    if not legal:
        return None

    # 1) Take an immediate win.
    for m in legal:
        board.play(m, player=opp)
        won = board.winner() is opp
        board.undo()
        if won:
            return m

    # 2) Block our immediate win.
    for m in legal:
        board.play(m, player=opp)
        we_can_win = False
        for our in board.legal_moves():
            board.play(our, player=me)
            if board.winner() is me:
                we_can_win = True
            board.undo()
            if we_can_win:
                break
        board.undo()
        if not we_can_win:
            # Found a move that prevents our immediate win. Prefer the best
            # such move by heuristic in step 3, so don't return yet —
            # instead keep this as a tentative "safe" move.
            safe_move = m
            break
    else:
        safe_move = None

    # 3) Score remaining moves by minimising our heuristic; heavily penalise
    # any move that lets us win immediately.
    best_score: Optional[int] = None
    best_move: Move = safe_move if safe_move is not None else legal[0]
    for m in legal:
        board.play(m, player=opp)
        penalty = 0
        for our in board.legal_moves():
            board.play(our, player=me)
            if board.winner() is me:
                penalty = 100
                board.undo()
                break
            board.undo()
        score = _heuristic(board, me) + penalty
        board.undo()
        if best_score is None or score < best_score:
            best_score = score
            best_move = m
    return best_move


def _apply_with_reply(board: Board, my_move: Move, me: Player) -> Board:
    nb = board.copy()
    nb.play(my_move, player=me)
    if nb.is_terminal():
        return nb
    reply = _opponent_reply(nb, me.opponent())
    if reply is not None:
        nb.play(reply, player=me.opponent())
    return nb


def _state_key(board: Board) -> Tuple:
    return tuple(c.value for row in board.grid for c in row) + (board.current_player.value,)


@dataclass(order=True)
class _Entry:
    f: int
    tie: int
    g: int = field(compare=False)
    board: Board = field(compare=False)
    first_move: Move = field(compare=False)


class AStarAgent(Agent):
    name = "A*-Heuristic"

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        node_budget: int = 50_000,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name)
        self.node_budget = node_budget
        self._rng = random.Random(seed)

    def select_move(self, board: Board, player: Player) -> Move:
        legal = board.legal_moves()
        if not legal:
            raise RuntimeError("AStarAgent asked to move on a terminal board")
        if len(legal) == 1:
            return legal[0]

        frontier: List[_Entry] = []
        counter = itertools.count()
        best_f_for: Dict[Tuple, int] = {}

        fallback_score: Optional[int] = None
        fallback_move: Move = legal[0]

        for m in legal:
            nb = _apply_with_reply(board, m, player)
            if nb.winner() is player:
                return m  # immediate forced win
            if nb.winner() is player.opponent():
                continue  # this root move loses against modelled opponent
            g = 1
            h = _heuristic(nb, player)
            heapq.heappush(frontier, _Entry(g + h, next(counter), g, nb, m))

            if fallback_score is None or h < fallback_score:
                fallback_score = h
                fallback_move = m

        # If every root move loses against the model, fall back to whichever
        # delays the loss / keeps options open (heuristic on raw post-our-move).
        if not frontier:
            best_h: Optional[int] = None
            for m in legal:
                child = board.copy()
                child.play(m, player=player)
                h = _heuristic(child, player)
                if best_h is None or h < best_h:
                    best_h = h
                    fallback_move = m
            return fallback_move

        nodes = 0
        while frontier and nodes < self.node_budget:
            entry = heapq.heappop(frontier)
            nodes += 1
            key = _state_key(entry.board)
            prev = best_f_for.get(key)
            if prev is not None and prev <= entry.f:
                continue
            best_f_for[key] = entry.f

            if entry.board.winner() is player:
                return entry.first_move

            if entry.board.is_terminal():
                continue

            assert entry.board.current_player is player
            for next_move in entry.board.legal_moves():
                child = _apply_with_reply(entry.board, next_move, player)
                if child.winner() is player.opponent():
                    continue
                g = entry.g + 1
                h = _heuristic(child, player)
                heapq.heappush(frontier, _Entry(g + h, next(counter), g, child, entry.first_move))

        return fallback_move
