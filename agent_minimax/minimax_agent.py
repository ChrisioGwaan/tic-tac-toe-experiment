"""Minimax search with alpha-beta pruning for tic-tac-toe.

Plays optimally on a 3x3 board: it explores the full game tree (which
is tiny here) and prunes branches that cannot improve the result.
Wins are preferred sooner and losses are deferred (depth-aware scoring),
so the agent will pick the *fastest* win and the *slowest* loss.
"""

from __future__ import annotations

import random
from typing import List, Optional

from referee.agent import Agent
from referee.board import Board, GameResult, Move, Player


# Score bounds chosen so that depth penalties never flip the sign.
_WIN = 10_000
_LOSS = -10_000


class MinimaxAgent(Agent):
    name = "Minimax-AB"

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name)
        # RNG only used to break ties between equally-good moves so games
        # between two Minimax agents aren't completely deterministic.
        self._rng = random.Random(seed)

    # ---- referee protocol --------------------------------------------------

    def select_move(self, board: Board, player: Player) -> Move:
        legal = board.legal_moves()
        if not legal:
            raise RuntimeError("MinimaxAgent asked to move on a terminal board")
        if len(legal) == 1:
            return legal[0]

        best_score = -float("inf")
        best_moves: List[Move] = []

        # Use a full alpha/beta window for each root move so the returned
        # score is exact (not just an upper/lower bound from a cutoff).
        # Tic-tac-toe's tree is tiny, so the lost pruning at the root is
        # negligible and avoids spurious ties between truly different moves.
        for move in legal:
            board.play(move, player=player)
            score = self._minimax(board, player, float("-inf"), float("inf"), depth=1)
            board.undo()

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return self._rng.choice(best_moves)

    # ---- search ------------------------------------------------------------

    def _minimax(
        self,
        board: Board,
        maximizer: Player,
        alpha: float,
        beta: float,
        depth: int,
    ) -> int:
        result = board.result()
        if result is not GameResult.IN_PROGRESS:
            return self._terminal_score(result, maximizer, depth)

        is_max = board.current_player is maximizer
        legal = board.legal_moves()

        if is_max:
            value = -float("inf")
            for move in legal:
                board.play(move)
                value = max(value, self._minimax(board, maximizer, alpha, beta, depth + 1))
                board.undo()
                if value >= beta:
                    return int(value)
                alpha = max(alpha, value)
            return int(value)
        else:
            value = float("inf")
            for move in legal:
                board.play(move)
                value = min(value, self._minimax(board, maximizer, alpha, beta, depth + 1))
                board.undo()
                if value <= alpha:
                    return int(value)
                beta = min(beta, value)
            return int(value)

    @staticmethod
    def _terminal_score(result: GameResult, maximizer: Player, depth: int) -> int:
        if result is GameResult.DRAW:
            return 0
        winner = Player.X if result is GameResult.X_WINS else Player.O
        # Prefer faster wins, slower losses.
        return (_WIN - depth) if winner is maximizer else (_LOSS + depth)
