from __future__ import annotations

import random
from typing import Optional

from .board import Board, Move, Player


class Agent:
    name: str = "Agent"

    def __init__(self, name: Optional[str] = None) -> None:
        if name is not None:
            self.name = name

    def select_move(self, board: Board, player: Player) -> Move:
        raise NotImplementedError

    def on_game_start(self, player: Player) -> None:
        pass

    def on_game_end(self, board: Board, player: Player) -> None:
        pass


class RandomAgent(Agent):
    name = "Random"

    def __init__(self, name: Optional[str] = None, seed: Optional[int] = None) -> None:
        super().__init__(name)
        self._rng = random.Random(seed)

    def select_move(self, board: Board, player: Player) -> Move:
        moves = board.legal_moves()
        if not moves:
            raise RuntimeError("RandomAgent asked to move on a terminal board")
        return self._rng.choice(moves)


class HumanAgent(Agent):
    name = "Human"

    def select_move(self, board: Board, player: Player) -> Move:
        legal = set(board.legal_moves())
        prompt = (
            f"\n{board.render()}\n"
            f"{self.name} ({player.symbol()}) move as 'row col' (0-2): "
        )
        while True:
            raw = input(prompt).strip().replace(",", " ")
            parts = raw.split()
            if len(parts) != 2 or not all(p.lstrip("-").isdigit() for p in parts):
                print("  -> please enter two integers, e.g. '1 2'")
                continue
            move = (int(parts[0]), int(parts[1]))
            if move not in legal:
                print(f"  -> {move} is not a legal move; legal: {sorted(legal)}")
                continue
            return move
