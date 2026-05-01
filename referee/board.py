from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, List, Optional, Tuple

Move = Tuple[int, int]

BOARD_SIZE = 3


class Player(Enum):
    EMPTY = 0
    X = 1
    O = 2

    def opponent(self) -> "Player":
        if self is Player.X:
            return Player.O
        if self is Player.O:
            return Player.X
        raise ValueError("EMPTY has no opponent")

    def symbol(self) -> str:
        return {Player.EMPTY: ".", Player.X: "X", Player.O: "O"}[self]


class GameResult(Enum):
    IN_PROGRESS = "in_progress"
    X_WINS = "x_wins"
    O_WINS = "o_wins"
    DRAW = "draw"

    @classmethod
    def from_winner(cls, winner: Optional[Player]) -> "GameResult":
        if winner is Player.X:
            return cls.X_WINS
        if winner is Player.O:
            return cls.O_WINS
        return cls.DRAW


_WINNING_LINES: Tuple[Tuple[Move, Move, Move], ...] = (
    # Rows
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    # Columns
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 2), (1, 2), (2, 2)),
    # Diagonals
    ((0, 0), (1, 1), (2, 2)),
    ((0, 2), (1, 1), (2, 0)),
)


class IllegalMoveError(Exception):
    pass


@dataclass
class Board:
    grid: List[List[Player]] = field(
        default_factory=lambda: [
            [Player.EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
        ]
    )
    current_player: Player = Player.X
    history: List[Tuple[Player, Move]] = field(default_factory=list)

    @classmethod
    def from_rows(cls, rows: List[List[Player]], current_player: Player = Player.X) -> "Board":
        if len(rows) != BOARD_SIZE or any(len(r) != BOARD_SIZE for r in rows):
            raise ValueError(f"Board must be {BOARD_SIZE}x{BOARD_SIZE}")
        return cls(grid=[list(r) for r in rows], current_player=current_player)

    def copy(self) -> "Board":
        return Board(
            grid=[row[:] for row in self.grid],
            current_player=self.current_player,
            history=list(self.history),
        )

    def cell(self, move: Move) -> Player:
        r, c = move
        return self.grid[r][c]

    def is_empty(self, move: Move) -> bool:
        return self.cell(move) is Player.EMPTY

    @staticmethod
    def in_bounds(move: Move) -> bool:
        r, c = move
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def legal_moves(self) -> List[Move]:
        return [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if self.grid[r][c] is Player.EMPTY
        ]

    def __iter__(self) -> Iterator[Tuple[Move, Player]]:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                yield (r, c), self.grid[r][c]

    def validate_move(self, move: Move, player: Optional[Player] = None) -> None:
        if self.result() is not GameResult.IN_PROGRESS:
            raise IllegalMoveError("Game is already over")
        if not isinstance(move, tuple) or len(move) != 2:
            raise IllegalMoveError(f"Move must be a (row, col) tuple, got {move!r}")
        if not self.in_bounds(move):
            raise IllegalMoveError(f"Move {move} is out of bounds")
        if not self.is_empty(move):
            raise IllegalMoveError(f"Cell {move} is already occupied by {self.cell(move).symbol()}")
        if player is not None and player is not self.current_player:
            raise IllegalMoveError(
                f"It is {self.current_player.symbol()}'s turn, not {player.symbol()}'s"
            )

    def play(self, move: Move, player: Optional[Player] = None) -> GameResult:
        self.validate_move(move, player)
        mover = self.current_player
        r, c = move
        self.grid[r][c] = mover
        self.history.append((mover, move))
        self.current_player = mover.opponent()
        return self.result()

    def undo(self) -> None:
        if not self.history:
            raise IndexError("No moves to undo")
        mover, (r, c) = self.history.pop()
        self.grid[r][c] = Player.EMPTY
        self.current_player = mover
        
    def winner(self) -> Optional[Player]:
        for line in _WINNING_LINES:
            (r1, c1), (r2, c2), (r3, c3) = line
            a = self.grid[r1][c1]
            if a is Player.EMPTY:
                continue
            if a is self.grid[r2][c2] is self.grid[r3][c3]:
                return a
        return None

    def is_full(self) -> bool:
        return all(cell is not Player.EMPTY for row in self.grid for cell in row)

    def result(self) -> GameResult:
        w = self.winner()
        if w is not None:
            return GameResult.from_winner(w)
        if self.is_full():
            return GameResult.DRAW
        return GameResult.IN_PROGRESS

    def is_terminal(self) -> bool:
        return self.result() is not GameResult.IN_PROGRESS

    def render(self) -> str:
        sep = "\n---+---+---\n"
        rows = [
            " " + " | ".join(cell.symbol() for cell in row) + " "
            for row in self.grid
        ]
        return sep.join(rows)

    def __str__(self) -> str:
        return self.render()
