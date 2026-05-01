from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .agent import Agent
from .board import Board, GameResult, IllegalMoveError, Move, Player


@dataclass
class MatchRecord:
    players: Dict[Player, str]
    moves: List[Tuple[Player, Move]] = field(default_factory=list)
    move_durations: List[float] = field(default_factory=list)
    result: GameResult = GameResult.IN_PROGRESS
    winner: Optional[Player] = None
    forfeited_by: Optional[Player] = None
    forfeit_reason: Optional[str] = None
    final_board: Optional[Board] = None

    def summary(self) -> str:
        x_name = self.players[Player.X]
        o_name = self.players[Player.O]
        header = f"{x_name} (X) vs {o_name} (O)"
        if self.forfeited_by is not None:
            loser = self.players[self.forfeited_by]
            return f"{header} -> {loser} forfeited: {self.forfeit_reason}"
        if self.result is GameResult.DRAW:
            outcome = "Draw"
        elif self.winner is not None:
            outcome = f"{self.players[self.winner]} ({self.winner.symbol()}) wins"
        else:
            outcome = "Unfinished"
        return f"{header} -> {outcome} in {len(self.moves)} moves"


class Referee:
    def __init__(
        self,
        agent_x: Agent,
        agent_o: Agent,
        *,
        verbose: bool = False,
        move_time_limit: Optional[float] = None,
        printer: Callable[[str], None] = print,
    ) -> None:
        self.agents: Dict[Player, Agent] = {Player.X: agent_x, Player.O: agent_o}
        self.verbose = verbose
        self.move_time_limit = move_time_limit
        self._print = printer

    def play_match(self) -> MatchRecord:
        board = Board()
        record = MatchRecord(
            players={
                Player.X: self.agents[Player.X].name,
                Player.O: self.agents[Player.O].name,
            }
        )

        for player, agent in self.agents.items():
            agent.on_game_start(player)

        if self.verbose:
            self._print(
                f"Match: {record.players[Player.X]} (X) vs {record.players[Player.O]} (O)"
            )
            self._print(board.render())

        while not board.is_terminal():
            mover = board.current_player
            agent = self.agents[mover]

            start = time.perf_counter()
            try:
                move = agent.select_move(board.copy(), mover)
            except Exception as exc:  # noqa: BLE001 - any agent error is a forfeit
                return self._finalize_forfeit(board, record, mover, f"agent raised {exc!r}")
            elapsed = time.perf_counter() - start

            if self.move_time_limit is not None and elapsed > self.move_time_limit:
                return self._finalize_forfeit(
                    board,
                    record,
                    mover,
                    f"exceeded time limit ({elapsed:.3f}s > {self.move_time_limit:.3f}s)",
                )

            try:
                board.play(move, player=mover)
            except IllegalMoveError as exc:
                return self._finalize_forfeit(board, record, mover, str(exc))

            record.moves.append((mover, move))
            record.move_durations.append(elapsed)

            if self.verbose:
                self._print(f"\n{mover.symbol()} ({agent.name}) plays {move}  [{elapsed*1000:.1f} ms]")
                self._print(board.render())

        record.result = board.result()
        record.winner = board.winner()
        record.final_board = board

        for player, agent in self.agents.items():
            agent.on_game_end(board, player)

        if self.verbose:
            self._print(f"\nResult: {record.summary()}")

        return record

    def play_series(self, games: int, swap_sides: bool = True) -> List[MatchRecord]:
        if games < 1:
            raise ValueError("games must be >= 1")
        records: List[MatchRecord] = []
        original_x = self.agents[Player.X]
        original_o = self.agents[Player.O]
        for i in range(games):
            if swap_sides and i % 2 == 1:
                self.agents = {Player.X: original_o, Player.O: original_x}
            else:
                self.agents = {Player.X: original_x, Player.O: original_o}
            records.append(self.play_match())
        self.agents = {Player.X: original_x, Player.O: original_o}
        return records

    def _finalize_forfeit(
        self,
        board: Board,
        record: MatchRecord,
        offender: Player,
        reason: str,
    ) -> MatchRecord:
        record.forfeited_by = offender
        record.forfeit_reason = reason
        record.winner = offender.opponent()
        record.result = GameResult.from_winner(record.winner)
        record.final_board = board
        if self.verbose:
            self._print(f"\nForfeit by {offender.symbol()} ({self.agents[offender].name}): {reason}")
        return record
