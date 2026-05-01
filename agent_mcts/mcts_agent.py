"""Monte Carlo Tree Search agent for tic-tac-toe.

The agent is deterministic given a seed, and is strong enough to play
tic-tac-toe perfectly with a modest number of iterations, so two MCTS
agents facing each other should always draw with sufficient simulations.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from referee.agent import Agent
from referee.board import Board, GameResult, Move, Player


@dataclass
class _Node:
    parent: Optional["_Node"]
    move: Optional[Move]                # move that led to this node (from parent)
    player_to_move: Player              # whose turn it is at this node
    untried_moves: List[Move]
    children: Dict[Move, "_Node"] = field(default_factory=dict)
    visits: int = 0
    wins: float = 0.0                   # accumulated reward from the perspective of the player who just moved into this node

    def is_fully_expanded(self) -> bool:
        return not self.untried_moves

    def is_terminal_leaf(self) -> bool:
        # Terminal in the game sense: no legal moves AND no children expanded
        # children come from legal moves, so if there were any they'd be tried.
        return not self.untried_moves and not self.children

    def best_child(self, exploration: float) -> "_Node":
        log_parent = math.log(self.visits) if self.visits > 0 else 0.0

        def ucb(child: "_Node") -> float:
            if child.visits == 0:
                return math.inf
            exploit = child.wins / child.visits
            explore = exploration * math.sqrt(log_parent / child.visits)
            return exploit + explore

        return max(self.children.values(), key=ucb)


class MCTSAgent(Agent):
    name = "MCTS"

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        iterations: int = 2000,
        exploration: float = math.sqrt(2),
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name)
        self.iterations = iterations
        self.exploration = exploration
        self._rng = random.Random(seed)

    # ---- referee protocol --------------------------------------------------

    def select_move(self, board: Board, player: Player) -> Move:
        legal = board.legal_moves()
        if not legal:
            raise RuntimeError("MCTSAgent asked to move on a terminal board")
        if len(legal) == 1:
            return legal[0]

        root = _Node(
            parent=None,
            move=None,
            player_to_move=player,
            untried_moves=list(legal),
        )

        for _ in range(self.iterations):
            node, sim_board = self._select(root, board.copy())
            node, sim_board = self._expand(node, sim_board)
            reward_for_x = self._simulate(sim_board)
            self._backpropagate(node, reward_for_x)

        # Pick the move of the most-visited child (robust child).
        best_move, _ = max(
            root.children.items(),
            key=lambda item: (item[1].visits, item[1].wins / max(item[1].visits, 1)),
        )
        return best_move

    # ---- MCTS phases -------------------------------------------------------

    def _select(self, node: _Node, sim_board: Board) -> "tuple[_Node, Board]":
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self.exploration)
            assert node.move is not None
            sim_board.play(node.move)
        return node, sim_board

    def _expand(self, node: _Node, sim_board: Board) -> "tuple[_Node, Board]":
        if not node.untried_moves or sim_board.is_terminal():
            return node, sim_board
        # Pick an untried move (random for diversity).
        idx = self._rng.randrange(len(node.untried_moves))
        move = node.untried_moves.pop(idx)
        mover = sim_board.current_player
        sim_board.play(move)
        child = _Node(
            parent=node,
            move=move,
            player_to_move=mover.opponent(),
            untried_moves=sim_board.legal_moves() if not sim_board.is_terminal() else [],
        )
        node.children[move] = child
        return child, sim_board

    def _simulate(self, sim_board: Board) -> float:
        """Random playout. Returns reward from X's perspective: 1 win, 0.5 draw, 0 loss."""
        while not sim_board.is_terminal():
            moves = sim_board.legal_moves()
            sim_board.play(self._rng.choice(moves))
        result = sim_board.result()
        if result is GameResult.X_WINS:
            return 1.0
        if result is GameResult.O_WINS:
            return 0.0
        return 0.5

    def _backpropagate(self, node: Optional[_Node], reward_for_x: float) -> None:
        # `node.wins` is from the perspective of the player who *just moved*
        # to reach `node`. That player is the opponent of `node.player_to_move`.
        while node is not None:
            visiting_player = node.player_to_move.opponent() if node.parent is not None else None
            node.visits += 1
            if visiting_player is Player.X:
                node.wins += reward_for_x
            elif visiting_player is Player.O:
                node.wins += 1.0 - reward_for_x
            # Root node has no "player who moved into it"; we still bump visits.
            node = node.parent
