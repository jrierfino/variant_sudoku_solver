from __future__ import annotations

from typing import List, Sequence

from constraints import (
    Constraint,
)


Grid = List[List[int]]


class PuzzleModel:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.grid: Grid = [[0 for _ in range(9)] for _ in range(9)]
        self.variant_constraints: List[Constraint] = []
        self.anti_knight: bool = False
        self.anti_king: bool = False
        self.anti_consecutive: bool = False
        self.diagonal_main: bool = False
        self.diagonal_anti: bool = False

    def set_value(self, row: int, col: int, value: int) -> None:
        self.grid[row][col] = value

    def clear_value(self, row: int, col: int) -> None:
        self.grid[row][col] = 0

    def add_constraint(self, constraint: Constraint) -> None:
        self.variant_constraints.append(constraint)

    def remove_constraint(self, constraint: Constraint) -> None:
        if constraint in self.variant_constraints:
            self.variant_constraints.remove(constraint)

    def remove_all_constraints(self) -> None:
        self.variant_constraints = []

    def clear_digits(self) -> None:
        for r in range(9):
            for c in range(9):
                self.grid[r][c] = 0

    def copy_grid(self) -> Grid:
        return [[self.grid[r][c] for c in range(9)] for r in range(9)]

    def constraints(self) -> Sequence[Constraint]:
        return list(self.variant_constraints)
