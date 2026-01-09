from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from constraints import (
    ALL_CELLS,
    AntiKingConstraint,
    AntiKnightConstraint,
    AntiConsecutiveConstraint,
    DiagonalAllDifferentConstraint,
    Candidates,
    Constraint,
    build_box_constraints,
    build_col_constraints,
    build_row_constraints,
    value_of,
)

Grid = List[List[int]]
Cell = Tuple[int, int]


@dataclass
class SolverResult:
    status: str
    solution: Optional[Grid]
    duration_ms: int
    solutions_found: int = 0
    message: str = ""


class SudokuSolver:
    def __init__(
        self,
        givens: Grid,
        variant_constraints: Sequence[Constraint],
        anti_knight: bool,
        anti_king: bool,
        anti_consecutive: bool = False,
        diagonal_main: bool = False,
        diagonal_anti: bool = False,
    ) -> None:
        self.givens = givens
        self.variant_constraints = list(variant_constraints)
        self.anti_knight = anti_knight
        self.anti_king = anti_king
        self.anti_consecutive = anti_consecutive
        self.diagonal_main = diagonal_main
        self.diagonal_anti = diagonal_anti

    def _base_constraints(self) -> List[Constraint]:
        constraints: List[Constraint] = []
        constraints.extend(build_row_constraints())
        constraints.extend(build_col_constraints())
        constraints.extend(build_box_constraints())
        constraints.extend(self.variant_constraints)
        if self.anti_knight:
            constraints.append(AntiKnightConstraint(neighbors=self._knight_neighbors))
        if self.anti_king:
            constraints.append(AntiKingConstraint(neighbors=self._king_neighbors))
        if self.anti_consecutive:
            constraints.append(
                AntiConsecutiveConstraint(neighbors=self._orth_neighbors)
            )
        if self.diagonal_main:
            constraints.append(
                DiagonalAllDifferentConstraint([(i, i) for i in range(9)])
            )
        if self.diagonal_anti:
            constraints.append(
                DiagonalAllDifferentConstraint([(i, 8 - i) for i in range(9)])
            )
        return constraints

    @property
    def _knight_neighbors(self):
        from constraints import knight_neighbors

        return knight_neighbors()

    @property
    def _orth_neighbors(self):
        neighbors = {}
        for r in range(9):
            for c in range(9):
                nbs = []
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < 9 and 0 <= nc < 9:
                        nbs.append((nr, nc))
                neighbors[(r, c)] = nbs
        return neighbors

    @property
    def _king_neighbors(self):
        from constraints import king_neighbors

        return king_neighbors()

    def _initial_candidates(
        self, constraints: Sequence[Constraint]
    ) -> Optional[Candidates]:
        candidates: Candidates = {
            (r, c): set(range(1, 10)) for r in range(9) for c in range(9)
        }
        for r in range(9):
            for c in range(9):
                val = self.givens[r][c]
                if val:
                    candidates[(r, c)] = {val}
        ok = self._propagate(candidates, list(constraints))
        return candidates if ok else None

    def _propagate(
        self, candidates: Candidates, constraints: Optional[List[Constraint]] = None
    ) -> bool:
        constraint_list = constraints or self._base_constraints()
        while True:
            changed_any = False
            for constraint in constraint_list:
                changed, ok = constraint.propagate(candidates)
                if not ok:
                    return False
                changed_any = changed_any or changed
            if not changed_any:
                break
        for cell, vals in candidates.items():
            if not vals:
                return False
        return True

    def propagation_step(
        self, candidates: Candidates, constraints: List[Constraint]
    ) -> tuple[bool, dict, bool]:
        before = {cell: set(vals) for cell, vals in candidates.items()}
        ok = self._propagate(candidates, constraints)
        if not ok:
            return False, {}, False
        deltas = {}
        for cell, prev in before.items():
            removed = prev - candidates[cell]
            if removed:
                deltas[cell] = removed
        changed = bool(deltas)
        return changed, deltas, True

    def _is_complete(self, candidates: Candidates) -> bool:
        return all(len(vals) == 1 for vals in candidates.values())

    def _search(
        self,
        candidates: Candidates,
        constraints: List[Constraint],
        require_uniqueness: bool,
        max_solutions: int,
        solutions: List[Grid],
        start_time: float,
        last_report: List[float],
        logger=None,
    ) -> None:
        now = time.time()
        if now - last_report[0] >= 60:
            filled = sum(1 for v in candidates.values() if len(v) == 1)
            print(
                f"[solver] {int(now - start_time)}s elapsed; filled {filled}/81 cells; solutions found {len(solutions)}"
            )
            last_report[0] = now
        if self._is_complete(candidates):
            solution_grid = [
                [value_of(candidates, (r, c)) for c in range(9)] for r in range(9)
            ]
            solutions.append(solution_grid)
            return
        cell = min(
            (cell for cell in candidates if len(candidates[cell]) > 1),
            key=lambda c: len(candidates[c]),
        )
        for val in sorted(candidates[cell]):
            new_cands = {k: set(v) for k, v in candidates.items()}
            new_cands[cell] = {val}
            if not self._propagate(new_cands, constraints):
                if logger:
                    logger(f"Backtrack: r{cell[0]+1}c{cell[1]+1} != {val}")
                continue
            if logger:
                logger(f"Guess: r{cell[0]+1}c{cell[1]+1} = {val}")
            self._search(
                new_cands,
                constraints,
                require_uniqueness,
                max_solutions,
                solutions,
                start_time,
                last_report,
                logger,
            )
            if require_uniqueness and len(solutions) >= max_solutions:
                return

    def solve(self, require_uniqueness: bool = False) -> SolverResult:
        start = time.time()
        constraints = self._base_constraints()
        candidates = self._initial_candidates(constraints)
        if candidates is None:
            duration_ms = int((time.time() - start) * 1000)
            return SolverResult(
                status="no-solution",
                solution=None,
                duration_ms=duration_ms,
                message="Contradiction in givens or constraints.",
            )
        solutions: List[Grid] = []
        max_solutions = 2 if require_uniqueness else 1
        last_report = [start]
        print("[solver] solve start")
        self._search(
            candidates,
            constraints,
            require_uniqueness,
            max_solutions,
            solutions,
            start,
            last_report,
        )
        duration_ms = int((time.time() - start) * 1000)
        print(
            f"[solver] solve end in {duration_ms} ms; solutions found {len(solutions)}"
        )
        if not solutions:
            return SolverResult(
                status="no-solution",
                solution=None,
                duration_ms=duration_ms,
                solutions_found=0,
                message="No solution found.",
            )
        if require_uniqueness and len(solutions) > 1:
            return SolverResult(
                status="multiple",
                solution=solutions[0],
                duration_ms=duration_ms,
                solutions_found=len(solutions),
                message="Multiple solutions exist.",
            )
        return SolverResult(
            status="solved",
            solution=solutions[0],
            duration_ms=duration_ms,
            solutions_found=len(solutions),
            message="Solved successfully.",
        )
