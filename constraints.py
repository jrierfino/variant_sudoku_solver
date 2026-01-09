from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Cell = Tuple[int, int]
Candidates = Dict[Cell, Set[int]]


def value_of(candidates: Candidates, cell: Cell) -> int:
    """Return assigned value if the cell is fixed to a single digit, else 0."""
    vals = candidates[cell]
    return next(iter(vals)) if len(vals) == 1 else 0


def ensure_non_empty(candidates: Candidates, cell: Cell) -> bool:
    return len(candidates[cell]) > 0


def orthogonal(a: Cell, b: Cell) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


ALL_CELLS: List[Cell] = [(r, c) for r in range(9) for c in range(9)]


class Constraint:
    name: str = "constraint"

    def affected_cells(self) -> Iterable[Cell]:
        raise NotImplementedError

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        """Returns (changed, ok)."""
        raise NotImplementedError

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        """Return True if the fully assigned grid satisfies the constraint."""
        raise NotImplementedError


@dataclass
class AllDifferentConstraint(Constraint):
    cells: Sequence[Cell]
    name: str = "all-different"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        assigned = {
            cell: value_of(candidates, cell)
            for cell in self.cells
            if value_of(candidates, cell)
        }
        seen: Dict[int, Cell] = {}
        for cell, val in assigned.items():
            if val in seen:
                return changed, False
            seen[val] = cell
        for cell in self.cells:
            if cell in assigned:
                continue
            for val in list(candidates[cell]):
                if val in assigned.values():
                    candidates[cell].remove(val)
                    changed = True
                    if not ensure_non_empty(candidates, cell):
                        return changed, False
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        vals = [assignment[cell] for cell in self.cells]
        return len(vals) == len(set(vals))


@dataclass
class ThermometerConstraint(Constraint):
    cells: Sequence[Cell]
    name: str = "thermometer"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        k = len(self.cells)
        if k < 2:
            return changed, True
        forward_min: List[int] = [0] * k
        backward_max: List[int] = [0] * k
        for i, cell in enumerate(self.cells):
            if not candidates[cell]:
                return changed, False
            lower = max(min(candidates[cell]), i + 1)
            if i > 0:
                lower = max(lower, forward_min[i - 1] + 1)
            forward_min[i] = lower
        for i in reversed(range(k)):
            cell = self.cells[i]
            upper = (
                min(candidates[cell])
                if len(candidates[cell]) == 1
                else max(candidates[cell])
            )
            upper = min(upper, 9 - (k - i - 1))
            if i < k - 1:
                upper = min(upper, backward_max[i + 1] - 1)
            backward_max[i] = upper
            if forward_min[i] > backward_max[i]:
                return True, False
        for i, cell in enumerate(self.cells):
            new_vals = {
                v for v in candidates[cell] if forward_min[i] <= v <= backward_max[i]
            }
            if not new_vals:
                return True, False
            if new_vals != candidates[cell]:
                candidates[cell] = new_vals
                changed = True
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        values = [assignment[cell] for cell in self.cells]
        return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def _pair_filter(
    a: Cell, b: Cell, candidates: Candidates, predicate
) -> Tuple[bool, bool]:
    changed = False
    ca = candidates[a]
    cb = candidates[b]
    allowed_a = {va for va in ca if any(predicate(va, vb) for vb in cb)}
    allowed_b = {vb for vb in cb if any(predicate(va, vb) for va in ca)}
    if not allowed_a or not allowed_b:
        return True, False
    if allowed_a != ca:
        candidates[a] = allowed_a
        changed = True
    if allowed_b != cb:
        candidates[b] = allowed_b
        changed = True
    return changed, True


@dataclass
class GermanWhispersConstraint(Constraint):
    cells: Sequence[Cell]
    wrap: bool = False
    name: str = "german-whispers"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        cells = list(self.cells)
        pairs = [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]
        if self.wrap and len(cells) > 2:
            pairs.append((cells[-1], cells[0]))
        for a, b in pairs:
            local_changed, ok = _pair_filter(
                a, b, candidates, lambda x, y: abs(x - y) >= 5
            )
            if not ok:
                return True, False
            changed = changed or local_changed
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        cells = list(self.cells)
        pairs = [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]
        if self.wrap and len(cells) > 2:
            pairs.append((cells[-1], cells[0]))
        return all(abs(assignment[a] - assignment[b]) >= 5 for a, b in pairs)


@dataclass
class RenbanConstraint(Constraint):
    cells: Sequence[Cell]
    wrap: bool = False
    name: str = "renban"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        length = len(self.cells)
        if length == 0:
            return changed, True

        feasible_starts = []
        for start in range(1, 11 - length):
            end = start + length - 1
            if all(
                any(start <= v <= end for v in candidates[cell]) for cell in self.cells
            ):
                feasible_starts.append(start)
        if not feasible_starts:
            return changed, False

        cells_sorted = sorted(self.cells, key=lambda c: len(candidates[c]))
        combined_allowed: Dict[Cell, Set[int]] = {cell: set() for cell in self.cells}
        any_valid = False

        for start in feasible_starts:
            digits = list(range(start, start + length))
            allowed_map: Dict[Cell, Set[int]] = {cell: set() for cell in self.cells}

            def dfs(idx: int, used: Set[int]) -> bool:
                if idx == len(cells_sorted):
                    return True
                cell = cells_sorted[idx]
                found = False
                for val in digits:
                    if val in used or val not in candidates[cell]:
                        continue
                    used.add(val)
                    if dfs(idx + 1, used):
                        allowed_map[cell].add(val)
                        found = True
                    used.remove(val)
                return found

            if dfs(0, set()):
                any_valid = True
                for cell, vals in allowed_map.items():
                    combined_allowed[cell].update(vals)

        if not any_valid:
            return changed, False

        for cell in self.cells:
            new_vals = candidates[cell] & combined_allowed[cell]
            if not new_vals:
                return True, False
            if new_vals != candidates[cell]:
                candidates[cell] = new_vals
                changed = True
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        vals = [assignment[cell] for cell in self.cells]
        return sorted(vals) == list(range(min(vals), min(vals) + len(vals)))


@dataclass
class KropkiDotConstraint(Constraint):
    a: Cell
    b: Cell
    dot_type: str  # "white" or "black"
    name: str = "kropki"

    def affected_cells(self) -> Iterable[Cell]:
        return [self.a, self.b]

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        if self.dot_type == "white":
            return _pair_filter(
                self.a, self.b, candidates, lambda x, y: abs(x - y) == 1
            )
        allowed_pairs = {(1, 2), (2, 1), (2, 4), (4, 2), (3, 6), (6, 3), (4, 8), (8, 4)}
        return _pair_filter(
            self.a, self.b, candidates, lambda x, y: (x, y) in allowed_pairs
        )

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        a_val = assignment[self.a]
        b_val = assignment[self.b]
        if self.dot_type == "white":
            return abs(a_val - b_val) == 1
        return {a_val, b_val} in ({1, 2}, {2, 4}, {3, 6}, {4, 8})


@dataclass
class KillerCageConstraint(Constraint):
    cells: Sequence[Cell]
    cage_sum: int
    name: str = "killer"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        cells_sorted = sorted(self.cells, key=lambda c: len(candidates[c]))
        allowed_values: Dict[Cell, Set[int]] = {cell: set() for cell in self.cells}

        def dfs(
            idx: int, used: Set[int], total: int, assignment: Dict[Cell, int]
        ) -> None:
            if total > self.cage_sum:
                return
            remaining = len(cells_sorted) - idx
            available_digits = [d for d in range(1, 10) if d not in used]
            if remaining > len(available_digits):
                return
            min_rem = sum(sorted(available_digits)[:remaining]) if remaining else 0
            max_rem = (
                sum(sorted(available_digits, reverse=True)[:remaining])
                if remaining
                else 0
            )
            if total + min_rem > self.cage_sum or total + max_rem < self.cage_sum:
                return
            if idx == len(cells_sorted):
                if total == self.cage_sum:
                    for cell, val in assignment.items():
                        allowed_values[cell].add(val)
                return
            cell = cells_sorted[idx]
            for val in candidates[cell]:
                if val in used:
                    continue
                assignment[cell] = val
                dfs(idx + 1, used | {val}, total + val, assignment)
                assignment.pop(cell, None)

        dfs(0, set(), 0, {})
        if any(len(v) == 0 for v in allowed_values.values()):
            return changed, False
        for cell in self.cells:
            new_vals = candidates[cell] & allowed_values[cell]
            if not new_vals:
                return True, False
            if new_vals != candidates[cell]:
                candidates[cell] = new_vals
                changed = True
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        values = [assignment[cell] for cell in self.cells]
        return sum(values) == self.cage_sum and len(values) == len(set(values))


@dataclass
class KillerCageNoSumConstraint(Constraint):
    cells: Sequence[Cell]
    name: str = "killer-nosum"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        return AllDifferentConstraint(self.cells).propagate(candidates)

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        vals = [assignment[cell] for cell in self.cells]
        return len(vals) == len(set(vals))


@dataclass
class PalindromeConstraint(Constraint):
    cells: Sequence[Cell]
    name: str = "palindrome"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        k = len(self.cells)
        for i in range(k // 2):
            a = self.cells[i]
            b = self.cells[k - 1 - i]
            common = candidates[a] & candidates[b]
            if not common:
                return True, False
            if common != candidates[a]:
                candidates[a] = set(common)
                changed = True
            if common != candidates[b]:
                candidates[b] = set(common)
                changed = True
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        k = len(self.cells)
        return all(
            assignment[self.cells[i]] == assignment[self.cells[k - 1 - i]]
            for i in range(k)
        )


@dataclass
class ZipperLineConstraint(Constraint):
    cells: Sequence[Cell]
    name: str = "zipper"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        n = len(self.cells)
        if n == 0:
            return False, True
        center_cell = self.cells[n // 2]
        allowed_center = set(candidates[center_cell])
        half = n // 2
        pairs = [(self.cells[i], self.cells[-i - 1]) for i in range(half)]

        # First, narrow the center to sums achievable by every symmetric pair.
        for a, b in pairs:
            possible_sums = {
                va + vb
                for va in candidates[a]
                for vb in candidates[b]
                if va + vb in allowed_center
            }
            if not possible_sums:
                return True, False
            allowed_center &= possible_sums
            if not allowed_center:
                return True, False

        changed = False
        # Then prune each pair based on the refined center values.
        for a, b in pairs:
            valid_pairs = {
                (va, vb)
                for va in candidates[a]
                for vb in candidates[b]
                if va + vb in allowed_center
            }
            if not valid_pairs:
                return True, False
            allowed_a = {va for va, _ in valid_pairs}
            allowed_b = {vb for _, vb in valid_pairs}
            if allowed_a != candidates[a]:
                candidates[a] = allowed_a
                changed = True
            if allowed_b != candidates[b]:
                candidates[b] = allowed_b
                changed = True

        if allowed_center != candidates[center_cell]:
            candidates[center_cell] = allowed_center
            changed = True

        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        n = len(self.cells)
        if n == 0:
            return True
        center_val = assignment[self.cells[n // 2]]
        for i in range(n // 2):
            if (
                assignment[self.cells[i]] + assignment[self.cells[n - i - 1]]
                != center_val
            ):
                return False
        return True


@dataclass
class DutchWhispersConstraint(Constraint):
    cells: Sequence[Cell]
    wrap: bool = False
    name: str = "dutch-whispers"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        cells = list(self.cells)
        pairs = [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]
        if self.wrap and len(cells) > 2:
            pairs.append((cells[-1], cells[0]))
        for a, b in pairs:
            local_changed, ok = _pair_filter(
                a, b, candidates, lambda x, y: abs(x - y) >= 4
            )
            if not ok:
                return True, False
            changed = changed or local_changed
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        cells = list(self.cells)
        pairs = [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]
        if self.wrap and len(cells) > 2:
            pairs.append((cells[-1], cells[0]))
        return all(abs(assignment[a] - assignment[b]) >= 4 for a, b in pairs)


@dataclass
class XVConstraint(Constraint):
    a: Cell
    b: Cell
    symbol: str  # "X" (sum 10) or "V" (sum 5)
    name: str = "xv"

    def affected_cells(self) -> Iterable[Cell]:
        return [self.a, self.b]

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        target = 10 if self.symbol.upper() == "X" else 5
        return _pair_filter(self.a, self.b, candidates, lambda x, y: x + y == target)

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        target = 10 if self.symbol.upper() == "X" else 5
        return assignment[self.a] + assignment[self.b] == target


@dataclass
class RegionSumLineConstraint(Constraint):
    cells: Sequence[Cell]
    wrap: bool = False
    name: str = "region-sum-line"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def _segments(self) -> List[List[Cell]]:
        if not self.cells:
            return []
        segments: List[List[Cell]] = [[self.cells[0]]]
        for cell in self.cells[1:]:
            prev = segments[-1][-1]
            if prev[0] // 3 == cell[0] // 3 and prev[1] // 3 == cell[1] // 3:
                segments[-1].append(cell)
            else:
                segments.append([cell])
        return segments

    def _segment_sums(self, segment: List[Cell], candidates: Candidates) -> Set[int]:
        possible: Set[int] = set()
        cells_sorted = sorted(segment, key=lambda c: len(candidates[c]))

        def dfs(idx: int, total: int) -> None:
            if idx == len(cells_sorted):
                possible.add(total)
                return
            cell = cells_sorted[idx]
            for val in candidates[cell]:
                dfs(idx + 1, total + val)

        dfs(0, 0)
        return possible

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        segments = self._segments()
        if len(segments) < 2:
            return changed, True
        segment_sums = [self._segment_sums(seg, candidates) for seg in segments]
        if any(len(sums) == 0 for sums in segment_sums):
            return changed, False
        common_sums = set.intersection(*segment_sums)
        if not common_sums:
            return changed, False

        def allowed_values_for_segment(segment: List[Cell]) -> Dict[Cell, Set[int]]:
            allowed: Dict[Cell, Set[int]] = {cell: set() for cell in segment}
            cells_sorted = sorted(segment, key=lambda c: len(candidates[c]))

            def dfs(idx: int, total: int, target: int) -> bool:
                if idx == len(cells_sorted):
                    return total == target
                cell = cells_sorted[idx]
                ok_any = False
                for val in candidates[cell]:
                    if dfs(idx + 1, total + val, target):
                        allowed[cell].add(val)
                        ok_any = True
                return ok_any

            for target in common_sums:
                dfs(0, 0, target)
            return allowed

        per_segment_allowed = [allowed_values_for_segment(seg) for seg in segments]
        for seg_allowed in per_segment_allowed:
            if any(len(vals) == 0 for vals in seg_allowed.values()):
                return changed, False
        for seg_allowed in per_segment_allowed:
            for cell, vals in seg_allowed.items():
                new_vals = candidates[cell] & vals
                if not new_vals:
                    return True, False
                if new_vals != candidates[cell]:
                    candidates[cell] = new_vals
                    changed = True
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        segments = self._segments()
        if len(segments) < 2:
            return True
        sums = [sum(assignment[cell] for cell in seg) for seg in segments]
        return all(s == sums[0] for s in sums)


@dataclass
class BetweenLineConstraint(Constraint):
    cells: Sequence[Cell]
    wrap: bool = False  # wrap is only for rendering
    name: str = "between-line"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        if len(self.cells) < 3:
            return changed, True
        a, b = self.cells[0], self.cells[-1]
        mid_cells = self.cells[1:-1]
        min_end = min(min(candidates[a]), min(candidates[b]))
        max_end = max(max(candidates[a]), max(candidates[b]))
        if min_end == max_end:
            return True, False
        for cell in mid_cells:
            new_vals = {v for v in candidates[cell] if min_end < v < max_end}
            if not new_vals:
                return True, False
            if new_vals != candidates[cell]:
                candidates[cell] = new_vals
                changed = True
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        if len(self.cells) < 3:
            return True
        a, b = assignment[self.cells[0]], assignment[self.cells[-1]]
        low, high = sorted((a, b))
        return all(low < assignment[c] < high for c in self.cells[1:-1])


@dataclass
class SandwichConstraint(Constraint):
    index: int
    is_row: bool
    total: int
    name: str = "sandwich"

    def affected_cells(self) -> Iterable[Cell]:
        if self.is_row:
            return [(self.index, c) for c in range(9)]
        return [(r, self.index) for r in range(9)]

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        cells = list(self.affected_cells())
        assigned = [value_of(candidates, c) for c in cells]
        if all(assigned):
            try:
                pos1 = assigned.index(1)
                pos9 = assigned.index(9)
            except ValueError:
                return True, False
            low, high = sorted((pos1, pos9))
            between_sum = sum(assigned[low + 1 : high])
            return False, between_sum == self.total

        possible_1 = [i for i, c in enumerate(cells) if 1 in candidates[c]]
        possible_9 = [i for i, c in enumerate(cells) if 9 in candidates[c]]
        if not possible_1 or not possible_9:
            return True, False

        endpoint_positions = set()
        feasible = False
        for i in possible_1:
            for j in possible_9:
                if i == j:
                    continue
                start, end = sorted((i, j))
                between = cells[start + 1 : end]
                min_sum = 0
                max_sum = 0
                ok = True
                for cell in between:
                    vals = candidates[cell] - {1, 9}
                    if not vals:
                        ok = False
                        break
                    min_sum += min(vals)
                    max_sum += max(vals)
                if not ok:
                    continue
                if min_sum <= self.total <= max_sum:
                    feasible = True
                    endpoint_positions.add(i)
                    endpoint_positions.add(j)
        if not feasible:
            return True, False

        changed = False
        for idx, cell in enumerate(cells):
            if idx in endpoint_positions:
                continue
            if 1 in candidates[cell] or 9 in candidates[cell]:
                new_vals = {v for v in candidates[cell] if v not in (1, 9)}
                if not new_vals:
                    return True, False
                if new_vals != candidates[cell]:
                    candidates[cell] = new_vals
                    changed = True
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        cells = list(self.affected_cells())
        pos1 = pos9 = None
        for idx, c in enumerate(cells):
            if assignment[c] == 1:
                pos1 = idx
            if assignment[c] == 9:
                pos9 = idx
        if pos1 is None or pos9 is None:
            return True
        if pos1 > pos9:
            pos1, pos9 = pos9, pos1
        between = cells[pos1 + 1 : pos9]
        return sum(assignment[c] for c in between) == self.total


@dataclass
class LittleKillerConstraint(Constraint):
    cells: Sequence[Cell]
    total: int
    direction: str  # "NE", "NW", "SE", "SW"
    name: str = "little-killer"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        min_sum = sum(min(candidates[c]) for c in self.cells)
        max_sum = sum(max(candidates[c]) for c in self.cells)
        if self.total < min_sum or self.total > max_sum:
            return True, False
        return False, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        return sum(assignment[c] for c in self.cells) == self.total


@dataclass
class AntiConsecutiveConstraint(Constraint):
    neighbors: Dict[Cell, List[Cell]]
    name: str = "anti-consecutive"

    def affected_cells(self) -> Iterable[Cell]:
        return ALL_CELLS

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        seen_pairs = set()
        for cell, nbs in self.neighbors.items():
            for nb in nbs:
                if (nb, cell) in seen_pairs:
                    continue
                seen_pairs.add((cell, nb))
                local_changed, ok = _pair_filter(
                    cell, nb, candidates, lambda x, y: abs(x - y) != 1
                )
                if not ok:
                    return True, False
                changed = changed or local_changed
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        for cell, nbs in self.neighbors.items():
            for nb in nbs:
                if cell >= nb:
                    continue
                if abs(assignment[cell] - assignment[nb]) == 1:
                    return False
        return True


@dataclass
class DiagonalAllDifferentConstraint(Constraint):
    cells: Sequence[Cell]
    name: str = "diagonal"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        return AllDifferentConstraint(self.cells).propagate(candidates)

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        return len({assignment[c] for c in self.cells}) == len(self.cells)


@dataclass
class ParityCellConstraint(Constraint):
    cell: Cell
    parity: str  # "odd" or "even"
    name: str = "parity-cell"

    def affected_cells(self) -> Iterable[Cell]:
        return [self.cell]

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        allowed = {
            v
            for v in candidates[self.cell]
            if (v % 2 == 1 if self.parity == "odd" else v % 2 == 0)
        }
        if not allowed:
            return True, False
        changed = allowed != candidates[self.cell]
        candidates[self.cell] = allowed
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        val = assignment[self.cell]
        return val % 2 == (1 if self.parity == "odd" else 0)


@dataclass
class ParityLineConstraint(Constraint):
    cells: Sequence[Cell]
    wrap: bool = False
    name: str = "parity-line"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        cells = list(self.cells)
        pairs = [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]
        if self.wrap and len(cells) > 2:
            pairs.append((cells[-1], cells[0]))

        def parity_ok(x: int, y: int) -> bool:
            return (x + y) % 2 == 1

        for a, b in pairs:
            local_changed, ok = _pair_filter(a, b, candidates, parity_ok)
            if not ok:
                return True, False
            changed = changed or local_changed
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        cells = list(self.cells)
        pairs = [(cells[i], cells[i + 1]) for i in range(len(cells) - 1)]
        if self.wrap and len(cells) > 2:
            pairs.append((cells[-1], cells[0]))
        return all((assignment[a] + assignment[b]) % 2 == 1 for a, b in pairs)


@dataclass
class ArrowConstraint(Constraint):
    bulb: Cell
    path: Sequence[Cell]
    name: str = "arrow"

    def affected_cells(self) -> Iterable[Cell]:
        return [self.bulb, *self.path]

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        arrow_cells = list(self.path)
        if not arrow_cells:
            return changed, False
        cells_sorted = sorted(arrow_cells, key=lambda c: len(candidates[c]))
        allowed_values: Dict[Cell, Set[int]] = {c: set() for c in arrow_cells}
        allowed_bulb: Set[int] = set()

        def dfs(idx: int, total: int) -> None:
            if idx == len(cells_sorted):
                allowed_bulb.add(total)
                return
            remaining_cells = cells_sorted[idx:]
            min_sum = total + sum(min(candidates[c]) for c in remaining_cells)
            max_sum = total + sum(max(candidates[c]) for c in remaining_cells)
            if min_sum > 9 or max_sum < 1:
                return
            cell = cells_sorted[idx]
            for val in candidates[cell]:
                dfs(idx + 1, total + val)
                allowed_values[cell].add(val)

        dfs(0, 0)
        if not allowed_bulb:
            return changed, False
        bulb_allowed = candidates[self.bulb] & allowed_bulb
        if not bulb_allowed:
            return True, False
        if bulb_allowed != candidates[self.bulb]:
            candidates[self.bulb] = bulb_allowed
            changed = True
        for cell in arrow_cells:
            new_vals = candidates[cell] & allowed_values[cell]
            if not new_vals:
                return True, False
            if new_vals != candidates[cell]:
                candidates[cell] = new_vals
                changed = True
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        return assignment[self.bulb] == sum(assignment[c] for c in self.path)


@dataclass
class QuadConstraint(Constraint):
    cells: Sequence[Cell]
    digits: Sequence[int]
    name: str = "quad"

    def affected_cells(self) -> Iterable[Cell]:
        return self.cells

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        required_counts = Counter(self.digits)
        cells = list(self.cells)
        allowed_per_cell: Dict[Cell, Set[int]] = {cell: set() for cell in cells}
        changed = False
        found = False

        def max_remaining_counts(start_idx: int) -> Dict[int, int]:
            remaining = cells[start_idx:]
            counts: Dict[int, int] = {d: 0 for d in required_counts}
            for cell in remaining:
                for d in required_counts:
                    if d in candidates[cell]:
                        counts[d] += 1
            return counts

        assignment: List[int] = []
        current_counts: Dict[int, int] = {}

        def dfs(idx: int) -> None:
            nonlocal found, changed
            if idx == len(cells):
                if all(current_counts.get(d, 0) >= required_counts[d] for d in required_counts):
                    found = True
                    for cell, val in zip(cells, assignment):
                        allowed_per_cell[cell].add(val)
                return
            cell = cells[idx]
            remaining_max = max_remaining_counts(idx + 1)
            for val in candidates[cell]:
                current_counts[val] = current_counts.get(val, 0) + 1
                ok = True
                for d, req in required_counts.items():
                    have = current_counts.get(d, 0)
                    max_possible = have + remaining_max.get(d, 0)
                    if max_possible < req:
                        ok = False
                        break
                if ok:
                    assignment.append(val)
                    dfs(idx + 1)
                    assignment.pop()
                current_counts[val] -= 1
                if current_counts[val] == 0:
                    current_counts.pop(val, None)

        dfs(0)
        if not found:
            return True, False

        for cell in cells:
            new_vals = candidates[cell] & allowed_per_cell[cell]
            if not new_vals:
                return True, False
            if new_vals != candidates[cell]:
                candidates[cell] = new_vals
                changed = True

        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        required_counts = Counter(self.digits)
        values = [assignment[cell] for cell in self.cells]
        value_counts = Counter(values)
        return all(value_counts.get(d, 0) >= count for d, count in required_counts.items())


def knight_neighbors() -> Dict[Cell, List[Cell]]:
    offsets = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
    neighbors: Dict[Cell, List[Cell]] = {cell: [] for cell in ALL_CELLS}
    for r in range(9):
        for c in range(9):
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 9 and 0 <= nc < 9:
                    neighbors[(r, c)].append((nr, nc))
    return neighbors


def king_neighbors() -> Dict[Cell, List[Cell]]:
    neighbors: Dict[Cell, List[Cell]] = {cell: [] for cell in ALL_CELLS}
    for r in range(9):
        for c in range(9):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 9 and 0 <= nc < 9:
                        neighbors[(r, c)].append((nr, nc))
    return neighbors


@dataclass
class AntiKnightConstraint(Constraint):
    neighbors: Dict[Cell, List[Cell]]
    name: str = "anti-knight"

    def affected_cells(self) -> Iterable[Cell]:
        return ALL_CELLS

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        for cell, vals in candidates.items():
            if len(vals) != 1:
                continue
            val = value_of(candidates, cell)
            for nb in self.neighbors[cell]:
                if val in candidates[nb]:
                    if len(candidates[nb]) == 1:
                        return True, False
                    candidates[nb].discard(val)
                    changed = True
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        for cell, nbs in self.neighbors.items():
            for nb in nbs:
                if assignment[cell] == assignment[nb]:
                    return False
        return True


@dataclass
class AntiKingConstraint(Constraint):
    neighbors: Dict[Cell, List[Cell]]
    name: str = "anti-king"

    def affected_cells(self) -> Iterable[Cell]:
        return ALL_CELLS

    def propagate(self, candidates: Candidates) -> Tuple[bool, bool]:
        changed = False
        for cell, vals in candidates.items():
            if len(vals) != 1:
                continue
            val = value_of(candidates, cell)
            for nb in self.neighbors[cell]:
                if val in candidates[nb]:
                    if len(candidates[nb]) == 1:
                        return True, False
                    candidates[nb].discard(val)
                    changed = True
        return changed, True

    def is_satisfied(self, assignment: Dict[Cell, int]) -> bool:
        for cell, nbs in self.neighbors.items():
            for nb in nbs:
                if assignment[cell] == assignment[nb]:
                    return False
        return True


def build_row_constraints() -> List[AllDifferentConstraint]:
    return [AllDifferentConstraint([(r, c) for c in range(9)]) for r in range(9)]


def build_col_constraints() -> List[AllDifferentConstraint]:
    return [AllDifferentConstraint([(r, c) for r in range(9)]) for c in range(9)]


def build_box_constraints() -> List[AllDifferentConstraint]:
    constraints: List[AllDifferentConstraint] = []
    for box_r in range(3):
        for box_c in range(3):
            cells = []
            for dr in range(3):
                for dc in range(3):
                    cells.append((box_r * 3 + dr, box_c * 3 + dc))
            constraints.append(AllDifferentConstraint(cells))
    return constraints
