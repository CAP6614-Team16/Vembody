"""Exact logical label provider for benchmark scoring.

``ExactLabelProvider`` implements the ``LabelProvider`` contract in ``labels.py``. From
the visible board alone (opened numbers, hidden cells, and the public mine total) it
returns every hidden cell that is safe in *all* mine layouts consistent with what is
shown, and every cell that is a mine in all of them. Everything else is undetermined.

- Flags are agent assertions, not facts: a flagged cell is treated exactly like any
  other hidden cell.
- The public mine count is a constraint: interior cells (hidden, touching no number)
  are labeled only when the count forces them.
- The search is exact (backtracking over each independent group of frontier cells),
  bounded by a wall-clock timeout and a node budget. Running out of either, or an
  inconsistent board, raises ``CertificationRequired``; the provider never returns a
  guessed label.

This is a runtime labeler. It is not an instance certificate: certification must still
verify template selection rules and guess-free completion (see benchmark-handoff.md).
"""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from .labels import CertificationRequired, LogicalLabels, VisibleBoard
from .spec import Cell

PROVIDER_ID = "vembody-exact-csp/1"

# A clue: (frontier cells around an opened number, the number)
Constraint = tuple[tuple[Cell, ...], int]


def _neighbors(board: VisibleBoard, cell: Cell) -> Iterable[Cell]:
    row, col = cell
    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            if (r, c) != cell and 0 <= r < board.rows and 0 <= c < board.cols:
                yield (r, c)


def _constraints(board: VisibleBoard) -> list[Constraint]:
    """One constraint per opened cell that touches a hidden cell."""
    hidden = board.hidden
    constraints = []
    for r in range(board.rows):
        for c in range(board.cols):
            symbol = board.cells[r][c]
            if symbol in ("#", "F"):
                continue
            if symbol == ".":
                number = 0
            elif symbol.isdigit() and 1 <= int(symbol) <= 8:
                number = int(symbol)
            else:
                raise CertificationRequired(f"Unknown visible symbol {symbol!r} at {(r, c)}")
            around = tuple(n for n in _neighbors(board, (r, c)) if n in hidden)
            if number > len(around):
                raise CertificationRequired(f"Clue at {(r, c)} cannot be satisfied")
            if around:
                constraints.append((around, number))
    return constraints


def _components(constraints: list[Constraint]) -> list[tuple[list[Cell], list[Constraint]]]:
    """Split frontier cells into groups that share no clue (union-find)."""
    parent: dict[Cell, Cell] = {}

    def find(cell: Cell) -> Cell:
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    for cells, _ in constraints:
        for cell in cells:
            parent.setdefault(cell, cell)
        for cell in cells[1:]:
            parent[find(cell)] = find(cells[0])

    groups: dict[Cell, tuple[list[Cell], list[Constraint]]] = {}
    for cell in sorted(parent):
        groups.setdefault(find(cell), ([], []))[0].append(cell)
    for constraint in constraints:
        groups[find(constraint[0][0])][1].append(constraint)
    return list(groups.values())


class _Search:
    """Enumerate all assignments of one component, summarized by mine count."""

    def __init__(self, cells: list[Cell], constraints: list[Constraint], budget: _Budget):
        self.budget = budget
        # Order cells so each clue closes as early as possible (better pruning)
        order: list[Cell] = []
        seen: set[Cell] = set()
        for clue_cells, _ in sorted(constraints, key=lambda c: min(c[0])):
            for cell in clue_cells:
                if cell not in seen:
                    seen.add(cell)
                    order.append(cell)
        self.cells = order
        index = {cell: i for i, cell in enumerate(order)}
        self.targets = [number for _, number in constraints]
        self.remaining = [len(clue_cells) for clue_cells, _ in constraints]
        self.mines = [0] * len(constraints)
        self.touching: list[list[int]] = [[] for _ in order]
        for k, (clue_cells, _) in enumerate(constraints):
            for cell in clue_cells:
                self.touching[index[cell]].append(k)
        # mine count -> (bitmask of cells that are a mine in some solution,
        #                bitmask of cells that are safe in some solution)
        self.by_count: dict[int, list[int]] = {}

    def run(self) -> dict[int, list[int]]:
        self._assign(0, 0, 0)
        return self.by_count

    def _assign(self, position: int, mask: int, count: int) -> None:
        self.budget.tick()
        if position == len(self.cells):
            full = (1 << len(self.cells)) - 1
            entry = self.by_count.setdefault(count, [0, 0])
            entry[0] |= mask
            entry[1] |= full & ~mask
            return
        clues = self.touching[position]
        for value in (0, 1):
            ok = True
            for k in clues:
                self.remaining[k] -= 1
                self.mines[k] += value
                if self.mines[k] > self.targets[k] or (
                    self.mines[k] + self.remaining[k] < self.targets[k]
                ):
                    ok = False
            if ok:
                self._assign(position + 1, mask | (value << position), count + value)
            for k in clues:
                self.remaining[k] += 1
                self.mines[k] -= value


class _Budget:
    def __init__(self, timeout_seconds: float, max_nodes: int) -> None:
        self.deadline = perf_counter() + timeout_seconds
        self.nodes_left = max_nodes

    def tick(self) -> None:
        self.nodes_left -= 1
        if self.nodes_left < 0:
            raise CertificationRequired("Logical solver exceeded its search budget")
        if self.nodes_left % 4096 == 0 and perf_counter() > self.deadline:
            raise CertificationRequired("Logical solver timed out")


def _sums(count_sets: list[set[int]], cap: int) -> set[int]:
    """All totals reachable by picking one count from each set, capped at ``cap``."""
    totals = {0}
    for counts in count_sets:
        totals = {t + k for t in totals for k in counts if t + k <= cap}
    return totals


class ExactLabelProvider:
    """Provably safe and provably mined cells from the visible board alone."""

    def __init__(self, timeout_seconds: float = 2.0, max_nodes: int = 2_000_000) -> None:
        if timeout_seconds <= 0 or max_nodes <= 0:
            raise ValueError("Solver limits must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_nodes = max_nodes

    def label(self, board: VisibleBoard) -> LogicalLabels:
        budget = _Budget(self.timeout_seconds, self.max_nodes)
        hidden = board.hidden
        total = board.total_mines
        components = _components(_constraints(board))
        frontier = {cell for cells, _ in components for cell in cells}
        interior = sorted(hidden - frontier)

        searches = []
        for cells, constraints in components:
            search = _Search(cells, constraints, budget)
            by_count = search.run()
            if not by_count:
                raise CertificationRequired("Visible clues are inconsistent")
            searches.append((search.cells, by_count))

        # A combination is feasible when the interior can hold the remaining mines
        low, high = total - len(interior), total
        count_sets = [set(by_count) for _, by_count in searches]
        feasible_totals = [s for s in _sums(count_sets, high) if s >= low]
        if not feasible_totals:
            raise CertificationRequired("Visible clues contradict the mine count")

        safe: set[Cell] = set()
        mines: set[Cell] = set()
        for j, (cells, by_count) in enumerate(searches):
            others = _sums(count_sets[:j] + count_sets[j + 1 :], high)
            ever_mine = ever_safe = 0
            for k, (mine_mask, safe_mask) in by_count.items():
                if any(low <= k + s <= high for s in others):
                    ever_mine |= mine_mask
                    ever_safe |= safe_mask
            for i, cell in enumerate(cells):
                if not ever_mine >> i & 1:
                    safe.add(cell)
                elif not ever_safe >> i & 1:
                    mines.add(cell)

        if interior:
            leftovers = {total - s for s in feasible_totals}
            if leftovers == {0}:
                safe.update(interior)
            elif leftovers == {len(interior)}:
                mines.update(interior)

        labels = LogicalLabels(frozenset(safe), frozenset(mines), PROVIDER_ID)
        labels.validate(board)
        return labels


LABEL_PROVIDERS = {"exact": ExactLabelProvider}


def build_label_provider(name: str | None) -> ExactLabelProvider | None:
    """Create a label provider by CLI name; ``None`` means no provider."""
    if name is None:
        return None
    if name not in LABEL_PROVIDERS:
        raise ValueError(f"Unknown label provider: {name}")
    return LABEL_PROVIDERS[name]()
