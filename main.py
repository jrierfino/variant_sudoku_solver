from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import json
import io
import os
import sys
from contextlib import contextmanager
from typing import List, Optional, Tuple

from constraints import (
    GermanWhispersConstraint,
    DutchWhispersConstraint,
    KillerCageConstraint,
    KillerCageNoSumConstraint,
    KropkiDotConstraint,
    PalindromeConstraint,
    RenbanConstraint,
    RegionSumLineConstraint,
    BetweenLineConstraint,
    SandwichConstraint,
    LittleKillerConstraint,
    AntiConsecutiveConstraint,
    DiagonalAllDifferentConstraint,
    ThermometerConstraint,
    XVConstraint,
    ParityCellConstraint,
    ParityLineConstraint,
    ArrowConstraint,
    QuadConstraint,
    ZipperLineConstraint,
    orthogonal,
)
from model import PuzzleModel
from solver import SudokuSolver

Cell = Tuple[int, int]


class SudokuApp:
    def __init__(self) -> None:
        self.model = PuzzleModel()
        self.root = tk.Tk()
        self.root.title("Variant Sudoku Solver")
        self.cell_size = 50
        self.margin = 20
        canvas_size = self.margin * 2 + self.cell_size * 9
        self.canvas = tk.Canvas(
            self.root,
            width=canvas_size,
            height=canvas_size,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.root.bind("<Key>", self.on_key_press)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=0, column=1, sticky="ns")

        self.current_tool = tk.StringVar(value="digit")
        self.status_var = tk.StringVar(value="Click a cell to start.")
        self.uniqueness_var = tk.BooleanVar(value=False)
        self.anti_knight_var = tk.BooleanVar(value=self.model.anti_knight)
        self.anti_king_var = tk.BooleanVar(value=self.model.anti_king)
        self.anti_consecutive_var = tk.BooleanVar(value=self.model.anti_consecutive)
        self.diag_main_var = tk.BooleanVar(value=self.model.diagonal_main)
        self.diag_anti_var = tk.BooleanVar(value=self.model.diagonal_anti)
        self.selected_cell: Optional[Cell] = None
        self.current_path: List[Cell] = []
        self.solve_running = False
        self.undo_stack: List = []
        self._last_input_cell: Optional[Cell] = None
        self._control_buttons: List[ttk.Button] = []

        self._build_controls(control_frame)
        self.draw()

    def _on_tool_selected(self, _event=None) -> None:
        label = self.tool_combo.get()
        value = self._tool_label_to_value.get(label)
        if value:
            self.current_tool.set(value)
            self.reset_path()

    def _record_constraint(self, constraint) -> None:
        self.undo_stack.append(lambda c=constraint: self.model.remove_constraint(c))

    def _record_digit_change(self, cell: Cell, prev: int) -> None:
        r, c = cell

        def action():
            if prev:
                self.model.set_value(r, c, prev)
            else:
                self.model.clear_value(r, c)

        self.undo_stack.append(action)

    def undo_last(self) -> None:
        if not self.undo_stack:
            self.status_var.set("Nothing to undo.")
            return
        action = self.undo_stack.pop()
        action()
        self.status_var.set("Undid last action.")
        self.draw()

    def _serialize_constraint(self, constraint):
        from constraints import (
            ThermometerConstraint,
            ArrowConstraint,
            GermanWhispersConstraint,
            DutchWhispersConstraint,
            BetweenLineConstraint,
            ParityLineConstraint,
            RenbanConstraint,
            PalindromeConstraint,
            RegionSumLineConstraint,
            ParityCellConstraint,
            KropkiDotConstraint,
            XVConstraint,
            QuadConstraint,
            LittleKillerConstraint,
            SandwichConstraint,
            KillerCageConstraint,
            KillerCageNoSumConstraint,
            ZipperLineConstraint,
        )

        if isinstance(constraint, ThermometerConstraint):
            return {"type": "thermo", "cells": constraint.cells}
        if isinstance(constraint, ArrowConstraint):
            return {"type": "arrow", "bulb": constraint.bulb, "cells": constraint.path}
        if isinstance(constraint, GermanWhispersConstraint):
            return {
                "type": "whispers",
                "cells": constraint.cells,
                "wrap": getattr(constraint, "wrap", False),
            }
        if isinstance(constraint, DutchWhispersConstraint):
            return {
                "type": "dutch",
                "cells": constraint.cells,
                "wrap": getattr(constraint, "wrap", False),
            }
        if isinstance(constraint, BetweenLineConstraint):
            return {
                "type": "between",
                "cells": constraint.cells,
                "wrap": getattr(constraint, "wrap", False),
            }
        if isinstance(constraint, ParityLineConstraint):
            return {
                "type": "parity-line",
                "cells": constraint.cells,
                "wrap": getattr(constraint, "wrap", False),
            }
        if isinstance(constraint, RenbanConstraint):
            return {
                "type": "renban",
                "cells": constraint.cells,
                "wrap": getattr(constraint, "wrap", False),
            }
        if isinstance(constraint, PalindromeConstraint):
            return {"type": "palindrome", "cells": constraint.cells}
        if isinstance(constraint, RegionSumLineConstraint):
            return {
                "type": "region-sum",
                "cells": constraint.cells,
                "wrap": getattr(constraint, "wrap", False),
            }
        if isinstance(constraint, ParityCellConstraint):
            return {
                "type": "parity-cell",
                "cell": constraint.cell,
                "parity": constraint.parity,
            }
        if isinstance(constraint, KropkiDotConstraint):
            return {
                "type": "kropki",
                "a": constraint.a,
                "b": constraint.b,
                "dot_type": constraint.dot_type,
            }
        if isinstance(constraint, XVConstraint):
            return {
                "type": "xv",
                "a": constraint.a,
                "b": constraint.b,
                "symbol": constraint.symbol,
            }
        if isinstance(constraint, QuadConstraint):
            return {
                "type": "quad",
                "cells": constraint.cells,
                "digits": constraint.digits,
            }
        if isinstance(constraint, LittleKillerConstraint):
            return {
                "type": "little-killer",
                "cells": constraint.cells,
                "total": constraint.total,
                "direction": constraint.direction,
            }
        if isinstance(constraint, SandwichConstraint):
            return {
                "type": "sandwich",
                "index": constraint.index,
                "is_row": constraint.is_row,
                "total": constraint.total,
            }
        if isinstance(constraint, KillerCageConstraint):
            return {
                "type": "killer",
                "cells": constraint.cells,
                "sum": constraint.cage_sum,
            }
        if isinstance(constraint, KillerCageNoSumConstraint):
            return {"type": "killer-nosum", "cells": constraint.cells}
        if isinstance(constraint, ZipperLineConstraint):
            return {"type": "zipper", "cells": constraint.cells}
        return None

    def _deserialize_constraint(self, data):
        from constraints import (
            ThermometerConstraint,
            ArrowConstraint,
            GermanWhispersConstraint,
            DutchWhispersConstraint,
            BetweenLineConstraint,
            ParityLineConstraint,
            RenbanConstraint,
            PalindromeConstraint,
            RegionSumLineConstraint,
            ParityCellConstraint,
            KropkiDotConstraint,
            XVConstraint,
            QuadConstraint,
            LittleKillerConstraint,
            SandwichConstraint,
            KillerCageConstraint,
            KillerCageNoSumConstraint,
            ZipperLineConstraint,
        )

        t = data.get("type")

        def cells_list(key="cells"):
            return [tuple(c) for c in data.get(key, [])]

        try:
            if t == "thermo":
                return ThermometerConstraint(cells_list())
            if t == "arrow":
                return ArrowConstraint(tuple(data["bulb"]), cells_list())
            if t == "whispers":
                return GermanWhispersConstraint(
                    cells_list(), wrap=data.get("wrap", False)
                )
            if t == "dutch":
                return DutchWhispersConstraint(
                    cells_list(), wrap=data.get("wrap", False)
                )
            if t == "between":
                return BetweenLineConstraint(cells_list(), wrap=data.get("wrap", False))
            if t == "parity-line":
                return ParityLineConstraint(cells_list(), wrap=data.get("wrap", False))
            if t == "renban":
                return RenbanConstraint(cells_list(), wrap=data.get("wrap", False))
            if t == "palindrome":
                return PalindromeConstraint(cells_list())
            if t == "region-sum":
                return RegionSumLineConstraint(
                    cells_list(), wrap=data.get("wrap", False)
                )
            if t == "parity-cell":
                return ParityCellConstraint(tuple(data["cell"]), data["parity"])
            if t == "kropki":
                return KropkiDotConstraint(
                    tuple(data["a"]), tuple(data["b"]), data["dot_type"]
                )
            if t == "xv":
                return XVConstraint(tuple(data["a"]), tuple(data["b"]), data["symbol"])
            if t == "quad":
                return QuadConstraint(cells_list(), data.get("digits", []))
            if t == "little-killer":
                return LittleKillerConstraint(
                    cells_list(), data["total"], data["direction"]
                )
            if t == "sandwich":
                return SandwichConstraint(
                    index=data["index"], is_row=data["is_row"], total=data["total"]
                )
            if t == "killer":
                return KillerCageConstraint(cells_list(), data["sum"])
            if t == "killer-nosum":
                return KillerCageNoSumConstraint(cells_list())
            if t == "zipper":
                return ZipperLineConstraint(cells_list())
        except Exception:
            return None
        return None

    def _serialize_puzzle(self) -> dict:
        return {
            "version": 1,
            "grid": self.model.copy_grid(),
            "constraints": [
                c
                for c in (
                    self._serialize_constraint(ct) for ct in self.model.constraints()
                )
                if c
            ],
            "anti_knight": self.model.anti_knight,
            "anti_king": self.model.anti_king,
            "anti_consecutive": self.model.anti_consecutive,
            "diag_main": self.model.diagonal_main,
            "diag_anti": self.model.diagonal_anti,
            "uniqueness": self.uniqueness_var.get(),
        }

    def _apply_puzzle(self, data: dict) -> None:
        try:
            grid = data.get("grid")
            if not grid or len(grid) != 9:
                raise ValueError("Invalid grid")
            self.model.reset()
            for r in range(9):
                for c in range(9):
                    self.model.grid[r][c] = int(grid[r][c]) if grid[r][c] else 0
            self.model.remove_all_constraints()
            for cdata in data.get("constraints", []):
                constraint = self._deserialize_constraint(cdata)
                if constraint:
                    self.model.add_constraint(constraint)
            self.model.anti_knight = bool(data.get("anti_knight", False))
            self.model.anti_king = bool(data.get("anti_king", False))
            self.model.anti_consecutive = bool(data.get("anti_consecutive", False))
            self.model.diagonal_main = bool(data.get("diag_main", False))
            self.model.diagonal_anti = bool(data.get("diag_anti", False))
            self.anti_knight_var.set(self.model.anti_knight)
            self.anti_king_var.set(self.model.anti_king)
            self.anti_consecutive_var.set(self.model.anti_consecutive)
            self.diag_main_var.set(self.model.diagonal_main)
            self.diag_anti_var.set(self.model.diagonal_anti)
            self.uniqueness_var.set(bool(data.get("uniqueness", False)))
            self.current_tool.set("digit")
            if hasattr(self, "tool_combo"):
                self.tool_combo.current(0)
            self.undo_stack = []
            self._last_input_cell = None
            self.current_path = []
            self.selected_cell = None
            self.status_var.set("Loaded puzzle.")
            self.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load puzzle: {e}")

    @contextmanager
    def _quiet_stderr(self):
        buf = io.StringIO()
        orig = sys.stderr
        try:
            sys.stderr = buf
            yield
        finally:
            sys.stderr = orig

    def save_puzzle(self) -> None:
        data = self._serialize_puzzle()
        with self._quiet_stderr():
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")],
                title="Save Puzzle",
            )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.status_var.set(f"Saved puzzle to {path}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save puzzle: {e}")

    def load_puzzle(self) -> None:
        with self._quiet_stderr():
            path = filedialog.askopenfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")],
                title="Load Puzzle",
            )
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._apply_puzzle(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load puzzle: {e}")

    def _build_controls(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Tool", font=("Arial", 12, "bold")).grid(
            row=0, column=0, pady=(0, 4), sticky="w"
        )
        self._tool_options = [
            ("Digits", "digit"),
            ("Thermometer", "thermo"),
            ("Arrow", "arrow"),
            ("Between Line", "between"),
            ("German Whispers", "whispers"),
            ("Dutch Whispers", "dutch"),
            ("Parity Line", "parity"),
            ("Renban", "renban"),
            ("Palindrome", "palindrome"),
            ("Region Sum Line", "region-sum"),
            ("Zipper Line", "zipper"),
            ("Grey Circle (odd)", "odd"),
            ("Grey Square (even)", "even"),
            ("Kropki White", "kropki-white"),
            ("Kropki Black", "kropki-black"),
            ("X (sum 10)", "xv-x"),
            ("V (sum 5)", "xv-v"),
            ("Quad Circle", "quad"),
            ("Little Killer", "little-killer"),
            ("Cage (no sum)", "killer-nosum"),
            ("Killer Cage", "killer"),
        ]
        self._tool_label_to_value = {
            label: value for label, value in self._tool_options
        }
        tool_names = [label for label, _ in self._tool_options]
        self.tool_combo = ttk.Combobox(frame, values=tool_names, state="readonly")
        self.tool_combo.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self.tool_combo.bind("<<ComboboxSelected>>", self._on_tool_selected)
        self.tool_combo.current(0)
        self.current_tool.set(self._tool_options[0][1])

        base_row = 2
        finish_btn = ttk.Button(frame, text="Finish Shape", command=self.complete_path)
        finish_btn.grid(row=base_row, column=0, pady=(6, 2), sticky="ew")
        clear_shape_btn = ttk.Button(
            frame, text="Clear Current Shape", command=self.reset_path
        )
        clear_shape_btn.grid(row=base_row + 1, column=0, pady=2, sticky="ew")
        undo_btn = ttk.Button(frame, text="Undo", command=self.undo_last)
        undo_btn.grid(row=base_row + 2, column=0, pady=2, sticky="ew")
        save_btn = ttk.Button(frame, text="Save Puzzle", command=self.save_puzzle)
        save_btn.grid(row=base_row + 3, column=0, pady=2, sticky="ew")
        load_btn = ttk.Button(frame, text="Load Puzzle", command=self.load_puzzle)
        load_btn.grid(row=base_row + 4, column=0, pady=2, sticky="ew")
        self._control_buttons.extend(
            [finish_btn, clear_shape_btn, undo_btn, save_btn, load_btn]
        )

        ttk.Separator(frame, orient="horizontal").grid(
            row=base_row + 5, column=0, sticky="ew", pady=6
        )

        ttk.Checkbutton(
            frame,
            text="Anti-Knight",
            variable=self.anti_knight_var,
            command=self.toggle_anti_knight,
        ).grid(row=base_row + 10, column=0, sticky="w")
        ttk.Checkbutton(
            frame,
            text="Anti-King",
            variable=self.anti_king_var,
            command=self.toggle_anti_king,
        ).grid(row=base_row + 9, column=0, sticky="w")
        ttk.Checkbutton(
            frame,
            text="Anti-Consecutive",
            variable=self.anti_consecutive_var,
            command=self.toggle_anti_consecutive,
        ).grid(row=base_row + 6, column=0, sticky="w")
        ttk.Checkbutton(
            frame,
            text="Diagonal (↖↘)",
            variable=self.diag_main_var,
            command=self.toggle_diag_main,
        ).grid(row=base_row + 7, column=0, sticky="w")
        ttk.Checkbutton(
            frame,
            text="Diagonal (↙↗)",
            variable=self.diag_anti_var,
            command=self.toggle_diag_anti,
        ).grid(row=base_row + 8, column=0, sticky="w")

        ttk.Checkbutton(
            frame, text="Check Uniqueness", variable=self.uniqueness_var
        ).grid(row=base_row + 11, column=0, sticky="w", pady=(4, 0))

        ttk.Button(frame, text="Add Sandwich", command=self.add_sandwich_clue).grid(
            row=base_row + 12, column=0, pady=(6, 2), sticky="ew"
        )

        solve_btn = ttk.Button(frame, text="Solve", command=self.solve)
        solve_btn.grid(row=base_row + 13, column=0, pady=(8, 2), sticky="ew")
        clear_digits_btn = ttk.Button(
            frame, text="Clear Digits", command=self.clear_digits
        )
        clear_digits_btn.grid(row=base_row + 16, column=0, pady=2, sticky="ew")
        clear_constraints_btn = ttk.Button(
            frame, text="Clear Constraints", command=self.clear_constraints
        )
        clear_constraints_btn.grid(row=base_row + 15, column=0, pady=2, sticky="ew")
        new_puzzle_btn = ttk.Button(frame, text="New Puzzle", command=self.new_puzzle)
        new_puzzle_btn.grid(row=base_row + 14, column=0, pady=2, sticky="ew")
        self._control_buttons.extend(
            [
                solve_btn,
                clear_digits_btn,
                clear_constraints_btn,
                new_puzzle_btn,
            ]
        )

        ttk.Label(
            frame, textvariable=self.status_var, wraplength=200, justify="left"
        ).grid(row=base_row + 17, column=0, pady=(8, 0), sticky="w")

    def coords_to_cell(self, x: int, y: int) -> Optional[Cell]:
        x -= self.margin
        y -= self.margin
        if x < 0 or y < 0:
            return None
        row, col = y // self.cell_size, x // self.cell_size
        if 0 <= row < 9 and 0 <= col < 9:
            return int(row), int(col)
        return None

    def on_drag(self, event) -> None:
        tool = self.current_tool.get()
        line_tools = {
            "thermo",
            "arrow",
            "between",
            "whispers",
            "dutch",
            "parity",
            "renban",
            "palindrome",
            "region-sum",
            "zipper",
        }
        if tool not in line_tools:
            return
        cell = self.coords_to_cell(event.x, event.y)
        if cell is None:
            return
        if not self.current_path:
            self.current_path = [cell]
        else:
            self._add_to_path(cell, require_chain=True, allow_diagonal=True)
        self.draw()

    def on_click(self, event) -> None:
        cell = self.coords_to_cell(event.x, event.y)
        if cell is None:
            return
        tool = self.current_tool.get()
        if tool == "digit":
            self.selected_cell = cell
            self.reset_path()
            self.status_var.set(
                f"Selected cell r{cell[0]+1} c{cell[1]+1}. Type a digit 1-9."
            )
        elif tool in (
            "thermo",
            "arrow",
            "between",
            "whispers",
            "dutch",
            "parity",
            "renban",
            "palindrome",
            "region-sum",
            "zipper",
        ):
            self._add_to_path(
                cell, require_chain=True, allow_diagonal=True, allow_any_connection=True
            )
        elif tool == "killer":
            self._add_to_path(
                cell,
                require_chain=True,
                allow_diagonal=False,
                allow_any_connection=True,
            )
        elif tool == "killer-nosum":
            self._add_to_path(
                cell,
                require_chain=True,
                allow_diagonal=False,
                allow_any_connection=True,
            )
        elif tool == "little-killer":
            direction = simpledialog.askstring(
                "Little Killer", "Direction (NE, NW, SE, SW):", parent=self.root
            )
            if not direction:
                return
            direction = direction.upper()
            dirs = {"NE": (-1, 1), "NW": (-1, -1), "SE": (1, 1), "SW": (1, -1)}
            if direction not in dirs:
                messagebox.showerror("Invalid", "Direction must be NE, NW, SE, or SW.")
                return
            total = simpledialog.askinteger(
                "Little Killer",
                "Enter clue sum:",
                parent=self.root,
                minvalue=0,
                maxvalue=50,
            )
            if total is None:
                return
            dr, dc = dirs[direction]
            cells = []
            r, c = cell
            while 0 <= r < 9 and 0 <= c < 9:
                cells.append((r, c))
                r += dr
                c += dc
            if len(cells) == 0:
                return
            constraint = LittleKillerConstraint(cells, total, direction)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set(f"Added little killer {direction} sum {total}.")
            self.reset_path()
        elif tool in ("kropki-white", "kropki-black", "xv-x", "xv-v"):
            if not self.current_path:
                self.current_path = [cell]
                self.status_var.set("Select the adjacent cell for the marker.")
            else:
                if not orthogonal(cell, self.current_path[0]):
                    messagebox.showerror(
                        "Invalid", "Markers must connect orthogonally adjacent cells."
                    )
                    return
                if tool in ("kropki-white", "kropki-black"):
                    constraint = KropkiDotConstraint(
                        self.current_path[0],
                        cell,
                        "white" if tool == "kropki-white" else "black",
                    )
                    label = "white dot" if tool == "kropki-white" else "black dot"
                else:
                    symbol = "X" if tool == "xv-x" else "V"
                    constraint = XVConstraint(self.current_path[0], cell, symbol)
                    label = f"{symbol} marker"
                self.model.add_constraint(constraint)
                self._record_constraint(constraint)
                self.status_var.set(f"Added {label}.")
                self.current_path = []
        elif tool in ("odd", "even"):
            parity = "odd" if tool == "odd" else "even"
            constraint = ParityCellConstraint(cell, parity)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set(
                f"Added {'odd' if tool=='odd' else 'even'} parity mark."
            )
        elif tool == "quad":
            self._add_to_path(cell, require_chain=False, allow_diagonal=False)
            if len(self.current_path) == 4:
                rows = {r for r, _ in self.current_path}
                cols = {c for _, c in self.current_path}
                if len(rows) != 2 or len(cols) != 2:
                    messagebox.showerror("Invalid", "Quad must cover a 2x2 block.")
                    self.reset_path()
                else:
                    digits_str = simpledialog.askstring(
                        "Quad digits", "Enter 1-4 digits (1-9):", parent=self.root
                    )
                    if digits_str:
                        digits = [int(ch) for ch in digits_str if ch.isdigit()]
                        if not 1 <= len(digits) <= 4 or any(
                            d < 1 or d > 9 for d in digits
                        ):
                            messagebox.showerror(
                                "Invalid", "Enter 1 to 4 digits between 1 and 9."
                            )
                        else:
                            constraint = QuadConstraint(list(self.current_path), digits)
                            self.model.add_constraint(constraint)
                            self._record_constraint(constraint)
                            self.status_var.set(f"Added quad with digits {digits_str}.")
                    self.reset_path()
        self.draw()

    def _add_to_path(
        self,
        cell: Cell,
        require_chain: bool = False,
        allow_diagonal: bool = False,
        allow_any_connection: bool = False,
    ) -> None:
        def is_adjacent(a: Cell, b: Cell) -> bool:
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            if allow_diagonal:
                return max(dr, dc) == 1 and (dr + dc) != 0
            return dr + dc == 1

        if not self.current_path:
            self.current_path = [cell]
            self._last_input_cell = cell
            self.status_var.set(f"Building shape with {len(self.current_path)} cells.")
            return

        if self.current_path and cell == self._last_input_cell:
            return

        chosen_index = None
        if require_chain:
            if self._last_input_cell and is_adjacent(cell, self._last_input_cell):
                # connect to the most recent input if possible
                for idx in range(len(self.current_path) - 1, -1, -1):
                    if self.current_path[idx] == self._last_input_cell:
                        chosen_index = idx
                        break
            elif allow_any_connection:
                for idx in range(len(self.current_path) - 1, -1, -1):
                    if is_adjacent(cell, self.current_path[idx]):
                        chosen_index = idx
                        break
            if chosen_index is None:
                messagebox.showerror(
                    "Invalid", "Cells must be adjacent for this shape."
                )
                return

        if chosen_index is not None:
            # Keep the chosen anchor in the path right before the new cell to ensure the draw order connects correctly.
            if self.current_path[chosen_index] != cell:
                self.current_path.insert(chosen_index + 1, cell)
        else:
            self.current_path.append(cell)
        self._last_input_cell = cell
        self.status_var.set(f"Building shape with {len(self.current_path)} cells.")

    def on_key_press(self, event) -> None:
        if self.current_tool.get() != "digit" or self.selected_cell is None:
            if event.keysym == "Return":
                self.complete_path()
            return
        row, col = self.selected_cell
        prev = self.model.grid[row][col]
        if event.char and event.char.isdigit() and event.char != "0":
            new_val = int(event.char)
            if prev != new_val:
                self._record_digit_change((row, col), prev)
                self.model.set_value(row, col, new_val)
                self.status_var.set(f"Set r{row+1} c{col+1} to {event.char}.")
        elif event.keysym in ("BackSpace", "Delete", "0"):
            if prev != 0:
                self._record_digit_change((row, col), prev)
                self.model.clear_value(row, col)
                self.status_var.set(f"Cleared r{row+1} c{col+1}.")
        self.draw()

    def complete_path(self) -> None:
        tool = self.current_tool.get()
        path = list(self.current_path)
        is_loop = len(path) > 1 and path[0] == path[-1]
        if is_loop:
            path = path[:-1]
        if (
            tool
            in (
                "thermo",
                "arrow",
                "between",
                "whispers",
                "dutch",
                "parity",
                "renban",
                "palindrome",
                "region-sum",
                "zipper",
                "killer",
            )
            and len(path) < 2
        ):
            messagebox.showerror(
                "Too short", "Add at least two cells before finishing."
            )
            return
        if tool == "thermo":
            constraint = ThermometerConstraint(path)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added thermometer.")
        elif tool == "arrow":
            if is_loop:
                messagebox.showerror("Invalid", "Arrows cannot loop back to the bulb.")
                return
            bulb, arrow_cells = path[0], path[1:]
            constraint = ArrowConstraint(bulb, arrow_cells)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added arrow.")
        elif tool == "whispers":
            constraint = GermanWhispersConstraint(path, wrap=is_loop)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added German Whispers line.")
        elif tool == "dutch":
            constraint = DutchWhispersConstraint(path, wrap=is_loop)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added Dutch Whispers line.")
        elif tool == "between":
            constraint = BetweenLineConstraint(path, wrap=is_loop)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added between line.")
        elif tool == "parity":
            constraint = ParityLineConstraint(path, wrap=is_loop)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added parity line.")
        elif tool == "renban":
            constraint = RenbanConstraint(path, wrap=is_loop)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added Renban line.")
        elif tool == "palindrome":
            constraint = PalindromeConstraint(path)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added palindrome line.")
        elif tool == "region-sum":
            constraint = RegionSumLineConstraint(path, wrap=is_loop)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added region sum line.")
        elif tool == "zipper":
            if is_loop:
                messagebox.showerror("Invalid", "Zipper lines cannot loop.")
                return
            if len(path) < 3 or len(path) % 2 == 0:
                messagebox.showerror(
                    "Invalid",
                    "Zipper lines must have an odd length of at least 3 cells.",
                )
                return
            constraint = ZipperLineConstraint(path)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added zipper line.")
        elif tool == "killer":
            cage_sum = simpledialog.askinteger(
                "Killer cage",
                "Enter cage sum (3-45):",
                parent=self.root,
                minvalue=1,
                maxvalue=45,
            )
            if cage_sum is None:
                return
            constraint = KillerCageConstraint(path, cage_sum)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set(f"Added killer cage sum {cage_sum}.")
        elif tool == "killer-nosum":
            constraint = KillerCageNoSumConstraint(path)
            self.model.add_constraint(constraint)
            self._record_constraint(constraint)
            self.status_var.set("Added no-sum cage.")
        self.reset_path()
        self.draw()

    def reset_path(self) -> None:
        self.current_path = []
        self._last_input_cell = None
        self.draw()

    def toggle_anti_knight(self) -> None:
        self.model.anti_knight = self.anti_knight_var.get()
        self.status_var.set(
            f"Anti-Knight {'enabled' if self.model.anti_knight else 'disabled'}."
        )
        self.draw()

    def toggle_anti_king(self) -> None:
        self.model.anti_king = self.anti_king_var.get()
        self.status_var.set(
            f"Anti-King {'enabled' if self.model.anti_king else 'disabled'}."
        )
        self.draw()

    def toggle_anti_consecutive(self) -> None:
        self.model.anti_consecutive = self.anti_consecutive_var.get()
        self.status_var.set(
            f"Anti-Consecutive {'enabled' if self.model.anti_consecutive else 'disabled'}."
        )
        self.draw()

    def toggle_diag_main(self) -> None:
        self.model.diagonal_main = self.diag_main_var.get()
        self.status_var.set(
            f"Main diagonal {'enabled' if self.model.diagonal_main else 'disabled'}."
        )
        self.draw()

    def toggle_diag_anti(self) -> None:
        self.model.diagonal_anti = self.diag_anti_var.get()
        self.status_var.set(
            f"Anti-diagonal {'enabled' if self.model.diagonal_anti else 'disabled'}."
        )
        self.draw()

    def add_sandwich_clue(self) -> None:
        rc = simpledialog.askstring(
            "Sandwich", "Enter clue as R# or C# (e.g., R5 or C3):", parent=self.root
        )
        if not rc:
            return
        rc = rc.strip().upper()
        if len(rc) < 2 or rc[0] not in ("R", "C") or not rc[1:].isdigit():
            messagebox.showerror("Invalid", "Format must be R# or C# (1-9).")
            return
        idx = int(rc[1:]) - 1
        if not 0 <= idx < 9:
            messagebox.showerror("Invalid", "Index must be 1-9.")
            return
        total = simpledialog.askinteger(
            "Sandwich", "Enter sandwich sum:", parent=self.root, minvalue=0, maxvalue=45
        )
        if total is None:
            return
        constraint = SandwichConstraint(index=idx, is_row=rc[0] == "R", total=total)
        self.model.add_constraint(constraint)
        self._record_constraint(constraint)
        self.status_var.set(f"Added sandwich clue {rc} = {total}.")
        self.draw()

    def clear_digits(self) -> None:
        self.model.clear_digits()
        self.undo_stack = []
        self.status_var.set("Cleared all digits.")
        self.draw()

    def clear_constraints(self) -> None:
        self.model.remove_all_constraints()
        self.reset_path()
        self.undo_stack = []
        self.status_var.set("Removed all variant constraints.")
        self.draw()

    def new_puzzle(self) -> None:
        self.model.reset()
        self.selected_cell = None
        self.current_tool.set("digit")
        self.anti_knight_var.set(False)
        self.anti_king_var.set(False)
        self.anti_consecutive_var.set(False)
        self.diag_main_var.set(False)
        self.diag_anti_var.set(False)
        self.undo_stack = []
        self._last_input_cell = None
        self.status_var.set("Started a new puzzle.")
        self.draw()

    def solve(self) -> None:
        if self.solve_running:
            return
        self.solve_running = True
        self._set_controls_enabled(False)
        self.status_var.set("Solving...")
        puzzle = self.model.copy_grid()
        constraints = self.model.constraints()
        anti_knight = self.model.anti_knight
        anti_king = self.model.anti_king
        check_unique = self.uniqueness_var.get()

        def worker() -> None:
            solver = SudokuSolver(
                puzzle,
                constraints,
                anti_knight=anti_knight,
                anti_king=anti_king,
                anti_consecutive=self.model.anti_consecutive,
                diagonal_main=self.model.diagonal_main,
                diagonal_anti=self.model.diagonal_anti,
            )
            result = solver.solve(require_uniqueness=check_unique)
            self.root.after(0, lambda: self._on_solve_finished(result))

        threading.Thread(target=worker, daemon=True).start()

    def _on_solve_finished(self, result) -> None:
        self.solve_running = False
        self._set_controls_enabled(True)
        if result.status == "no-solution":
            messagebox.showerror("No solution", result.message)
            self.status_var.set(result.message)
            return
        if result.solution:
            self.model.grid = result.solution
        msg = f"{result.message} ({result.duration_ms} ms)"
        if result.status == "multiple":
            messagebox.showwarning("Multiple solutions", msg)
        else:
            messagebox.showinfo("Solved", msg)
        self.status_var.set(msg)
        self.draw()

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "!disabled" if enabled else "disabled"
        for btn in self._control_buttons:
            btn.state([state])

    def draw(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_constraints()
        self._draw_path_preview()
        self._draw_digits()

    def _draw_grid(self) -> None:
        for i in range(10):
            thickness = 3 if i % 3 == 0 else 1
            x0 = self.margin + i * self.cell_size
            y0 = self.margin
            x1 = x0
            y1 = self.margin + 9 * self.cell_size
            self.canvas.create_line(x0, y0, x1, y1, width=thickness)
            x0 = self.margin
            y0 = self.margin + i * self.cell_size
            x1 = self.margin + 9 * self.cell_size
            y1 = y0
            self.canvas.create_line(x0, y0, x1, y1, width=thickness)
        # Bold 3x3 box outlines on top for clarity
        for br in range(3):
            for bc in range(3):
                x0 = self.margin + bc * 3 * self.cell_size
                y0 = self.margin + br * 3 * self.cell_size
                x1 = x0 + 3 * self.cell_size
                y1 = y0 + 3 * self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black", width=4)
        # Cell outlines for clarity
        for r in range(9):
            for c in range(9):
                x0, y0, x1, y1 = self._cell_rect(r, c)
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black", width=1)
        if self.selected_cell:
            r, c = self.selected_cell
            x0, y0, x1, y1 = self._cell_rect(r, c)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="blue", width=2)

    def _cell_rect(self, row: int, col: int) -> Tuple[int, int, int, int]:
        x0 = self.margin + col * self.cell_size
        y0 = self.margin + row * self.cell_size
        x1 = x0 + self.cell_size
        y1 = y0 + self.cell_size
        return x0, y0, x1, y1

    def _center_of(self, row: int, col: int) -> Tuple[int, int]:
        x0, y0, x1, y1 = self._cell_rect(row, col)
        return (x0 + x1) // 2, (y0 + y1) // 2

    def on_canvas_resize(self, event) -> None:
        available = min(event.width, event.height)
        new_size = max(15, int((available - 2 * self.margin) / 9))
        if new_size != self.cell_size:
            self.cell_size = new_size
            self.draw()

    def _draw_digits(self) -> None:
        for r in range(9):
            for c in range(9):
                val = self.model.grid[r][c]
                if val:
                    x0, y0, x1, y1 = self._cell_rect(r, c)
                    self.canvas.create_text(
                        (x0 + x1) // 2,
                        (y0 + y1) // 2,
                        text=str(val),
                        font=("Arial", 18, "bold"),
                        fill="black",
                    )

    def _draw_constraints(self) -> None:
        for constraint in self.model.constraints():
            if isinstance(constraint, ThermometerConstraint):
                self._draw_thermometer(constraint)
            elif isinstance(constraint, GermanWhispersConstraint):
                self._draw_path(
                    constraint.cells,
                    color="#2f9e44",
                    wrap=getattr(constraint, "wrap", False),
                )
            elif isinstance(constraint, DutchWhispersConstraint):
                self._draw_path(
                    constraint.cells,
                    color="#f76707",
                    wrap=getattr(constraint, "wrap", False),
                )
            elif isinstance(constraint, BetweenLineConstraint):
                self._draw_path(
                    constraint.cells,
                    color="#ffa94d",
                    wrap=getattr(constraint, "wrap", False),
                )
            elif isinstance(constraint, ParityLineConstraint):
                self._draw_path(
                    constraint.cells,
                    color="#fa5252",
                    wrap=getattr(constraint, "wrap", False),
                )
            elif isinstance(constraint, RenbanConstraint):
                self._draw_path(
                    constraint.cells,
                    color="#7048e8",
                    wrap=getattr(constraint, "wrap", False),
                )
            elif isinstance(constraint, ZipperLineConstraint):
                self._draw_path(
                    constraint.cells,
                    color="#12b886",
                    wrap=getattr(constraint, "wrap", False),
                )
            elif isinstance(constraint, PalindromeConstraint):
                self._draw_path(constraint.cells, color="#868e96")
            elif isinstance(constraint, RegionSumLineConstraint):
                self._draw_path(
                    constraint.cells,
                    color="#339af0",
                    wrap=getattr(constraint, "wrap", False),
                )
            elif isinstance(constraint, ArrowConstraint):
                self._draw_arrow(constraint)
            elif isinstance(constraint, ParityCellConstraint):
                self._draw_parity_cell(constraint)
            elif isinstance(constraint, SandwichConstraint):
                self._draw_sandwich(constraint)
            elif isinstance(constraint, LittleKillerConstraint):
                self._draw_little_killer(constraint)
            elif isinstance(constraint, KropkiDotConstraint):
                self._draw_kropki(constraint)
            elif isinstance(constraint, XVConstraint):
                self._draw_xv(constraint)
            elif isinstance(constraint, QuadConstraint):
                self._draw_quad(constraint)
            elif isinstance(constraint, KillerCageConstraint):
                self._draw_killer(constraint, show_sum=True)
            elif isinstance(constraint, KillerCageNoSumConstraint):
                self._draw_killer(constraint, show_sum=False)
        if self.model.anti_knight:
            self.canvas.create_text(
                self.margin + self.cell_size * 4.5,
                10,
                text="Anti-Knight",
                fill="#555",
            )
        if self.model.anti_king:
            self.canvas.create_text(
                self.margin + self.cell_size * 4.5,
                25,
                text="Anti-King",
                fill="#555",
            )

    def _draw_path(self, cells: List[Cell], color: str, wrap: bool = False) -> None:
        if len(cells) < 2:
            return
        centers = [self._center_of(r, c) for r, c in cells]
        segments = []
        for i in range(1, len(centers)):
            cx, cy = centers[i]
            prev_idx = min(
                range(i),
                key=lambda j: (centers[j][0] - cx) ** 2 + (centers[j][1] - cy) ** 2,
            )
            segments.append((centers[prev_idx], centers[i]))
        if wrap and len(centers) > 2:
            cx, cy = centers[0]
            other_idx = min(
                range(1, len(centers)),
                key=lambda j: (centers[j][0] - cx) ** 2 + (centers[j][1] - cy) ** 2,
            )
            segments.append((centers[0], centers[other_idx]))
        for (x0, y0), (x1, y1) in segments:
            self.canvas.create_line(
                x0,
                y0,
                x1,
                y1,
                fill=color,
                width=4,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
            )

    def _draw_thermometer(self, constraint: ThermometerConstraint) -> None:
        if not constraint.cells:
            return
        cells = list(constraint.cells)
        bulb = cells[0]
        cx, cy = self._center_of(*bulb)
        radius = self.cell_size // 3
        self.canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            fill="#ced4da",
            outline="",
        )
        self._draw_path(cells, color="#adb5bd")

    def _draw_kropki(self, constraint: KropkiDotConstraint) -> None:
        (r1, c1), (r2, c2) = constraint.a, constraint.b
        x0, y0 = self._center_of(r1, c1)
        x1, y1 = self._center_of(r2, c2)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        radius = self.cell_size // 8
        fill = "white" if constraint.dot_type == "white" else "black"
        outline = "black"
        self.canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            fill=fill,
            outline=outline,
            width=2,
        )

    def _draw_xv(self, constraint: XVConstraint) -> None:
        (r1, c1), (r2, c2) = constraint.a, constraint.b
        x0, y0 = self._center_of(r1, c1)
        x1, y1 = self._center_of(r2, c2)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        self.canvas.create_text(
            cx,
            cy,
            text=constraint.symbol.upper(),
            fill="#000",
            font=("Arial", 12, "bold"),
        )

    def _draw_arrow(self, constraint: ArrowConstraint) -> None:
        if not constraint.path:
            return
        bulb = constraint.bulb
        cells = [bulb] + list(constraint.path)
        self._draw_path(cells, color="#228be6", wrap=False)
        cx, cy = self._center_of(*bulb)
        radius = self.cell_size // 3
        self.canvas.create_oval(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            outline="#228be6",
            width=3,
        )
        end = constraint.path[-1]
        prev = constraint.path[-2] if len(constraint.path) > 1 else bulb
        ex, ey = self._center_of(*end)
        px, py = self._center_of(*prev)
        dx, dy = ex - px, ey - py
        scale = max(1, self.cell_size // 6)
        if dx == 0 and dy == 0:
            return
        dirx = 1 if dx > 0 else -1 if dx < 0 else 0
        diry = 1 if dy > 0 else -1 if dy < 0 else 0
        perp = (-diry, dirx)
        p1 = (ex - dirx * scale, ey - diry * scale)
        p2 = (p1[0] + perp[0] * scale, p1[1] + perp[1] * scale)
        p3 = (p1[0] - perp[0] * scale, p1[1] - perp[1] * scale)
        self.canvas.create_polygon(
            ex, ey, p2[0], p2[1], p3[0], p3[1], fill="#228be6", outline=""
        )

    def _draw_parity_cell(self, constraint: ParityCellConstraint) -> None:
        r, c = constraint.cell
        x0, y0, x1, y1 = self._cell_rect(r, c)
        if constraint.parity == "odd":
            radius = self.cell_size // 4
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill="#adb5bd",
                outline="",
            )
        else:
            inset = self.cell_size // 4
            self.canvas.create_rectangle(
                x0 + inset,
                y0 + inset,
                x1 - inset,
                y1 - inset,
                fill="#adb5bd",
                outline="",
            )

    def _draw_sandwich(self, constraint: SandwichConstraint) -> None:
        if constraint.is_row:
            y0 = self.margin + constraint.index * self.cell_size + self.cell_size // 2
            x0 = self.margin - 12
            self.canvas.create_text(
                x0,
                y0,
                text=str(constraint.total),
                fill="#495057",
                font=("Arial", 10, "bold"),
            )
        else:
            x0 = self.margin + constraint.index * self.cell_size + self.cell_size // 2
            y0 = self.margin - 12
            self.canvas.create_text(
                x0,
                y0,
                text=str(constraint.total),
                fill="#495057",
                font=("Arial", 10, "bold"),
            )

    def _draw_little_killer(self, constraint: LittleKillerConstraint) -> None:
        if not constraint.cells:
            return
        start = constraint.cells[0]
        sx, sy = self._center_of(*start)
        dir_map = {"NE": (1, -1), "NW": (-1, -1), "SE": (1, 1), "SW": (-1, 1)}
        dx, dy = dir_map.get(constraint.direction, (1, -1))
        tx, ty = sx + dx * (self.cell_size // 2), sy + dy * (self.cell_size // 2)
        self.canvas.create_line(sx, sy, tx, ty, arrow=tk.LAST, fill="#228be6", width=2)
        self.canvas.create_text(
            sx + dx * (self.cell_size // 2 + 10),
            sy + dy * (self.cell_size // 2 + 10),
            text=str(constraint.total),
            fill="#228be6",
            font=("Arial", 10, "bold"),
        )

    def _draw_quad(self, constraint: QuadConstraint) -> None:
        rows = {r for r, _ in constraint.cells}
        cols = {c for _, c in constraint.cells}
        if len(rows) != 2 or len(cols) != 2:
            return
        r0, r1 = sorted(rows)
        c0, c1 = sorted(cols)
        x = self.margin + (c0 + 1) * self.cell_size
        y = self.margin + (r0 + 1) * self.cell_size
        radius = self.cell_size // 5
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            outline="#212529",
            fill="#f8f9fa",
        )
        txt = "".join(str(d) for d in constraint.digits)
        self.canvas.create_text(
            x, y, text=txt, fill="#212529", font=("Arial", 10, "bold")
        )

    def _draw_killer(self, constraint, show_sum: bool = True) -> None:
        cells = constraint.cells
        if not cells:
            return
        pad = max(2, self.cell_size // 12)
        for cell in cells:
            r, c = cell
            x0, y0, x1, y1 = self._cell_rect(r, c)
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    if dr == -1:
                        self.canvas.create_line(
                            x0 + pad,
                            y0 + pad,
                            x1 - pad,
                            y0 + pad,
                            fill="black",
                            width=2,
                            dash=(4, 2),
                        )
                    if dr == 1:
                        self.canvas.create_line(
                            x0 + pad,
                            y1 - pad,
                            x1 - pad,
                            y1 - pad,
                            fill="black",
                            width=2,
                            dash=(4, 2),
                        )
                    if dc == -1:
                        self.canvas.create_line(
                            x0 + pad,
                            y0 + pad,
                            x0 + pad,
                            y1 - pad,
                            fill="black",
                            width=2,
                            dash=(4, 2),
                        )
                    if dc == 1:
                        self.canvas.create_line(
                            x1 - pad,
                            y0 + pad,
                            x1 - pad,
                            y1 - pad,
                            fill="black",
                            width=2,
                            dash=(4, 2),
                        )
        if show_sum and hasattr(constraint, "cage_sum"):
            top_left = min(cells, key=lambda c: (c[0], c[1]))
            x0, y0, _, _ = self._cell_rect(*top_left)
            self.canvas.create_text(
                x0 + pad + 4,
                y0 + pad + 4,
                text=str(constraint.cage_sum),
                anchor="nw",
                fill="#fa5252",
                font=("Arial", 10, "bold"),
            )

    def _draw_path_preview(self) -> None:
        if not self.current_path:
            return
        color_map = {
            "thermo": "#adb5bd",
            "arrow": "#228be6",
            "between": "#ffa94d",
            "whispers": "#2f9e44",
            "dutch": "#f76707",
            "parity": "#fa5252",
            "renban": "#7048e8",
            "palindrome": "#868e96",
            "region-sum": "#339af0",
            "zipper": "#12b886",
            "killer": "#fa5252",
            "kropki-white": "#000",
            "kropki-black": "#000",
            "xv-x": "#000",
            "xv-v": "#000",
            "odd": "#adb5bd",
            "even": "#adb5bd",
            "quad": "#212529",
            "little-killer": "#228be6",
        }
        color = color_map.get(self.current_tool.get(), "#339af0")
        for cell in self.current_path:
            x0, y0, x1, y1 = self._cell_rect(*cell)
            self.canvas.create_rectangle(
                x0 + 2, y0 + 2, x1 - 2, y1 - 2, outline=color, width=2
            )
        if len(self.current_path) > 1:
            self._draw_path(self.current_path, color=color)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = SudokuApp()
    app.run()
