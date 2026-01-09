# Variant Sudoku Solver

Desktop Sudoku solver and designer with support for popular variant constraints. Create puzzles in the UI, add constraints, solve locally, and run batch tooling to import and validate large puzzle sets.

## Features
- Tkinter GUI for building and solving puzzles
- Classic Sudoku rules + variants (thermo, arrows, renban, kropki, killer cages, palindromes, etc.)
- Constraint propagation + backtracking solver
- JSON save/load format for puzzles
- Batch tools to import SudokuPad links, solve many puzzles, and report errors

## Requirements
- Python 3.11+
- Tkinter (included with most Python installs)

## Run the GUI
From the repo root:
```
python main.py
```

## UI Basics
- Select a tool from the dropdown (Digits, Thermometer, Arrow, etc.)
- Click cells to place digits or draw constraints
- Use **Finish Shape** to finalize a line/cage tool
- **Undo** reverses the last action
- **Save Puzzle** / **Load Puzzle** use JSON files
- **Solve** runs the solver

## Puzzle JSON Format
Saved puzzles are JSON objects with:
- `grid`: 9x9 list of ints (0 = empty)
- `constraints`: list of constraint objects
- toggle flags: `anti_knight`, `anti_king`, `anti_consecutive`, `diag_main`, `diag_anti`


## Project Structure
- `main.py`: GUI app
- `constraints.py`: constraint definitions and propagation
- `solver.py`: solver engine
- `model.py`: puzzle state

## Notes
- If a puzzle has multiple solutions, it usually indicates missing constraints or under-specification.
