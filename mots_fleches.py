# -*- coding: utf-8 -*-
"""Mots fleches generator (French/Swedish style).

Implements a strict grid validator, a two-step generation pipeline, and PDF output.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple
import logging

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
except Exception:  # pragma: no cover - optional dependency
    A4 = None
    canvas = None
    mm = 1.0

import urllib.request
import urllib.error

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


logger = logging.getLogger("mots_fleches")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler("def_debug.log", encoding="utf-8")],
    )


def _read_env_file(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                # Handle UTF-8 BOM if present
                if key.startswith("\ufeff"):
                    key = key.lstrip("\ufeff")
                val = val.strip().strip("\"'")
                if key:
                    out[key] = val
    except Exception:
        return out
    return out


def load_env_fallback(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        data = _read_env_file(path)
        for key, val in data.items():
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


WIDTH = 13
HEIGHT = 21

DIRECTIONS = {"RIGHT", "DOWN", "RIGHT_DOWN"}


@dataclass
class DefEntry:
    direction: str
    clue_text: str = ""


@dataclass
class Cell:
    type: Optional[str] = None  # "LETTER" | "DEF" | None during build
    letter: Optional[str] = None
    defs: List[DefEntry] = field(default_factory=list)


@dataclass
class WordPlacement:
    word: str
    start_x: int
    start_y: int
    direction: str  # "RIGHT" | "DOWN"


@dataclass
class Slot:
    start_x: int
    start_y: int
    direction: str  # "RIGHT" | "DOWN"
    length: int
    pattern: str


@dataclass
class DictionaryIndex:
    by_length: Dict[int, List[str]]
    words: Set[str]
    pattern_cache: Dict[str, bool] = field(default_factory=dict)
    bigrams: Set[str] = field(default_factory=set)

    def matches_pattern(self, pattern: str) -> bool:
        cached = self.pattern_cache.get(pattern)
        if cached is not None:
            return cached
        n = len(pattern)
        words = self.by_length.get(n, [])
        for w in words:
            ok = True
            for ch, p in zip(w, pattern):
                if p != "." and p != ch:
                    ok = False
                    break
            if ok:
                self.pattern_cache[pattern] = True
                return True
        self.pattern_cache[pattern] = False
        return False


class Grid:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.cells = [[Cell() for _ in range(width)] for _ in range(height)]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int) -> Cell:
        return self.cells[y][x]

    def set_letter(self, x: int, y: int, letter: str) -> bool:
        cell = self.get(x, y)
        if cell.type == "DEF":
            return False
        if cell.letter is not None and cell.letter != letter:
            return False
        cell.type = "LETTER"
        cell.letter = letter
        return True

    def set_def(self, x: int, y: int) -> bool:
        cell = self.get(x, y)
        if cell.type == "LETTER":
            return False
        cell.type = "DEF"
        cell.letter = None
        if cell.defs is None:
            cell.defs = []
        return True

    def to_json(self) -> Dict:
        cells_out = []
        for y in range(self.height):
            for x in range(self.width):
                c = self.get(x, y)
                cell_out = {
                    "x": x,
                    "y": y,
                    "type": c.type,
                    "letter": c.letter,
                    "defs": [
                        {"direction": d.direction, "clue_text": d.clue_text}
                        for d in (c.defs or [])
                    ],
                }
                cells_out.append(cell_out)
        return {"width": self.width, "height": self.height, "cells": cells_out}


def normalize_word(raw: str) -> Optional[str]:
    raw = raw.strip()
    if not raw:
        return None
    # Remove accents/diacritics, keep letters only
    raw = unicodedata.normalize("NFD", raw)
    letters = [ch for ch in raw if ch.isalpha()]
    if not letters:
        return None
    word = "".join(letters).upper()
    if len(word) < 2:
        return None
    return word


def load_mandatory_words(path: str) -> List[str]:
    # Try utf-8 first, then latin-1
    data = None
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                data = f.read()
            break
        except Exception:
            continue
    if data is None:
        raise RuntimeError(f"Unable to read {path}")

    words: List[str] = []
    for line in data.splitlines():
        w = normalize_word(line)
        if w:
            words.append(w)
    if not words:
        raise RuntimeError("No valid mandatory words found")
    return words


def load_dictionary(path: str) -> Set[str]:
    data = None
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                data = f.read()
            break
        except Exception:
            continue
    if data is None:
        raise RuntimeError(f"Unable to read dictionary {path}")
    out = set()
    for line in data.splitlines():
        w = normalize_word(line)
        if w:
            out.add(w)
    return out


def build_dictionary_index(words: Sequence[str]) -> DictionaryIndex:
    by_length: Dict[int, List[str]] = {}
    word_set = set(words)
    bigrams: Set[str] = set()
    for w in words:
        by_length.setdefault(len(w), []).append(w)
        for i in range(len(w) - 1):
            bigrams.add(w[i : i + 2])
    return DictionaryIndex(by_length=by_length, words=word_set, bigrams=bigrams)


def def_priority(def_pos: Optional[Tuple[int, int]], width: int, height: int) -> int:
    if def_pos is None:
        return 3
    x, y = def_pos
    corners = {(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)}
    if def_pos in corners:
        return 0
    # Adjacent to corner
    for cx, cy in corners:
        if abs(cx - x) + abs(cy - y) <= 1:
            return 0
    if x == 0 or y == 0 or x == width - 1 or y == height - 1:
        return 1
    return 2


def placement_cells(word: str, start_x: int, start_y: int, direction: str) -> List[Tuple[int, int]]:
    coords = []
    for i in range(len(word)):
        if direction == "RIGHT":
            coords.append((start_x + i, start_y))
        else:
            coords.append((start_x, start_y + i))
    return coords


def reserved_def_positions_for(word: str, start_x: int, start_y: int, direction: str, grid: Grid) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    if direction == "RIGHT":
        def_pos = (start_x - 1, start_y) if start_x - 1 >= 0 else None
        end_pos = (start_x + len(word), start_y) if start_x + len(word) < grid.width else None
    else:
        def_pos = (start_x, start_y - 1) if start_y - 1 >= 0 else None
        end_pos = (start_x, start_y + len(word)) if start_y + len(word) < grid.height else None
    return def_pos, end_pos


def creates_forced_single_letter_word(grid: Grid, new_def_positions: Set[Tuple[int, int]]) -> bool:
    # Conservative check: if a fixed letter is isolated between two boundaries in a row or column.
    def is_boundary(x: int, y: int) -> bool:
        if not grid.in_bounds(x, y):
            return True
        if (x, y) in new_def_positions:
            return True
        c = grid.get(x, y)
        return c.type == "DEF"

    # Check rows
    for y in range(grid.height):
        x = 0
        while x < grid.width:
            # Skip boundary
            while x < grid.width and is_boundary(x, y):
                x += 1
            start = x
            while x < grid.width and not is_boundary(x, y):
                x += 1
            end = x
            if end - start == 1:
                return True
    # Check columns
    for x in range(grid.width):
        y = 0
        while y < grid.height:
            while y < grid.height and is_boundary(x, y):
                y += 1
            start = y
            while y < grid.height and not is_boundary(x, y):
                y += 1
            end = y
            if end - start == 1:
                return True
    return False


def def_run_too_long(grid: Grid, extra_defs: Set[Tuple[int, int]]) -> bool:
    def is_def(x: int, y: int) -> bool:
        if not grid.in_bounds(x, y):
            return False
        if (x, y) in extra_defs:
            return True
        return grid.get(x, y).type == "DEF"

    # Rows
    for y in range(grid.height):
        run = 0
        for x in range(grid.width):
            if is_def(x, y):
                run += 1
                if run > 2:
                    return True
            else:
                run = 0
    # Columns
    for x in range(grid.width):
        run = 0
        for y in range(grid.height):
            if is_def(x, y):
                run += 1
                if run > 2:
                    return True
            else:
                run = 0
    return False


def _adjacent_has_letter(grid: Grid, x: int, y: int) -> bool:
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if grid.in_bounds(nx, ny):
            c = grid.get(nx, ny)
            if c.type == "LETTER":
                return True
    return False


def try_place_word(
    grid: Grid,
    word: str,
    start_x: int,
    start_y: int,
    direction: str,
    reserved_defs: Set[Tuple[int, int]],
    placements: List[WordPlacement],
    max_crossings: int = 999,
    min_parallel_gap: int = 0,
    forbid_border_parallel_gap1: bool = False,
    allow_adjacent_letters: bool = False,
    skip_new_defs: bool = False,
    force_end_def: bool = False,
    relax_end_def_checks: bool = False,
) -> bool:
    # Bounds
    if direction == "RIGHT" and start_x + len(word) > grid.width:
        return False
    if direction == "DOWN" and start_y + len(word) > grid.height:
        return False

    if forbid_border_parallel_gap1:
        if direction == "RIGHT" and (start_y == 1 or start_y == grid.height - 2):
            return False
        if direction == "DOWN" and (start_x == 1 or start_x == grid.width - 2):
            return False

    def_pos, end_pos = reserved_def_positions_for(word, start_x, start_y, direction, grid)

    # Require a def cell if possible (prefer internal). If no def_pos (border-start), allow but mark as lower priority.
    if def_pos is not None:
        c = grid.get(*def_pos)
        if c.type == "LETTER":
            return False
    if end_pos is not None:
        c = grid.get(*end_pos)
        if c.type == "LETTER":
            return False

    # Check letters
    crossings = 0
    coords = placement_cells(word, start_x, start_y, direction)
    for (x, y), ch in zip(coords, word):
        c = grid.get(x, y)
        if c.type == "DEF":
            return False
        if c.letter is not None and c.letter != ch:
            return False
        if c.letter is not None and c.letter == ch:
            crossings += 1
        if c.letter is None and not allow_adjacent_letters:
            # Enforce no touching letters orthogonally (simple crossings only)
            if direction == "RIGHT":
                if (grid.in_bounds(x, y - 1) and grid.get(x, y - 1).type == "LETTER") or (
                    grid.in_bounds(x, y + 1) and grid.get(x, y + 1).type == "LETTER"
                ):
                    return False
            else:
                if (grid.in_bounds(x - 1, y) and grid.get(x - 1, y).type == "LETTER") or (
                    grid.in_bounds(x + 1, y) and grid.get(x + 1, y).type == "LETTER"
                ):
                    return False
            # Enforce extra spacing between parallel words (avoid dense bands)
            if min_parallel_gap > 0:
                if direction == "RIGHT":
                    for dy in range(1, min_parallel_gap + 1):
                        for ny in (y - dy, y + dy):
                            if grid.in_bounds(x, ny) and grid.get(x, ny).type == "LETTER":
                                return False
                else:
                    for dx in range(1, min_parallel_gap + 1):
                        for nx in (x - dx, x + dx):
                            if grid.in_bounds(nx, y) and grid.get(nx, y).type == "LETTER":
                                return False

    # Tentatively reserve defs and check single-letter forced words
    new_defs = set(reserved_defs)
    if not skip_new_defs or force_end_def:
        if not skip_new_defs and def_pos is not None:
            new_defs.add(def_pos)
        if end_pos is not None:
            new_defs.add(end_pos)
        if not relax_end_def_checks:
            if def_run_too_long(grid, new_defs):
                return False
            if creates_forced_single_letter_word(grid, new_defs):
                return False

    if crossings > max_crossings:
        return False

    # Apply
    for (x, y), ch in zip(coords, word):
        if not grid.set_letter(x, y, ch):
            return False
    if not skip_new_defs:
        if def_pos is not None:
            grid.set_def(*def_pos)
            reserved_defs.add(def_pos)
    if not skip_new_defs or force_end_def:
        if end_pos is not None:
            grid.set_def(*end_pos)
            reserved_defs.add(end_pos)

    placements.append(WordPlacement(word=word, start_x=start_x, start_y=start_y, direction=direction))
    return True


def analyze_candidate(
    grid: Grid,
    word: str,
    start_x: int,
    start_y: int,
    direction: str,
    min_parallel_gap: int,
) -> Optional[Tuple[int, int]]:
    """Return (crossings, parallel_close) if candidate is locally compatible, else None."""
    # Bounds
    if direction == "RIGHT" and start_x + len(word) > grid.width:
        return None
    if direction == "DOWN" and start_y + len(word) > grid.height:
        return None

    crossings = 0
    parallel_close = 0
    coords = placement_cells(word, start_x, start_y, direction)
    for (x, y), ch in zip(coords, word):
        c = grid.get(x, y)
        if c.type == "DEF":
            return None
        if c.letter is not None and c.letter != ch:
            return None
        if c.letter is not None and c.letter == ch:
            crossings += 1
            continue
        # No touching letters orthogonally
        if direction == "RIGHT":
            if (grid.in_bounds(x, y - 1) and grid.get(x, y - 1).type == "LETTER") or (
                grid.in_bounds(x, y + 1) and grid.get(x, y + 1).type == "LETTER"
            ):
                return None
        else:
            if (grid.in_bounds(x - 1, y) and grid.get(x - 1, y).type == "LETTER") or (
                grid.in_bounds(x + 1, y) and grid.get(x + 1, y).type == "LETTER"
            ):
                return None
        # Parallel proximity penalty (count how many nearby letters in parallel bands)
        if min_parallel_gap > 0:
            if direction == "RIGHT":
                for dy in range(1, min_parallel_gap + 1):
                    for ny in (y - dy, y + dy):
                        if grid.in_bounds(x, ny) and grid.get(x, ny).type == "LETTER":
                            parallel_close += 1
            else:
                for dx in range(1, min_parallel_gap + 1):
                    for nx in (x - dx, x + dx):
                        if grid.in_bounds(nx, y) and grid.get(nx, y).type == "LETTER":
                            parallel_close += 1
    return (crossings, parallel_close)


def select_mandatory_subset(words: List[str], max_words: int, seed: int) -> List[str]:
    if max_words <= 0 or max_words >= len(words):
        return words[:]
    rng = random.Random(seed)
    # Bucket by length to encourage variety
    buckets: Dict[int, List[str]] = {}
    for w in words:
        buckets.setdefault(len(w), []).append(w)
    lengths = sorted(buckets.keys())
    # Aim for a spread across short/medium/long
    pick: List[str] = []
    while len(pick) < max_words and lengths:
        for L in list(lengths):
            if len(pick) >= max_words:
                break
            if not buckets[L]:
                lengths.remove(L)
                continue
            pick.append(buckets[L].pop(rng.randrange(len(buckets[L]))))
        if not lengths:
            break
    return pick


def select_words_for_slots(words: List[str], slot_lengths: List[int], max_words: int, seed: int) -> List[str]:
    if not slot_lengths:
        return []
    rng = random.Random(seed)
    max_len = max(slot_lengths)
    lengths = sorted(set(slot_lengths))
    buckets: Dict[int, List[str]] = {}
    for w in words:
        if 2 <= len(w) <= max_len:
            buckets.setdefault(len(w), []).append(w)
    # Round-robin across available slot lengths
    pick: List[str] = []
    while len(pick) < max_words and lengths:
        for L in list(lengths):
            if len(pick) >= max_words:
                break
            if not buckets.get(L):
                lengths.remove(L)
                continue
            pick.append(buckets[L].pop(rng.randrange(len(buckets[L]))))
        if not lengths:
            break
    return pick


def _predef_slots(grid: Grid, predefs: Set[Tuple[int, int]]) -> List[Slot]:
    slots: List[Slot] = []
    seen = set()
    for y in range(grid.height):
        for x in range(grid.width):
            if (x, y) not in predefs:
                continue
            c = grid.get(x, y)
            if c.type != "DEF":
                continue
            for d in c.defs or []:
                if d.direction == "RIGHT":
                    sx, sy, direction = x + 1, y, "RIGHT"
                elif d.direction == "DOWN":
                    sx, sy, direction = x, y + 1, "DOWN"
                elif d.direction == "RIGHT_DOWN":
                    sx, sy, direction = x, y + 1, "RIGHT"
                else:
                    continue
                if not grid.in_bounds(sx, sy) or grid.get(sx, sy).type == "DEF":
                    continue
                slot = slot_from(grid, sx, sy, direction)
                if slot.length < 2:
                    continue
                key = (slot.start_x, slot.start_y, slot.direction)
                if key in seen:
                    continue
                seen.add(key)
                slots.append(slot)
    return slots


def predef_slot_lengths(grid: Grid, predefs: Set[Tuple[int, int]]) -> List[int]:
    return [s.length for s in _predef_slots(grid, predefs)]


def place_mandatory_words_backtracking(
    grid: Grid,
    words: List[str],
    dict_index: DictionaryIndex,
    predefs: Set[Tuple[int, int]],
    max_words: int,
    seed: int = 0,
) -> Tuple[List[WordPlacement], List[str]]:
    rng = random.Random(seed)
    slots = _predef_slots(grid, predefs)
    lengths = [s.length for s in slots]
    selected = select_words_for_slots(words, lengths, max_words, seed)
    # Ensure dictionary contains mandatory words
    for w in selected:
        if w not in dict_index.words:
            dict_index.words.add(w)
            dict_index.by_length.setdefault(len(w), []).append(w)
    selected = sorted(selected, key=len, reverse=True)
    slots_by_len: Dict[int, List[Slot]] = {}
    for s in slots:
        slots_by_len.setdefault(s.length, []).append(s)
    placements: List[WordPlacement] = []
    used_slots: Set[Tuple[int, int, str]] = set()

    def backtrack(i: int) -> bool:
        if i >= len(selected):
            return True
        word = selected[i]
        # No preference: allow exact or longer (with end-DEF cut)
        cand = [
            s for s in slots
            if s.length >= len(word) and (s.start_x, s.start_y, s.direction) not in used_slots
        ]
        rng.shuffle(cand)
        for slot in cand:
            # Check conflicts
            ok = True
            x, y = slot.start_x, slot.start_y
            for ch in word:
                c = grid.get(x, y)
                if c.type == "DEF":
                    ok = False
                    break
                if c.letter is not None and c.letter != ch:
                    ok = False
                    break
                if slot.direction == "RIGHT":
                    x += 1
                else:
                    y += 1
            if not ok:
                continue
            # Optional end DEF to cut longer slots
            added_def = None
            if len(word) < slot.length:
                end_x = slot.start_x + (len(word) if slot.direction == "RIGHT" else 0)
                end_y = slot.start_y + (len(word) if slot.direction == "DOWN" else 0)
                if grid.in_bounds(end_x, end_y):
                    end_cell = grid.get(end_x, end_y)
                    if end_cell.type == "DEF" or end_cell.letter is not None:
                        continue
                    grid.set_def(end_x, end_y)
                    dirs = possible_def_dirs(grid, end_x, end_y)
                    if not dirs:
                        end_cell.type = None
                        end_cell.defs = []
                        continue
                    end_cell.defs = [DefEntry(direction=d) for d in dirs[:1]]
                    added_def = (end_x, end_y)
                else:
                    # Should not happen, but skip
                    continue
            changed = place_word_letters(grid, slot, word)
            coords = [(x, y) for x, y, _ in changed]
            if perpendicular_ok(grid, dict_index, coords, slot.direction):
                used_slots.add((slot.start_x, slot.start_y, slot.direction))
                placements.append(
                    WordPlacement(word=word, start_x=slot.start_x, start_y=slot.start_y, direction=slot.direction)
                )
                if backtrack(i + 1):
                    return True
                placements.pop()
                used_slots.remove((slot.start_x, slot.start_y, slot.direction))
            revert_letters(grid, changed)
            if added_def:
                ax, ay = added_def
                c = grid.get(ax, ay)
                c.type = None
                c.defs = []
        return False

    backtrack(0)
    placed_words = {p.word for p in placements}
    unplaced = [w for w in selected if w not in placed_words]
    return placements, unplaced


def place_mandatory_with_skeleton(
    words: List[str],
    dict_index: DictionaryIndex,
    max_words: int,
    seed: int = 0,
    attempts: int = 6,
    def_ratio: float = 0.22,
) -> Tuple[Grid, List[WordPlacement], List[str]]:
    best = None
    for i in range(attempts):
        grid = Grid(WIDTH, HEIGHT)
        apply_border_def_pattern(grid, seed=seed + i)
        try:
            auto_place_defs(grid, seed=seed + i, target_def_ratio=def_ratio)
        except Exception:
            continue
        predefs = {(x, y) for y in range(HEIGHT) for x in range(WIDTH) if grid.get(x, y).type == "DEF"}
        placements, unplaced = place_mandatory_words_backtracking(
            grid, words, dict_index, predefs, max_words, seed=seed + i
        )
        if best is None or len(placements) > len(best[1]):
            best = (grid, placements, unplaced)
        if len(placements) >= max_words:
            return grid, placements, unplaced
    if best:
        return best
    # Fallback: border only, no interior defs
    grid = Grid(WIDTH, HEIGHT)
    apply_border_def_pattern(grid, seed=seed)
    predefs = {(x, y) for y in range(HEIGHT) for x in range(WIDTH) if grid.get(x, y).type == "DEF"}
    placements, unplaced = place_mandatory_words_backtracking(
        grid, words, dict_index, predefs, max_words, seed=seed
    )
    return grid, placements, unplaced


def progressive_place_next(
    grid: Grid,
    word: str,
    dict_index: DictionaryIndex,
    predefs: Set[Tuple[int, int]],
    used_slots: Set[Tuple[int, int, str]],
    seed: int = 0,
) -> Optional[Tuple[WordPlacement, Optional[Tuple[int, int]], Slot]]:
    rng = random.Random(seed)
    slots = list_slots(grid)
    # Prefer shorter slots and vertical slots to avoid filling full rows early
    rng.shuffle(slots)
    slots.sort(key=lambda s: (s.length, 0 if s.direction == "DOWN" else 1, rng.random()))
    for slot in slots:
        key = (slot.start_x, slot.start_y, slot.direction)
        if key in used_slots:
            continue
        # Allow shorter words by cutting slots with an end DEF
        res = try_place_word_progressive(
            grid,
            slot,
            word,
            dict_index,
            predefs,
            check_perpendicular=True,
            relax_end_def_checks=True,
        )
        if res:
            placement, added = res
            used_slots.add(key)
            return placement, added, slot
    return None


def place_mandatory_words(
    grid: Grid,
    words: List[str],
    seed: int,
    predefs: Optional[Set[Tuple[int, int]]] = None,
    min_parallel_gap: int = 1,
    forbid_border_parallel_gap1: bool = False,
    predef_only: bool = False,
    allow_adjacent_letters: bool = True,
    relax_end_def_checks: bool = True,
) -> Tuple[List[WordPlacement], List[str], Set[Tuple[int, int]]]:
    rng = random.Random(seed)
    # Shuffle but bias to longer words first
    words_sorted = sorted(words, key=lambda w: (-len(w), rng.random()))
    placements: List[WordPlacement] = []
    reserved_defs: Set[Tuple[int, int]] = set()
    unplaced: List[str] = []
    predefs = predefs or set()
    target_existing = max(1, len(words_sorted) // 2)
    used_existing = 0

    for idx, word in enumerate(words_sorted):
        # Try to use pre-existing DEF slots for about half of words (or only those slots)
        if (used_existing < target_existing or predef_only) and predefs:
            slots = _predef_slots(grid, predefs)
            slots = [s for s in slots if s.length >= len(word)]
            rng.shuffle(slots)
            placed = False
            for slot in slots:
                if try_place_word(
                    grid,
                    word,
                    slot.start_x,
                    slot.start_y,
                    slot.direction,
                    reserved_defs,
                    placements,
                    max_crossings=999,
                    min_parallel_gap=min_parallel_gap,
                    forbid_border_parallel_gap1=forbid_border_parallel_gap1,
                    allow_adjacent_letters=allow_adjacent_letters,
                    skip_new_defs=predef_only,
                    force_end_def=predef_only,
                    relax_end_def_checks=relax_end_def_checks,
                ):
                    used_existing += 1
                    placed = True
                    break
            if placed:
                continue
            if predef_only:
                unplaced.append(word)
                continue

        candidates = []
        # Alternate preferred direction to vary orientation
        directions = ("RIGHT", "DOWN") if idx % 2 == 0 else ("DOWN", "RIGHT")
        for direction in directions:
            if direction == "RIGHT":
                max_x = grid.width - len(word)
                max_y = grid.height - 1
                for y in range(grid.height):
                    for x in range(max_x + 1):
                        def_pos, _ = reserved_def_positions_for(word, x, y, direction, grid)
                        prio = def_priority(def_pos, grid.width, grid.height)
                        candidates.append((prio, x, y, direction))
            else:
                max_x = grid.width - 1
                max_y = grid.height - len(word)
                for y in range(max_y + 1):
                    for x in range(grid.width):
                        def_pos, _ = reserved_def_positions_for(word, x, y, direction, grid)
                        prio = def_priority(def_pos, grid.width, grid.height)
                        candidates.append((prio, x, y, direction))

        # Score candidates: prefer crossings, avoid parallel bands
        scored = []
        for prio, x, y, direction in candidates:
            res = analyze_candidate(grid, word, x, y, direction, min_parallel_gap=2)
            if res is None:
                continue
            crossings, parallel_close = res
            # Strongly favor crossings, mildly penalize parallel proximity
            score = crossings * 10 - parallel_close * 2 - prio * 3
            scored.append((score, prio, x, y, direction, crossings))

        if not scored:
            unplaced.append(word)
            continue

        # If any candidate crosses, drop non-crossing candidates to encourage intersections
        if any(s[5] > 0 for s in scored):
            scored = [s for s in scored if s[5] > 0]

        rng.shuffle(scored)
        scored.sort(key=lambda t: (-t[0], t[1]))

        placed = False
        for _, _, x, y, direction, _cross in scored:
            if try_place_word(
                grid,
                word,
                x,
                y,
                direction,
                reserved_defs,
                placements,
                max_crossings=999,
                min_parallel_gap=min_parallel_gap,
                forbid_border_parallel_gap1=forbid_border_parallel_gap1,
                allow_adjacent_letters=predef_only,
                skip_new_defs=predef_only,
            ):
                placed = True
                break
        if not placed:
            unplaced.append(word)

    return placements, unplaced, reserved_defs


def mandatory_placement_generator(
    grid: Grid,
    words: List[str],
    seed: int,
    predefs: Optional[Set[Tuple[int, int]]] = None,
    min_parallel_gap: int = 1,
    forbid_border_parallel_gap1: bool = False,
    predef_only: bool = False,
    allow_adjacent_letters: bool = True,
    relax_end_def_checks: bool = True,
):
    """Yield after each successful mandatory word placement.

    Yields dicts with keys: placed_word, placements, reserved_defs, unplaced, done.
    """
    rng = random.Random(seed)
    words_sorted = sorted(words, key=lambda w: (-len(w), rng.random()))
    placements: List[WordPlacement] = []
    reserved_defs: Set[Tuple[int, int]] = set()
    unplaced: List[str] = []
    predefs = predefs or set()
    target_existing = max(1, len(words_sorted) // 2)
    used_existing = 0

    for idx, word in enumerate(words_sorted):
        if (used_existing < target_existing or predef_only) and predefs:
            slots = _predef_slots(grid, predefs)
            slots = [s for s in slots if s.length >= len(word)]
            rng.shuffle(slots)
            placed = False
            for slot in slots:
                if try_place_word(
                    grid,
                    word,
                    slot.start_x,
                    slot.start_y,
                    slot.direction,
                    reserved_defs,
                    placements,
                max_crossings=999,
                min_parallel_gap=min_parallel_gap,
                forbid_border_parallel_gap1=forbid_border_parallel_gap1,
                allow_adjacent_letters=allow_adjacent_letters,
                skip_new_defs=predef_only,
                force_end_def=predef_only,
                relax_end_def_checks=relax_end_def_checks,
            ):
                    used_existing += 1
                    placed = True
                    yield {
                        "placed_word": word,
                        "placements": placements,
                        "reserved_defs": reserved_defs,
                        "unplaced": unplaced,
                        "done": False,
                    }
                    break
            if placed:
                continue
            if predef_only:
                unplaced.append(word)
                continue
        candidates = []
        directions = ("RIGHT", "DOWN") if idx % 2 == 0 else ("DOWN", "RIGHT")
        for direction in directions:
            if direction == "RIGHT":
                max_x = grid.width - len(word)
                for y in range(grid.height):
                    for x in range(max_x + 1):
                        def_pos, _ = reserved_def_positions_for(word, x, y, direction, grid)
                        prio = def_priority(def_pos, grid.width, grid.height)
                        candidates.append((prio, x, y, direction))
            else:
                max_y = grid.height - len(word)
                for y in range(max_y + 1):
                    for x in range(grid.width):
                        def_pos, _ = reserved_def_positions_for(word, x, y, direction, grid)
                        prio = def_priority(def_pos, grid.width, grid.height)
                        candidates.append((prio, x, y, direction))

        scored = []
        for prio, x, y, direction in candidates:
            res = analyze_candidate(grid, word, x, y, direction, min_parallel_gap=2)
            if res is None:
                continue
            crossings, parallel_close = res
            score = crossings * 10 - parallel_close * 2 - prio * 3
            scored.append((score, prio, x, y, direction, crossings))

        if not scored:
            unplaced.append(word)
            continue

        if any(s[5] > 0 for s in scored):
            scored = [s for s in scored if s[5] > 0]

        rng.shuffle(scored)
        scored.sort(key=lambda t: (-t[0], t[1]))

        placed = False
        for _, _, x, y, direction, _cross in scored:
            if try_place_word(
                grid,
                word,
                x,
                y,
                direction,
                reserved_defs,
                placements,
                max_crossings=999,
                min_parallel_gap=min_parallel_gap,
                forbid_border_parallel_gap1=forbid_border_parallel_gap1,
                allow_adjacent_letters=predef_only,
                skip_new_defs=predef_only,
            ):
                placed = True
                yield {
                    "placed_word": word,
                    "placements": placements,
                    "reserved_defs": reserved_defs,
                    "unplaced": unplaced,
                    "done": False,
                }
                break
        if not placed:
            unplaced.append(word)

    yield {
        "placed_word": None,
        "placements": placements,
        "reserved_defs": reserved_defs,
        "unplaced": unplaced,
        "done": True,
    }


def call_openai_json(api_key: str, model: str, system: str, user: str) -> Dict:
    # Minimal OpenAI Responses API call
    url = "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # Responses API JSON mode is now configured via text.format
        "text": {"format": {"type": "json_object"}},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI API error: {e.code} {detail}")
    payload_out = json.loads(raw)
    # Responses API returns output in payload_out["output"][0]["content"][0]["text"]
    # Be defensive
    text = None
    try:
        for item in payload_out.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text = content.get("text")
                    break
            if text:
                break
    except Exception:
        text = None
    if text is None:
        # Fallback: try 'output_text'
        text = payload_out.get("output_text")
    if text is None:
        raise RuntimeError("OpenAI API returned no text")
    return json.loads(text)


def grid_constraints_for_llm(grid: Grid, placements: List[WordPlacement], reserved_defs: Set[Tuple[int, int]],
                             unplaced_words: List[str], all_words: List[str]) -> str:
    fixed_letters = []
    for y in range(grid.height):
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type == "LETTER" and c.letter:
                fixed_letters.append({"x": x, "y": y, "letter": c.letter})
    reserved = [{"x": x, "y": y} for (x, y) in sorted(reserved_defs)]
    placed = [
        {"word": p.word, "start_x": p.start_x, "start_y": p.start_y, "direction": p.direction}
        for p in placements
    ]

    payload = {
        "width": grid.width,
        "height": grid.height,
        "fixed_letters": fixed_letters,
        "reserved_def_cells": reserved,
        "placed_words": placed,
        "mandatory_words": all_words,
        "mandatory_unplaced": unplaced_words,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def grid_compact_repr(grid: Grid) -> str:
    lines = []
    for y in range(grid.height):
        tokens = []
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type == "LETTER" and c.letter:
                tokens.append(c.letter)
            elif c.type == "DEF":
                dirs = []
                for d in c.defs or []:
                    if d.direction == "RIGHT":
                        dirs.append("R")
                    elif d.direction == "DOWN":
                        dirs.append("D")
                    else:
                        dirs.append("RD")
                if dirs:
                    tokens.append("DEF(" + ",".join(sorted(set(dirs))) + ")")
                else:
                    tokens.append("DEF")
            else:
                tokens.append(".")
        lines.append(" | ".join(tokens))
    return "\n".join(lines)


def parse_compact_grid(text: str) -> Grid:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) != HEIGHT:
        raise ValueError("Invalid compact grid line count")
    grid = Grid(WIDTH, HEIGHT)
    for y, line in enumerate(lines):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != WIDTH:
            raise ValueError("Invalid compact grid column count")
        for x, tok in enumerate(parts):
            if tok == ".":
                # Placeholder, should not happen in final grid
                raise ValueError("Compact grid contains empty placeholder")
            if len(tok) == 1 and tok.isalpha():
                grid.set_letter(x, y, tok.upper())
                continue
            if tok.startswith("DEF"):
                grid.set_def(x, y)
                c = grid.get(x, y)
                c.defs = []
                if tok.startswith("DEF(") and tok.endswith(")"):
                    inside = tok[4:-1].strip()
                    if inside:
                        for part in inside.split(","):
                            part = part.strip().upper()
                            if part == "R":
                                c.defs.append(DefEntry(direction="RIGHT"))
                            elif part == "D":
                                c.defs.append(DefEntry(direction="DOWN"))
                            elif part == "RD":
                                c.defs.append(DefEntry(direction="RIGHT_DOWN"))
                            else:
                                raise ValueError("Invalid DEF direction in compact grid")
                else:
                    c.defs.append(DefEntry(direction="RIGHT"))
                continue
            raise ValueError("Invalid token in compact grid")
    return grid


def slot_from(grid: Grid, start_x: int, start_y: int, direction: str) -> Slot:
    letters = []
    x, y = start_x, start_y
    while grid.in_bounds(x, y):
        c = grid.get(x, y)
        if c.type == "DEF":
            break
        if c.type == "LETTER" and c.letter:
            letters.append(c.letter)
        else:
            letters.append(".")
        if direction == "RIGHT":
            x += 1
        else:
            y += 1
    pattern = "".join(letters)
    return Slot(start_x=start_x, start_y=start_y, direction=direction, length=len(pattern), pattern=pattern)


def segment_pattern(grid: Grid, x: int, y: int, direction: str) -> Tuple[int, int, str]:
    if direction == "RIGHT":
        sx = x
        while sx - 1 >= 0 and grid.get(sx - 1, y).type != "DEF":
            sx -= 1
        ex = sx
        while ex < grid.width and grid.get(ex, y).type != "DEF":
            ex += 1
        letters = []
        for xi in range(sx, ex):
            c = grid.get(xi, y)
            letters.append(c.letter if c.type == "LETTER" and c.letter else ".")
        return sx, y, "".join(letters)
    else:
        sy = y
        while sy - 1 >= 0 and grid.get(x, sy - 1).type != "DEF":
            sy -= 1
        ey = sy
        while ey < grid.height and grid.get(x, ey).type != "DEF":
            ey += 1
        letters = []
        for yi in range(sy, ey):
            c = grid.get(x, yi)
            letters.append(c.letter if c.type == "LETTER" and c.letter else ".")
        return x, sy, "".join(letters)


def validate_segments_with_dict(
    grid: Grid, dict_index: DictionaryIndex, changed: List[Tuple[int, int]], min_fixed: int = 2
) -> bool:
    seen = set()
    for x, y in changed:
        for direction in ("RIGHT", "DOWN"):
            sx, sy, pattern = segment_pattern(grid, x, y, direction)
            key = (sx, sy, direction)
            if key in seen:
                continue
            seen.add(key)
            if len(pattern) < 2:
                return False
            if "." in pattern:
                fixed = sum(1 for ch in pattern if ch != ".")
                if fixed >= min_fixed:
                    if not dict_index.matches_pattern(pattern):
                        return False
            else:
                if pattern not in dict_index.words:
                    return False
    return True


def adjacent_pairs_ok(grid: Grid, dict_index: DictionaryIndex, changed: List[Tuple[int, int]]) -> bool:
    seen = set()
    for x, y in changed:
        for direction in ("RIGHT", "DOWN"):
            sx, sy, pattern = segment_pattern(grid, x, y, direction)
            key = (sx, sy, direction)
            if key in seen:
                continue
            seen.add(key)
            if len(pattern) < 2:
                continue
            for i in range(len(pattern) - 1):
                a, b = pattern[i], pattern[i + 1]
                if a != "." and b != ".":
                    if a + b not in dict_index.bigrams:
                        return False
    return True


def perpendicular_ok(grid: Grid, dict_index: DictionaryIndex, changed: List[Tuple[int, int]], slot_dir: str) -> bool:
    perp = "DOWN" if slot_dir == "RIGHT" else "RIGHT"
    seen = set()
    for x, y in changed:
        sx, sy, pattern = segment_pattern(grid, x, y, perp)
        key = (sx, sy, perp)
        if key in seen:
            continue
        seen.add(key)
        if len(pattern) < 2:
            return False
        if "." not in pattern:
            if pattern not in dict_index.words:
                return False
    return True


def try_place_word_progressive(
    grid: Grid,
    slot: Slot,
    word: str,
    dict_index: DictionaryIndex,
    predefs: Set[Tuple[int, int]],
    check_perpendicular: bool = True,
    relax_end_def_checks: bool = False,
    check_patterns: bool = True,
    min_fixed_for_pattern: int = 1,
) -> Optional[Tuple[WordPlacement, Optional[Tuple[int, int]]]]:
    # Check fit and conflicts
    if slot.length < len(word):
        return None
    x, y = slot.start_x, slot.start_y
    for ch in word:
        c = grid.get(x, y)
        if c.type == "DEF":
            return None
        if c.letter is not None and c.letter != ch:
            return None
        if slot.direction == "RIGHT":
            x += 1
        else:
            y += 1

    added_def = None
    # Cut slot if word is shorter
    if len(word) < slot.length:
        end_x = slot.start_x + (len(word) if slot.direction == "RIGHT" else 0)
        end_y = slot.start_y + (len(word) if slot.direction == "DOWN" else 0)
        if not grid.in_bounds(end_x, end_y):
            return None
        end_cell = grid.get(end_x, end_y)
        if end_cell.type is not None:
            return None
        grid.set_def(end_x, end_y)
        dirs = possible_def_dirs(grid, end_x, end_y)
        if not dirs:
            end_cell.type = None
            end_cell.defs = []
            return None
        end_cell.defs = [DefEntry(direction=dirs[0])]
        if not relax_end_def_checks:
            if def_run_too_long(grid, set()) or has_singleton_segments(grid):
                end_cell.type = None
                end_cell.defs = []
                return None
        added_def = (end_x, end_y)
        predefs.add(added_def)

    changed = place_word_letters(grid, slot, word)
    coords = [(cx, cy) for cx, cy, _ in changed]
    if check_perpendicular and not perpendicular_ok(grid, dict_index, coords, slot.direction):
        revert_letters(grid, changed)
        if added_def:
            ax, ay = added_def
            predefs.discard(added_def)
            c = grid.get(ax, ay)
            c.type = None
            c.defs = []
        return None

    placement = WordPlacement(word=word, start_x=slot.start_x, start_y=slot.start_y, direction=slot.direction)
    return placement, added_def


def place_word_letters(grid: Grid, slot: Slot, word: str) -> List[Tuple[int, int, Optional[str]]]:
    changed: List[Tuple[int, int, Optional[str]]] = []
    x, y = slot.start_x, slot.start_y
    for ch in word:
        c = grid.get(x, y)
        if c.letter is None:
            changed.append((x, y, None))
            grid.set_letter(x, y, ch)
        if slot.direction == "RIGHT":
            x += 1
        else:
            y += 1
    return changed


def revert_letters(grid: Grid, changed: List[Tuple[int, int, Optional[str]]]) -> None:
    for x, y, prev in changed:
        c = grid.get(x, y)
        c.letter = prev
        if prev is None and c.type == "LETTER":
            c.type = None


def list_slots(grid: Grid) -> List[Slot]:
    slots: List[Slot] = []
    seen = set()
    # From defs
    for y in range(grid.height):
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type != "DEF":
                continue
            for d in c.defs:
                if d.direction == "RIGHT":
                    sx, sy, direction = x + 1, y, "RIGHT"
                elif d.direction == "DOWN":
                    sx, sy, direction = x, y + 1, "DOWN"
                else:
                    continue
                if not grid.in_bounds(sx, sy) or grid.get(sx, sy).type == "DEF":
                    continue
                slot = slot_from(grid, sx, sy, direction)
                if slot.length < 2:
                    continue
                key = (slot.start_x, slot.start_y, slot.direction)
                if key in seen:
                    continue
                seen.add(key)
                slots.append(slot)
    # Border-start slots
    for y in range(grid.height):
        if grid.get(0, y).type != "DEF":
            slot = slot_from(grid, 0, y, "RIGHT")
            if slot.length >= 2:
                key = (slot.start_x, slot.start_y, slot.direction)
                if key not in seen:
                    seen.add(key)
                    slots.append(slot)
    for x in range(grid.width):
        if grid.get(x, 0).type != "DEF":
            slot = slot_from(grid, x, 0, "DOWN")
            if slot.length >= 2:
                key = (slot.start_x, slot.start_y, slot.direction)
                if key not in seen:
                    seen.add(key)
                    slots.append(slot)
    return slots


def apply_word_to_grid(grid: Grid, slot: Slot, word: str) -> None:
    if len(word) != slot.length:
        raise ValueError("Word length mismatch")
    x, y = slot.start_x, slot.start_y
    for ch in word:
        grid.set_letter(x, y, ch)
        if slot.direction == "RIGHT":
            x += 1
        else:
            y += 1


def validate_partial_structure(grid: Grid) -> None:
    # Only structural checks, allow empty LETTERs
    if def_run_too_long(grid, set()):
        raise ValueError("More than 2 adjacent DEF cells in a row/column")
    if has_singleton_segments(grid):
        loc = find_singleton_segment(grid)
        if loc:
            raise ValueError(f"1-letter segment exists at {loc[0]} {loc[1]},{loc[2]}")
        raise ValueError("1-letter segment exists")
    for y in range(grid.height):
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type == "DEF":
                for d in c.defs:
                    if d.direction == "RIGHT":
                        sx, sy, direction = x + 1, y, "RIGHT"
                    elif d.direction == "DOWN":
                        sx, sy, direction = x, y + 1, "DOWN"
                    else:
                        continue
                    if not grid.in_bounds(sx, sy) or grid.get(sx, sy).type == "DEF":
                        raise ValueError("DEF points to invalid cell")
                    slot = slot_from(grid, sx, sy, direction)
                    if slot.length < 2:
                        raise ValueError("Word length <2")


def possible_def_dirs(grid: Grid, x: int, y: int) -> List[str]:
    dirs = []
    if grid.in_bounds(x + 1, y) and grid.get(x + 1, y).type != "DEF":
        slot = slot_from(grid, x + 1, y, "RIGHT")
        if slot.length >= 2:
            dirs.append("RIGHT")
    if grid.in_bounds(x, y + 1) and grid.get(x, y + 1).type != "DEF":
        slot = slot_from(grid, x, y + 1, "DOWN")
        if slot.length >= 2:
            dirs.append("DOWN")
    return dirs


def apply_border_def_pattern(grid: Grid, seed: int = 0) -> None:
    """Apply alternating DEF pattern on row 0 and column 0 starting at (0,0)."""
    rng = random.Random(seed)
    targets = set()
    for x in range(0, grid.width, 2):
        targets.add((x, 0))
    for y in range(0, grid.height, 2):
        targets.add((0, y))
    for x, y in sorted(targets):
        c = grid.get(x, y)
        if c.type == "LETTER":
            continue
        grid.set_def(x, y)
        # Row 0: always RIGHT_DOWN + DOWN (never RIGHT alone)
        if y == 0:
            chosen = ["RIGHT_DOWN", "DOWN"]
            grid.get(x, y).defs = [DefEntry(direction=d) for d in chosen]
            continue
        # Col 0: always RIGHT, sometimes +RIGHT_DOWN, never DOWN
        if x == 0:
            chosen = ["RIGHT"]
            if rng.random() < 0.7:
                chosen = ["RIGHT", "RIGHT_DOWN"]
            grid.get(x, y).defs = [DefEntry(direction=d) for d in chosen]
            continue


def _points_from_def(grid: Grid, x: int, y: int, direction: str) -> bool:
    c = grid.get(x, y)
    if c.type != "DEF":
        return False
    for d in c.defs or []:
        if d.direction == direction:
            return True
    return False


def has_singleton_segments(grid: Grid) -> bool:
    # Check rows
    for y in range(grid.height):
        x = 0
        while x < grid.width:
            while x < grid.width and grid.get(x, y).type == "DEF":
                x += 1
            start = x
            while x < grid.width and grid.get(x, y).type != "DEF":
                x += 1
            length = x - start
            if length == 1:
                return True
    # Check columns
    for x in range(grid.width):
        y = 0
        while y < grid.height:
            while y < grid.height and grid.get(x, y).type == "DEF":
                y += 1
            start = y
            while y < grid.height and grid.get(x, y).type != "DEF":
                y += 1
            length = y - start
            if length == 1:
                return True
    return False


def find_singleton_segment(grid: Grid) -> Optional[Tuple[str, int, int]]:
    # Returns (direction, x, y) for a 1-letter segment start
    for y in range(grid.height):
        x = 0
        while x < grid.width:
            while x < grid.width and grid.get(x, y).type == "DEF":
                x += 1
            start = x
            while x < grid.width and grid.get(x, y).type != "DEF":
                x += 1
            length = x - start
            if length == 1:
                return ("RIGHT", start, y)
    for x in range(grid.width):
        y = 0
        while y < grid.height:
            while y < grid.height and grid.get(x, y).type == "DEF":
                y += 1
            start = y
            while y < grid.height and grid.get(x, y).type != "DEF":
                y += 1
            length = y - start
            if length == 1:
                return ("DOWN", x, start)
    return None


def auto_place_defs(
    grid: Grid,
    seed: int = 0,
    target_def_ratio: float = 0.22,
    max_retries: int = 10,
) -> None:
    rng = random.Random(seed)
    fixed = {(x, y) for y in range(grid.height) for x in range(grid.width) if grid.get(x, y).type is not None}
    empties = [(x, y) for y in range(grid.height) for x in range(grid.width) if (x, y) not in fixed]
    if not empties:
        return
    target_defs = int(len(empties) * target_def_ratio)

    def reset_placed() -> None:
        for x, y in empties:
            c = grid.get(x, y)
            if (x, y) not in fixed:
                c.type = None
                c.letter = None
                c.defs = []

    for _ in range(max_retries):
        reset_placed()
        candidates = empties[:]
        rng.shuffle(candidates)
        placed = 0
        logger.info("DEF pass: target_defs=%d candidates=%d", target_defs, len(candidates))
        for x, y in candidates:
            if placed >= target_defs:
                break
            if grid.get(x, y).type is not None:
                continue
            # Do not invalidate existing DEF starts
            if grid.in_bounds(x - 1, y) and _points_from_def(grid, x - 1, y, "RIGHT"):
                continue
            if grid.in_bounds(x, y - 1) and _points_from_def(grid, x, y - 1, "DOWN"):
                continue
            grid.set_def(x, y)
            dirs = possible_def_dirs(grid, x, y)
            if not dirs:
                grid.get(x, y).type = None
                grid.get(x, y).defs = []
                continue
            chosen = [dirs[0]]
            if len(dirs) == 2 and rng.random() < 0.25:
                chosen = dirs
            grid.get(x, y).defs = [DefEntry(direction=d) for d in chosen]
            if def_run_too_long(grid, set()) or has_singleton_segments(grid):
                grid.get(x, y).type = None
                grid.get(x, y).defs = []
                continue
            placed += 1
        try:
            validate_partial_structure(grid)
            if placed >= int(target_defs * 0.7):
                logger.info("DEF pass success: placed=%d", placed)
                return
        except Exception as e:
            logger.warning("DEF pass failed: placed=%d err=%s", placed, e)
            continue
    # Final attempt failed
    validate_partial_structure(grid)


def propose_word_llm(
    pattern: str,
    api_key: str,
    model: str,
) -> Optional[str]:
    system = (
        "You suggest a single French word that fits a pattern. "
        "Use uppercase A-Z only, no accents. Return JSON {\"word\":\"...\"}."
    )
    user = (
        "Pattern uses '.' for unknowns. "
        "Give a word that fits exactly and is easy but not trivial. "
        f"Pattern: {pattern}"
    )
    data = call_openai_json(api_key, model, system, user)
    w = str(data.get("word", "")).strip().upper()
    if w and all("A" <= ch <= "Z" for ch in w):
        return w
    return None


def build_llm_prompt(
    grid: Grid,
    placements: List[WordPlacement],
    reserved_defs: Set[Tuple[int, int]],
    mandatory_words: List[str],
    unplaced_words: List[str],
) -> Tuple[str, str]:
    system = (
        "You are generating a complete French-style arrowword grid. "
        "Return ONLY valid JSON. All letters must be A-Z. "
        "No black cells. All cells must be LETTER or DEF. "
        "No more than 2 adjacent DEF cells in any row or column. "
        "Each DEF has at least one direction among RIGHT, DOWN, RIGHT_DOWN. "
        "Each DEF direction must point to a real word of length >=2. "
        "Word starts at border OR after a DEF pointing to it. "
        "Words end at border or just before a DEF. "
        "Every LETTER belongs to at least one word. No 1-letter words." 
        "Mandatory words must appear exactly once each. "
        "All fixed letters and reserved DEF cells must be preserved." 
    )

    compact = grid_compact_repr(grid)

    user = (
        "Build full grid with cells and defs. "
        "Return JSON with a single field 'grid_compact' that matches the compact format. "
        "Compact format: 21 lines, each with 13 tokens separated by ' | '. "
        "Each token is either a single letter A-Z, or DEF with directions: "
        "DEF(R), DEF(D), DEF(RD), or multiple like DEF(R,D) or DEF(R,RD). "
        "No '.' placeholders in final output.\n\n"
        "Current grid (compact) for reference:\n"
        + compact
    )

    return system, user


def llm_attempt(
    grid: Grid,
    placements: List[WordPlacement],
    reserved_defs: Set[Tuple[int, int]],
    mandatory_words: List[str],
    unplaced_words: List[str],
    api_key: str,
    model: str,
) -> Tuple[Optional[Grid], Dict, Optional[str]]:
    system, user = build_llm_prompt(grid, placements, reserved_defs, mandatory_words, unplaced_words)
    data = call_openai_json(api_key, model, system, user)
    try:
        if "grid_compact" in data and isinstance(data["grid_compact"], str):
            grid_out = parse_compact_grid(data["grid_compact"])
        else:
            grid_out = grid_from_json(data)
        validate_grid(grid_out, mandatory_words, placements, reserved_defs, grid)
        return grid_out, data, None
    except Exception as e:
        return None, data, str(e)


def fill_with_llm(grid: Grid, placements: List[WordPlacement], reserved_defs: Set[Tuple[int, int]],
                  mandatory_words: List[str], unplaced_words: List[str], api_key: str, model: str,
                  max_attempts: int = 10, seed: int = 0) -> Grid:
    system, user = build_llm_prompt(grid, placements, reserved_defs, mandatory_words, unplaced_words)

    rng = random.Random(seed)
    for attempt in range(1, max_attempts + 1):
        data = call_openai_json(api_key, model, system, user)
        try:
            grid_out = grid_from_json(data)
            validate_grid(grid_out, mandatory_words, placements, reserved_defs, grid)
            return grid_out
        except Exception as e:
            if attempt == max_attempts:
                raise RuntimeError(f"LLM failed to generate valid grid after {max_attempts} attempts: {e}")
            time.sleep(0.5)
    raise RuntimeError("LLM generation failed")


def grid_from_json(data: Dict) -> Grid:
    width = data.get("width")
    height = data.get("height")
    if width != WIDTH or height != HEIGHT:
        raise ValueError("Invalid dimensions in JSON")
    grid = Grid(width, height)
    cells = data.get("cells", [])
    if len(cells) != width * height:
        raise ValueError("Invalid cell count in JSON")
    seen = set()
    for cell in cells:
        x = cell["x"]
        y = cell["y"]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Cell out of bounds in JSON")
        if (x, y) in seen:
            raise ValueError("Duplicate cell in JSON")
        seen.add((x, y))
        c = grid.get(x, y)
        c.type = cell["type"]
        c.letter = cell.get("letter")
        defs = []
        for d in cell.get("defs", []) or []:
            defs.append(DefEntry(direction=d["direction"], clue_text=d.get("clue_text", "")))
        c.defs = defs
    return grid


def word_from(grid: Grid, start_x: int, start_y: int, direction: str) -> str:
    letters = []
    x, y = start_x, start_y
    while grid.in_bounds(x, y):
        c = grid.get(x, y)
        if c.type == "DEF":
            break
        if c.type != "LETTER" or not c.letter:
            break
        letters.append(c.letter)
        if direction == "RIGHT":
            x += 1
        else:
            y += 1
    return "".join(letters)


def validate_grid(grid: Grid, mandatory_words: List[str], placements: List[WordPlacement],
                  reserved_defs: Set[Tuple[int, int]], base_grid: Grid) -> None:
    # Basic validation
    if grid.width != WIDTH or grid.height != HEIGHT:
        raise ValueError("Invalid grid dimensions")

    # Validate cells
    for y in range(grid.height):
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type not in {"LETTER", "DEF"}:
                raise ValueError(f"Invalid cell type at {x},{y}")
            if c.type == "LETTER":
                if not c.letter or not c.letter.isalpha() or len(c.letter) != 1:
                    raise ValueError(f"Invalid letter at {x},{y}")
                if not c.letter.isupper():
                    raise ValueError(f"Letter not uppercase at {x},{y}")
            else:
                if not c.defs or any(d.direction not in DIRECTIONS for d in c.defs):
                    raise ValueError(f"Invalid defs at {x},{y}")

    # No more than 2 adjacent DEF cells in any row/column
    if def_run_too_long(grid, set()):
        raise ValueError("More than 2 adjacent DEF cells in a row/column")

    # Check reserved defs and fixed letters from base grid
    for (x, y) in reserved_defs:
        if grid.get(x, y).type != "DEF":
            raise ValueError(f"Reserved def cell not DEF at {x},{y}")
    for y in range(base_grid.height):
        for x in range(base_grid.width):
            bc = base_grid.get(x, y)
            if bc.type == "LETTER" and bc.letter:
                gc = grid.get(x, y)
                if gc.type != "LETTER" or gc.letter != bc.letter:
                    raise ValueError(f"Fixed letter mismatch at {x},{y}")

    # Build words based on defs and border starts
    words: List[Tuple[str, int, int, str]] = []
    def_starts: Set[Tuple[int, int, str]] = set()
    coverage: Dict[Tuple[int, int], int] = {}

    def add_word(w: str, sx: int, sy: int, direction: str) -> None:
        if len(w) < 2:
            raise ValueError(f"Word length <2 at {sx},{sy} {direction}")
        words.append((w, sx, sy, direction))
        for i in range(len(w)):
            x = sx + i if direction == "RIGHT" else sx
            y = sy if direction == "RIGHT" else sy + i
            coverage[(x, y)] = coverage.get((x, y), 0) + 1

    # From defs
    for y in range(grid.height):
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type != "DEF":
                continue
            for d in c.defs:
                if d.direction == "RIGHT":
                    sx, sy = x + 1, y
                    if not grid.in_bounds(sx, sy) or grid.get(sx, sy).type != "LETTER":
                        raise ValueError(f"DEF RIGHT points to invalid cell at {x},{y}")
                    w = word_from(grid, sx, sy, "RIGHT")
                    add_word(w, sx, sy, "RIGHT")
                    def_starts.add((sx, sy, "RIGHT"))
                elif d.direction == "DOWN":
                    sx, sy = x, y + 1
                    if not grid.in_bounds(sx, sy) or grid.get(sx, sy).type != "LETTER":
                        raise ValueError(f"DEF DOWN points to invalid cell at {x},{y}")
                    w = word_from(grid, sx, sy, "DOWN")
                    add_word(w, sx, sy, "DOWN")
                    def_starts.add((sx, sy, "DOWN"))
                elif d.direction == "RIGHT_DOWN":
                    sx, sy = x, y + 1
                    if not grid.in_bounds(sx, sy) or grid.get(sx, sy).type != "LETTER":
                        raise ValueError(f"DEF RIGHT_DOWN points to invalid cell at {x},{y}")
                    w = word_from(grid, sx, sy, "RIGHT")
                    add_word(w, sx, sy, "RIGHT")
                    def_starts.add((sx, sy, "RIGHT"))

    # Border-start words (RIGHT on left border, DOWN on top border)
    for y in range(grid.height):
        if grid.get(0, y).type == "LETTER" and (0, y, "RIGHT") not in def_starts:
            w = word_from(grid, 0, y, "RIGHT")
            add_word(w, 0, y, "RIGHT")
    for x in range(grid.width):
        if grid.get(x, 0).type == "LETTER" and (x, 0, "DOWN") not in def_starts:
            w = word_from(grid, x, 0, "DOWN")
            add_word(w, x, 0, "DOWN")

    # Validate coverage
    for y in range(grid.height):
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type == "LETTER":
                if (x, y) not in coverage:
                    raise ValueError(f"Letter not covered by any word at {x},{y}")
                if coverage[(x, y)] > 2:
                    raise ValueError(f"Letter belongs to >2 words at {x},{y}")

    # Mandatory words exactly once
    counts: Dict[str, int] = {}
    for w, _, _, _ in words:
        counts[w] = counts.get(w, 0) + 1
    for mw in mandatory_words:
        if counts.get(mw, 0) != 1:
            raise ValueError(f"Mandatory word count != 1: {mw} -> {counts.get(mw, 0)}")

    # Verify placed words from step1 are preserved at same position/direction
    for p in placements:
        w = word_from(grid, p.start_x, p.start_y, p.direction)
        if w != p.word:
            raise ValueError(f"Placed word moved or altered: {p.word} at {p.start_x},{p.start_y} {p.direction}")


def generate_definitions_llm(api_key: str, model: str, words: List[str]) -> Dict[str, str]:
    system = (
        "You write short French crossword-style clues. "
        "No accent marks. Uppercase words. "
        "Return JSON mapping words to clues."
    )
    user = (
        "Create a concise French clue for each word. "
        "Do not repeat the word in the clue. "
        "Return JSON: {\"WORD\":\"CLUE\",...}.\n\n"
        + "Words:\n" + " ".join(words)
    )
    data = call_openai_json(api_key, model, system, user)
    # Ensure uppercase keys
    out = {}
    for k, v in data.items():
        out[k.upper()] = str(v)
    return out


def attach_definitions(grid: Grid, definitions: Dict[str, str]) -> None:
    # Map each def-direction to the word it defines
    for y in range(grid.height):
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type != "DEF":
                continue
            for d in c.defs:
                if d.direction == "RIGHT":
                    w = word_from(grid, x + 1, y, "RIGHT")
                elif d.direction == "DOWN":
                    w = word_from(grid, x, y + 1, "DOWN")
                else:
                    w = word_from(grid, x, y + 1, "RIGHT")
                d.clue_text = definitions.get(w, f"{w}")


def words_list(grid: Grid) -> List[Dict]:
    out = []
    seen = set()
    for y in range(grid.height):
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type != "DEF":
                continue
            for d in c.defs:
                if d.direction == "RIGHT":
                    sx, sy, direction = x + 1, y, "RIGHT"
                    w = word_from(grid, sx, sy, "RIGHT")
                elif d.direction == "DOWN":
                    sx, sy, direction = x, y + 1, "DOWN"
                    w = word_from(grid, sx, sy, "DOWN")
                else:
                    sx, sy, direction = x, y + 1, "RIGHT"
                    w = word_from(grid, sx, sy, "RIGHT")
                key = (sx, sy, direction)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"word": w, "start_x": sx, "start_y": sy, "direction": direction})
    # Border-start words
    for y in range(grid.height):
        if grid.get(0, y).type == "LETTER":
            key = (0, y, "RIGHT")
            if key in seen:
                continue
            w = word_from(grid, 0, y, "RIGHT")
            out.append({"word": w, "start_x": 0, "start_y": y, "direction": "RIGHT"})
    for x in range(grid.width):
        if grid.get(x, 0).type == "LETTER":
            key = (x, 0, "DOWN")
            if key in seen:
                continue
            w = word_from(grid, x, 0, "DOWN")
            out.append({"word": w, "start_x": x, "start_y": 0, "direction": "DOWN"})
    return out


def draw_pdf(grid: Grid, out_path: str, show_letters: bool) -> None:
    if canvas is None:
        raise RuntimeError("reportlab is not installed")
    c = canvas.Canvas(out_path, pagesize=A4)
    page_w, page_h = A4
    # Compute cell size with margins
    margin = 12 * mm
    cell_size = min((page_w - 2 * margin) / grid.width, (page_h - 2 * margin) / grid.height)
    origin_x = margin
    origin_y = page_h - margin - cell_size * grid.height

    c.setLineWidth(0.5)
    for y in range(grid.height):
        for x in range(grid.width):
            cx = origin_x + x * cell_size
            cy = origin_y + (grid.height - 1 - y) * cell_size
            c.rect(cx, cy, cell_size, cell_size)
            cell = grid.get(x, y)
            if cell.type == "DEF":
                # Draw small triangle arrows for each direction
                for d in cell.defs:
                    if d.direction == "RIGHT":
                        c.line(cx + 2, cy + cell_size / 2, cx + cell_size - 2, cy + cell_size / 2)
                        c.line(cx + cell_size - 4, cy + cell_size / 2 + 2, cx + cell_size - 2, cy + cell_size / 2)
                        c.line(cx + cell_size - 4, cy + cell_size / 2 - 2, cx + cell_size - 2, cy + cell_size / 2)
                    elif d.direction == "DOWN":
                        c.line(cx + cell_size / 2, cy + cell_size - 2, cx + cell_size / 2, cy + 2)
                        c.line(cx + cell_size / 2 - 2, cy + 4, cx + cell_size / 2, cy + 2)
                        c.line(cx + cell_size / 2 + 2, cy + 4, cx + cell_size / 2, cy + 2)
                    else:
                        # RIGHT_DOWN: draw a down then right corner
                        c.line(cx + cell_size / 2, cy + cell_size - 2, cx + cell_size / 2, cy + cell_size / 2)
                        c.line(cx + cell_size / 2, cy + cell_size / 2, cx + cell_size - 2, cy + cell_size / 2)
                # Clue text
                if cell.defs:
                    c.setFont("Helvetica", 5)
                    text = " / ".join([d.clue_text for d in cell.defs])
                    c.drawString(cx + 1, cy + 1, text[:30])
            else:
                if show_letters and cell.letter:
                    c.setFont("Helvetica-Bold", 9)
                    c.drawCentredString(cx + cell_size / 2, cy + cell_size / 2 - 3, cell.letter)
    c.showPage()
    c.save()


def main() -> None:
    if load_dotenv is not None:
        # Load .env from current working directory if present
        load_dotenv()
    else:
        # Minimal .env loader fallback
        load_env_fallback()
    parser = argparse.ArgumentParser(description="Mots fleches generator")
    parser.add_argument("--input", default="mots_oblig.txt", help="Mandatory words file")
    parser.add_argument("--outdir", default="out", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for fill and definitions")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key")
    parser.add_argument("--model", default="gpt-5", help="OpenAI model")
    parser.add_argument("--max-attempts", type=int, default=10, help="Max LLM attempts")
    args = parser.parse_args()

    words = load_mandatory_words(args.input)
    grid = Grid(WIDTH, HEIGHT)
    apply_border_def_pattern(grid, seed=args.seed)
    predefs = {(x, y) for y in range(HEIGHT) for x in range(WIDTH) if grid.get(x, y).type == "DEF"}

    placements, unplaced, reserved_defs = place_mandatory_words(
        grid, words, args.seed, predefs=predefs, min_parallel_gap=2, forbid_border_parallel_gap1=True
    )
    if len(placements) == 0:
        placements, unplaced, reserved_defs = place_mandatory_words(
            grid, words, args.seed, predefs=predefs, min_parallel_gap=1, forbid_border_parallel_gap1=False
        )

    # Late .env lookup if needed (cwd or script directory)
    if not args.api_key:
        for env_path in (".env", os.path.join(os.path.dirname(__file__), ".env")):
            data = _read_env_file(env_path)
            if data.get("OPENAI_API_KEY"):
                args.api_key = data["OPENAI_API_KEY"]
                os.environ["OPENAI_API_KEY"] = args.api_key
                break

    use_llm = args.use_llm or bool(args.api_key)
    if not use_llm:
        raise RuntimeError(
            "LLM required. Rerun with --use-llm and an API key, e.g.:\n"
            "python mots_fleches.py --use-llm --api-key YOUR_KEY"
        )
    if not args.api_key:
        raise RuntimeError("--use-llm requires --api-key or OPENAI_API_KEY")
    filled = fill_with_llm(
        grid,
        placements,
        reserved_defs,
        words,
        unplaced,
        args.api_key,
        args.model,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )

    # Validate with full mandatory list
    validate_grid(filled, words, placements, reserved_defs, grid)

    # Generate definitions
    if args.use_llm:
        # Use words defined by defs
        wl = words_list(filled)
        unique_words = sorted({w["word"] for w in wl})
        defs = generate_definitions_llm(args.api_key, args.model, unique_words)
        attach_definitions(filled, defs)

    # Outputs
    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, "grille.json")
    with open(json_path, "w", encoding="utf-8") as f:
        out = filled.to_json()
        out["words"] = words_list(filled)
        f.write(json.dumps(out, ensure_ascii=False, indent=2))

    if canvas is not None:
        draw_pdf(filled, os.path.join(args.outdir, "grille_vide.pdf"), show_letters=False)
        draw_pdf(filled, os.path.join(args.outdir, "grille_solution.pdf"), show_letters=True)


if __name__ == "__main__":
    main()
