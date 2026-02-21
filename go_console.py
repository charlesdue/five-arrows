import argparse
import copy
import random
import json
import os
import time

from mots_fleches import (
    Grid,
    WIDTH,
    HEIGHT,
    Slot,
    load_mandatory_words,
    load_dictionary,
    build_dictionary_index,
    apply_border_def_pattern,
    auto_place_defs,
    predef_slot_lengths,
    select_words_for_slots,
    list_slots,
    progressive_place_next,
    try_place_word_progressive,
    expand_def_directions,
    DefEntry,
    possible_def_dirs,
    slot_from,
    def_run_too_long,
    has_singleton_segments,
)


def _pattern_prefix_matches(word: str, pattern: str) -> bool:
    if len(word) > len(pattern):
        return False
    for ch, p in zip(word, pattern[: len(word)]):
        if p != "." and p != ch:
            return False
    return True


def _pick_word_for_slot(slot: Slot, dict_index, used_words: set, rng) -> str | None:
    max_len = slot.length
    lengths = list(range(2, max_len + 1))
    rng.shuffle(lengths)
    pick = None
    seen = 0
    for L in lengths:
        words = dict_index.by_length.get(L, [])
        if not words:
            continue
        start = rng.randrange(len(words))
        for k in range(len(words)):
            w = words[(start + k) % len(words)]
            if w in used_words:
                continue
            if _pattern_prefix_matches(w, slot.pattern):
                seen += 1
                # Reservoir sampling over all matching words
                if rng.randrange(seen) == 0:
                    pick = w
    return pick


def _pick_word_candidates_for_slot(
    slot: Slot,
    dict_index,
    used_words: set,
    rng,
    max_candidates: int = 6,
) -> list[str]:
    max_len = slot.length
    lengths = list(range(2, max_len + 1))
    rng.shuffle(lengths)
    sample: list[str] = []
    seen = 0
    for L in lengths:
        words = dict_index.by_length.get(L, [])
        if not words:
            continue
        start = rng.randrange(len(words))
        for k in range(len(words)):
            w = words[(start + k) % len(words)]
            if w in used_words:
                continue
            if _pattern_prefix_matches(w, slot.pattern):
                seen += 1
                if len(sample) < max_candidates:
                    sample.append(w)
                else:
                    j = rng.randrange(seen)
                    if j < max_candidates:
                        sample[j] = w
    rng.shuffle(sample)
    return sample


def _empty_score(grid: Grid, x: int, y: int) -> int:
    if not grid.in_bounds(x, y):
        return -1
    c = grid.get(x, y)
    if c.type is not None or c.letter is not None:
        return -1
    score = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            nx, ny = x + dx, y + dy
            if not grid.in_bounds(nx, ny):
                continue
            c2 = grid.get(nx, ny)
            if c2.type is None and c2.letter is None:
                score += 1
    return score


def _slot_end_on_border(slot: Slot, grid: Grid) -> bool:
    end_x = slot.start_x + (slot.length - 1 if slot.direction == "RIGHT" else 0)
    end_y = slot.start_y + (slot.length - 1 if slot.direction == "DOWN" else 0)
    return end_x == 0 or end_x == grid.width - 1 or end_y == 0 or end_y == grid.height - 1


def _weighted_choice(candidates: list[dict], rng: random.Random) -> dict | None:
    if not candidates:
        return None
    total = sum(c.get("_weight", 1.0) for c in candidates)
    if total <= 0:
        return rng.choice(candidates)
    r = rng.random() * total
    acc = 0.0
    for c in candidates:
        acc += c.get("_weight", 1.0)
        if r <= acc:
            return c
    return candidates[-1]


def _slot_priority(slots, rng):
    rng.shuffle(slots)
    slots.sort(key=lambda s: (s.length, 0 if s.direction == "DOWN" else 1, rng.random()))
    return slots


def _slot_has_empty(grid: Grid, slot: Slot) -> bool:
    x, y = slot.start_x, slot.start_y
    for _ in range(slot.length):
        c = grid.get(x, y)
        if c.letter is None:
            return True
        if slot.direction == "RIGHT":
            x += 1
        else:
            y += 1
    return False


def _print_place(step: int, word: str, mandatory: bool, slot: Slot, added_def, log_fn) -> None:
    label = "oblig" if mandatory else "fill"
    extra = f" +DEF ({added_def[0]},{added_def[1]})" if added_def else ""
    log_fn(
        f"[{step:03d}] {label} {word} {slot.direction} @ ({slot.start_x},{slot.start_y}) "
        f"len={slot.length} pattern={slot.pattern}{extra}"
    )


STATE_PATH = "state.json"


def run(
    words_path: str,
    dict_path: str,
    seed: int = -1,
    max_mandatory: int = 5,
    max_steps: int = 500,
    log_fn=None,
    grid_cb=None,
) -> None:
    log_buffer: list[str] = []

    base_log = log_fn
    if base_log is None:
        def base_log(msg: str) -> None:
            print(msg, flush=True)

    if seed is None or seed < 0:
        seed = int(time.time() * 1000) & 0xFFFFFFFF

    mandatory_words = load_mandatory_words(words_path)
    dictionary = load_dictionary(dict_path)
    dict_words = dictionary.union(mandatory_words)
    dict_index = build_dictionary_index(list(dict_words))
    last_letter_freq: dict[str, int] = {}
    for w in dict_words:
        if not w:
            continue
        last_letter_freq[w[-1]] = last_letter_freq.get(w[-1], 0) + 1
    max_last_freq = max(last_letter_freq.values()) if last_letter_freq else 0

    def _end_letter_weight(word: str) -> float:
        if not word or max_last_freq <= 0:
            return 1.0
        freq = last_letter_freq.get(word[-1], 0)
        # Scale to [0.25, 1.25]
        return 0.25 + (freq / max_last_freq)

    base_grid = Grid(WIDTH, HEIGHT)
    apply_border_def_pattern(base_grid, seed=seed)
    grid = base_grid
    for i in range(8):
        trial = copy.deepcopy(base_grid)
        try:
            auto_place_defs(trial, seed=seed + i)
            grid = trial
            break
        except Exception:
            continue

    predefs = {(x, y) for y in range(HEIGHT) for x in range(WIDTH) if grid.get(x, y).type == "DEF"}
    slot_lengths = predef_slot_lengths(grid, predefs)
    queue = select_words_for_slots(mandatory_words, slot_lengths, max_mandatory, seed=seed)
    random.Random(seed ^ 0xA5A5A5A5).shuffle(queue)

    queue_index = 0
    used_slots: set[tuple[int, int, str]] = set()
    used_words: set[str] = set()
    unplaced: list[str] = []
    word_attempt = 0
    def_retry = 0
    steps = 0
    stagnation = 0
    max_stagnation = 200

    rng = random.Random(1000 + seed)
    placements: list[dict] = []
    cell_use_count: dict[tuple[int, int], int] = {}

    def write_state() -> None:
        try:
            payload = {"grid": grid.to_json(), "logs": log_buffer, "placements": placements[-200:]}
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            pass

    def log(msg: str) -> None:
        base_log(msg)
        log_buffer.append(str(msg))
        if len(log_buffer) > 2000:
            del log_buffer[:1000]
        write_state()

    def emit_grid() -> None:
        if grid_cb:
            grid_cb(grid)
        write_state()

    def _iter_word_cells(start_x: int, start_y: int, direction: str, length: int):
        x, y = start_x, start_y
        for _ in range(length):
            yield x, y
            if direction == "RIGHT":
                x += 1
            else:
                y += 1

    def _track_place_on(
        grid_obj: Grid,
        use_map: dict[tuple[int, int], int],
        placement_list: list[dict],
        word: str,
        slot: Slot,
        mandatory: bool,
    ) -> None:
        for x, y in _iter_word_cells(slot.start_x, slot.start_y, slot.direction, len(word)):
            use_map[(x, y)] = use_map.get((x, y), 0) + 1
        placement_list.append(
            {
                "word": word,
                "start_x": slot.start_x,
                "start_y": slot.start_y,
                "direction": slot.direction,
                "length": len(word),
                "mandatory": bool(mandatory),
            }
        )

    def _placement_covers_cell(p: dict, x: int, y: int) -> bool:
        if p["direction"] == "RIGHT":
            if y != p["start_y"]:
                return False
            return p["start_x"] <= x < p["start_x"] + p["length"]
        if x != p["start_x"]:
            return False
        return p["start_y"] <= y < p["start_y"] + p["length"]

    def _slot_cells(slot: Slot, word_len: int | None = None):
        n = slot.length if word_len is None else word_len
        return list(_iter_word_cells(slot.start_x, slot.start_y, slot.direction, n))

    def _clear_letter_if_unused(grid_obj: Grid, use_map: dict[tuple[int, int], int], x: int, y: int) -> None:
        if use_map.get((x, y), 0) != 0:
            return
        c = grid_obj.get(x, y)
        c.letter = None
        if c.type == "LETTER":
            c.type = None

    def _remove_placement_at_index(
        grid_obj: Grid,
        use_map: dict[tuple[int, int], int],
        placement_list: list[dict],
        idx: int,
    ) -> dict:
        p = placement_list.pop(idx)
        for x, y in _iter_word_cells(p["start_x"], p["start_y"], p["direction"], p["length"]):
            cur = use_map.get((x, y), 0)
            if cur <= 1:
                use_map.pop((x, y), None)
            else:
                use_map[(x, y)] = cur - 1
            _clear_letter_if_unused(grid_obj, use_map, x, y)
        return p

    def _pattern_matches_exact(word: str, pattern: str) -> bool:
        if len(word) != len(pattern):
            return False
        for ch, pch in zip(word, pattern):
            if pch != "." and pch != ch:
                return False
        return True

    def _sample_words_for_exact_pattern(
        length: int,
        pattern: str,
        used: set[str],
        rng_obj: random.Random,
        max_candidates: int = 50,
        scan_cap: int = 4000,
    ) -> list[str]:
        words = dict_index.by_length.get(length, [])
        if not words:
            return []
        out: list[str] = []
        start = rng_obj.randrange(len(words))
        scanned = 0
        for k in range(len(words)):
            if scanned >= scan_cap or len(out) >= max_candidates:
                break
            scanned += 1
            w = words[(start + k) % len(words)]
            if w in used:
                continue
            if _pattern_matches_exact(w, pattern):
                out.append(w)
        rng_obj.shuffle(out)
        return out

    def _try_add_def_at(x: int, y: int, check_perp: bool):
        if not grid.in_bounds(x, y):
            return None
        c0 = grid.get(x, y)
        if c0.type is not None or c0.letter is not None:
            return None
        possible = possible_def_dirs(grid, x, y)
        # Allow RIGHT_DOWN (word to the right starting below)
        if grid.in_bounds(x, y + 1) and grid.get(x, y + 1).type != "DEF":
            slot_rd = slot_from(grid, x, y + 1, "RIGHT")
            if slot_rd.length >= 2 and "RIGHT_DOWN" not in possible:
                possible.append("RIGHT_DOWN")
        if not possible:
            return None
        dir_sets: list[list[str]] = []
        if "RIGHT" in possible and "DOWN" in possible:
            dir_sets.append(["RIGHT", "DOWN"])
        if "RIGHT" in possible and "RIGHT_DOWN" in possible:
            dir_sets.append(["RIGHT", "RIGHT_DOWN"])
        if "DOWN" in possible and "RIGHT_DOWN" in possible:
            dir_sets.append(["DOWN", "RIGHT_DOWN"])
        if "RIGHT" in possible:
            dir_sets.append(["RIGHT"])
        if "DOWN" in possible:
            dir_sets.append(["DOWN"])
        if "RIGHT_DOWN" in possible:
            dir_sets.append(["RIGHT_DOWN"])
        rng.shuffle(dir_sets)
        for dirs in dir_sets:
            trial = copy.deepcopy(grid)
            t_predefs = set(predefs)
            t_used_slots = set(used_slots)
            t_used_words = set(used_words)
            t_placements = [p.copy() for p in placements]
            t_use = dict(cell_use_count)
            cell = trial.get(x, y)
            cell.type = "DEF"
            cell.letter = None
            cell.defs = [DefEntry(direction=d) for d in dirs]
            if def_run_too_long(trial, set()) or has_singleton_segments(trial):
                continue
            t_predefs.add((x, y))
            placed_words = []
            ok = True
            for d in dirs:
                if d == "RIGHT":
                    sx, sy, sdir = x + 1, y, "RIGHT"
                elif d == "DOWN":
                    sx, sy, sdir = x, y + 1, "DOWN"
                else:  # RIGHT_DOWN
                    sx, sy, sdir = x, y + 1, "RIGHT"
                if not trial.in_bounds(sx, sy) or trial.get(sx, sy).type == "DEF":
                    ok = False
                    break
                slot = slot_from(trial, sx, sy, sdir)
                if slot.length < 2:
                    ok = False
                    break
                candidates = _pick_word_candidates_for_slot(
                    slot, dict_index, t_used_words, rng, max_candidates=12
                )
                if candidates:
                    end_on_border = _slot_end_on_border(slot, trial)
                    candidates.sort(
                        key=lambda w: (
                            -_end_letter_weight(w) if end_on_border and len(w) == slot.length else -1.0,
                            rng.random(),
                        )
                    )
                placed = False
                for w in candidates:
                    res = try_place_word_progressive(
                        trial,
                        slot,
                        w,
                        dict_index,
                        t_predefs,
                        check_perpendicular=check_perp,
                        relax_end_def_checks=True,
                    )
                    if res:
                        _placement, added_def = res
                        t_used_words.add(w)
                        t_used_slots.add((slot.start_x, slot.start_y, slot.direction))
                        _track_place_on(trial, t_use, t_placements, w, slot, mandatory=False)
                        placed_words.append((w, slot, added_def))
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if not ok:
                continue
            if def_run_too_long(trial, set()) or has_singleton_segments(trial):
                continue
            return trial, t_predefs, t_used_slots, t_used_words, placed_words, t_placements, t_use
        return None

    def _inject_defs_with_words(check_perp: bool) -> int:
        nonlocal grid, predefs, used_slots, used_words, steps, placements, cell_use_count
        target = rng.randint(2, 7)
        log(f"INJECT: target={target} (check_perp={check_perp})")
        candidates = []
        for yy in range(HEIGHT):
            for xx in range(WIDTH):
                score = _empty_score(grid, xx, yy)
                if score >= 0:
                    candidates.append((score, rng.random(), xx, yy))
        candidates.sort(key=lambda t: (-t[0], t[1]))
        added_defs = 0
        attempts = 0
        for _score, _r, xx, yy in candidates:
            if added_defs >= target:
                break
            attempts += 1
            res = _try_add_def_at(xx, yy, check_perp)
            if not res:
                continue
            grid, predefs, used_slots, used_words, placed_words, placements, cell_use_count = res
            dirs = ",".join([d.direction for d in grid.get(xx, yy).defs])
            detail = []
            for w, slot, added_def in placed_words:
                detail.append(f"{w}@({slot.start_x},{slot.start_y}){slot.direction}")
            log(f"INJECT OK ({xx},{yy}) dirs={dirs} words={'; '.join(detail)}")
            for w, slot, added_def in placed_words:
                steps += 1
                _print_place(steps, w, False, slot, added_def, log)
            emit_grid()
            added_defs += 1
        if added_defs == 0:
            log(f"INJECT: no placement found after {attempts} attempts")
        else:
            log(f"INJECT: done added={added_defs} attempts={attempts}")
        return added_defs

    def _fill_unfilled_defs(max_passes: int = 2) -> int:
        nonlocal grid, predefs, used_slots, used_words, steps
        placed_total = 0
        for p in range(max_passes):
            progress = 0
            log(f"DEF-FILL pass={p + 1}")
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    c = grid.get(x, y)
                    if c.type != "DEF":
                        continue
                    for d in (c.defs or []):
                        if d.direction == "RIGHT":
                            sx, sy, sdir = x + 1, y, "RIGHT"
                        elif d.direction == "DOWN":
                            sx, sy, sdir = x, y + 1, "DOWN"
                        else:  # RIGHT_DOWN
                            sx, sy, sdir = x, y + 1, "RIGHT"
                        if not grid.in_bounds(sx, sy) or grid.get(sx, sy).type == "DEF":
                            continue
                        slot = slot_from(grid, sx, sy, sdir)
                        if slot.length < 2:
                            continue
                        key = (slot.start_x, slot.start_y, slot.direction)
                        if key in used_slots and not _slot_has_empty(grid, slot):
                            continue
                        log(
                            f"DEFCHK ({x},{y}) dir={d.direction} "
                            f"slot=({slot.start_x},{slot.start_y}){slot.direction} "
                            f"len={slot.length} pattern={slot.pattern}"
                        )
                        if not _slot_has_empty(grid, slot):
                            continue
                        candidates = _pick_word_candidates_for_slot(
                            slot, dict_index, used_words, rng, max_candidates=12
                        )
                        if not candidates:
                            continue
                        end_on_border = _slot_end_on_border(slot, grid)
                        candidates.sort(
                            key=lambda w: (
                                -_end_letter_weight(w) if end_on_border and len(w) == slot.length else -1.0,
                                rng.random(),
                            )
                        )
                        placed = False
                        for w in candidates:
                            res = try_place_word_progressive(
                                grid,
                                slot,
                                w,
                                dict_index,
                                predefs,
                                check_perpendicular=True,
                                relax_end_def_checks=True,
                            )
                            if not res:
                                continue
                            _placement, added_def = res
                            used_words.add(w)
                            used_slots.add((slot.start_x, slot.start_y, slot.direction))
                            _track_place_on(grid, cell_use_count, placements, w, slot, mandatory=False)
                            steps += 1
                            _print_place(steps, w, False, slot, added_def, log)
                            emit_grid()
                            progress += 1
                            placed_total += 1
                            placed = True
                            break
                        if placed:
                            continue
            if progress == 0:
                break
        return placed_total

    def _repair_unfilled_def_dirs(
        max_passes: int = 3,
        max_def_evals: int = 500,
        max_cross_words: int = 5,
        max_replace_candidates: int = 50,
    ) -> tuple[int, int]:
        nonlocal grid, predefs, used_slots, used_words, steps, placements, cell_use_count

        placed_total = 0
        replaced_total = 0

        def _slot_from_def_on(grid_obj: Grid, x: int, y: int, def_dir: str) -> Slot | None:
            if def_dir == "RIGHT":
                sx, sy, sdir = x + 1, y, "RIGHT"
            elif def_dir == "DOWN":
                sx, sy, sdir = x, y + 1, "DOWN"
            else:  # RIGHT_DOWN
                sx, sy, sdir = x, y + 1, "RIGHT"
            if not grid_obj.in_bounds(sx, sy) or grid_obj.get(sx, sy).type == "DEF":
                return None
            slot = slot_from(grid_obj, sx, sy, sdir)
            if slot.length < 2:
                return None
            return slot

        def _locked_pattern_for_placement(p: dict) -> str:
            letters: list[str] = []
            for x, y in _iter_word_cells(p["start_x"], p["start_y"], p["direction"], p["length"]):
                if cell_use_count.get((x, y), 0) > 1:
                    letters.append(grid.get(x, y).letter or ".")
                else:
                    letters.append(".")
            return "".join(letters)

        def _crossing_placement_indices_for_slot(target_slot: Slot) -> list[int]:
            perp = "DOWN" if target_slot.direction == "RIGHT" else "RIGHT"
            idxs: set[int] = set()
            for x, y in _slot_cells(target_slot):
                if grid.get(x, y).letter is None:
                    continue
                for idx, p in enumerate(placements):
                    if p["direction"] != perp:
                        continue
                    if _placement_covers_cell(p, x, y):
                        idxs.add(idx)
            out = list(idxs)
            rng.shuffle(out)
            return out

        def _word_still_used(word: str, plist: list[dict]) -> bool:
            for p in plist:
                if p["word"] == word:
                    return True
            return False

        for p in range(max_passes):
            progress = 0
            log(f"REPAIR start pass={p + 1}")

            targets = []
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    c = grid.get(x, y)
                    if c.type != "DEF":
                        continue
                    for d in (c.defs or []):
                        slot = _slot_from_def_on(grid, x, y, d.direction)
                        if not slot:
                            continue
                        empties = slot.pattern.count(".")
                        if empties <= 0:
                            continue
                        bonus = _empty_score(grid, slot.start_x, slot.start_y)
                        targets.append((empties, slot.length, bonus, rng.random(), x, y, d.direction, slot))

            targets.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3]))

            evals = 0
            for empties, _len, _bonus, _r, dx, dy, def_dir, slot in targets:
                if steps >= max_steps:
                    break
                evals += 1
                if evals > max_def_evals:
                    break

                # Refresh slot pattern in case grid changed since target collection
                slot = _slot_from_def_on(grid, dx, dy, def_dir)
                if not slot or slot.pattern.count(".") <= 0:
                    continue

                log(
                    f"REPAIR DEFCHK ({dx},{dy}) dir={def_dir} "
                    f"slot=({slot.start_x},{slot.start_y}){slot.direction} "
                    f"len={slot.length} empties={empties} pattern={slot.pattern}"
                )

                # A) direct fill attempt
                direct_candidates = _pick_word_candidates_for_slot(
                    slot, dict_index, used_words, rng, max_candidates=40
                )
                placed = False
                if not direct_candidates:
                    log("REPAIR direct: no candidates")
                else:
                    for w in direct_candidates:
                        res = try_place_word_progressive(
                            grid,
                            slot,
                            w,
                            dict_index,
                            predefs,
                            check_perpendicular=True,
                            relax_end_def_checks=True,
                        )
                        if not res:
                            continue
                        _placement, added_def = res
                        used_words.add(w)
                        used_slots.add((slot.start_x, slot.start_y, slot.direction))
                        _track_place_on(grid, cell_use_count, placements, w, slot, mandatory=False)
                        steps += 1
                        log(f"REPAIR OK direct word={w}")
                        _print_place(steps, w, False, slot, added_def, log)
                        emit_grid()
                        placed_total += 1
                        progress += 1
                        placed = True
                        break
                if placed:
                    continue

                # B) depth-1 replacement of a crossing word, only if it enables a new word from this DEF slot
                crossing_idxs = _crossing_placement_indices_for_slot(slot)
                if not crossing_idxs:
                    continue
                crossing_idxs = crossing_idxs[:max_cross_words]

                for old_idx in crossing_idxs:
                    old_p = placements[old_idx]
                    lock_pattern = _locked_pattern_for_placement(old_p)
                    log(
                        f"REPAIR TRY replace old={old_p['word']} @ ({old_p['start_x']},{old_p['start_y']}){old_p['direction']} "
                        f"len={old_p['length']} locked={lock_pattern} because DEF ({dx},{dy}) dir={def_dir}"
                    )

                    # Build replacement slot pattern from the *locked* letters
                    rep_slot = Slot(
                        start_x=old_p["start_x"],
                        start_y=old_p["start_y"],
                        direction=old_p["direction"],
                        length=old_p["length"],
                        pattern=lock_pattern,
                    )

                    exclude = set(used_words)
                    exclude.discard(old_p["word"])
                    rep_candidates = _sample_words_for_exact_pattern(
                        old_p["length"],
                        lock_pattern,
                        exclude,
                        rng,
                        max_candidates=max_replace_candidates,
                    )
                    if not rep_candidates:
                        continue

                    replaced = False
                    for new_w in rep_candidates:
                        # Simulate on copies
                        trial_grid = copy.deepcopy(grid)
                        t_predefs = set(predefs)
                        t_used_slots = set(used_slots)
                        t_placements = [p.copy() for p in placements]
                        t_use = dict(cell_use_count)
                        # remove old
                        removed = _remove_placement_at_index(trial_grid, t_use, t_placements, old_idx)
                        t_used_words = {p["word"] for p in t_placements}

                        # place replacement word
                        res_rep = try_place_word_progressive(
                            trial_grid,
                            rep_slot,
                            new_w,
                            dict_index,
                            t_predefs,
                            check_perpendicular=True,
                            relax_end_def_checks=True,
                        )
                        if not res_rep:
                            continue
                        _placement_rep, _added_def_rep = res_rep
                        _track_place_on(
                            trial_grid,
                            t_use,
                            t_placements,
                            new_w,
                            rep_slot,
                            mandatory=bool(old_p.get("mandatory")),
                        )
                        t_used_words = {pp["word"] for pp in t_placements}

                        # retry target slot after replacement (pattern may have changed)
                        trial_slot = _slot_from_def_on(trial_grid, dx, dy, def_dir)
                        if not trial_slot or trial_slot.pattern.count(".") <= 0:
                            continue
                        target_candidates = _pick_word_candidates_for_slot(
                            trial_slot, dict_index, t_used_words, rng, max_candidates=60
                        )
                        if not target_candidates:
                            continue
                        ok_target = None
                        for tw in target_candidates:
                            res_t = try_place_word_progressive(
                                trial_grid,
                                trial_slot,
                                tw,
                                dict_index,
                                t_predefs,
                                check_perpendicular=True,
                                relax_end_def_checks=True,
                            )
                            if not res_t:
                                continue
                            ok_target = (tw, trial_slot, res_t[1])
                            _track_place_on(trial_grid, t_use, t_placements, tw, trial_slot, mandatory=False)
                            t_used_slots.add((trial_slot.start_x, trial_slot.start_y, trial_slot.direction))
                            break
                        if not ok_target:
                            continue

                        if steps + 2 > max_steps:
                            continue

                        # Commit simulated state
                        grid = trial_grid
                        predefs = t_predefs
                        used_slots = t_used_slots
                        placements = t_placements
                        cell_use_count = t_use
                        used_words = {pp["word"] for pp in placements}

                        steps += 1
                        log(
                            f"REPAIR OK replaced {removed['word']}-> {new_w} @ ({rep_slot.start_x},{rep_slot.start_y}){rep_slot.direction} "
                            f"then placed target={ok_target[0]} from DEF ({dx},{dy}) dir={def_dir}"
                        )
                        # Log the replacement as a placement event, then the target placement event
                        _print_place(steps, new_w, bool(old_p.get("mandatory")), rep_slot, None, log)
                        steps += 1
                        _print_place(steps, ok_target[0], False, ok_target[1], ok_target[2], log)
                        emit_grid()

                        placed_total += 1
                        replaced_total += 1
                        progress += 1
                        replaced = True
                        break

                    if replaced:
                        break

            log(f"REPAIR done pass={p + 1} placed={progress} replaced={replaced_total}")
            if progress == 0:
                break

        return placed_total, replaced_total

    emit_grid()
    log(f"RUN start seed={seed} max_mandatory={max_mandatory} max_steps={max_steps}")
    last_hb = time.time()

    while steps < max_steps:
        candidate = None

        # Try mandatory words first
        i = queue_index
        while i < len(queue):
            word = queue[i]
            grid_copy = copy.deepcopy(grid)
            predefs_copy = set(predefs)
            used_slots_copy = set(used_slots)
            res = progressive_place_next(
                grid_copy,
                word,
                dict_index,
                predefs_copy,
                used_slots_copy,
                seed=seed + i + word_attempt,
            )
            if res:
                _placement, added_def, slot = res
                candidate = {
                    "word": word,
                    "slot": slot,
                    "added_def": added_def,
                    "mandatory": True,
                    "queue_index": i,
                    "check_perp": True,
                }
                break
            unplaced.append(word)
            i += 1
        if candidate:
            queue_index = candidate["queue_index"]
        else:
            queue_index = i

        if candidate is None:
            def _try_fill(check_perp: bool):
                slots = _slot_priority(list_slots(grid), rng)
                found: list[dict] = []
                for slot in slots:
                    key = (slot.start_x, slot.start_y, slot.direction)
                    if key in used_slots:
                        continue
                    end_on_border = _slot_end_on_border(slot, grid)
                    candidates = _pick_word_candidates_for_slot(slot, dict_index, used_words, rng, max_candidates=6)
                    if not candidates:
                        continue
                    for w in candidates:
                        grid_copy = copy.deepcopy(grid)
                        predefs_copy = set(predefs)
                        res = try_place_word_progressive(
                            grid_copy,
                            slot,
                            w,
                            dict_index,
                            predefs_copy,
                            check_perpendicular=check_perp,
                            relax_end_def_checks=True,
                        )
                        if not res:
                            continue
                        _placement, added_def = res
                        weight = 1.0
                        if end_on_border and len(w) == slot.length:
                            weight *= _end_letter_weight(w)
                        found.append(
                            {
                                "word": w,
                                "slot": slot,
                                "added_def": added_def,
                                "mandatory": False,
                                "queue_index": None,
                                "check_perp": check_perp,
                                "_weight": weight,
                            }
                        )
                        if len(found) >= 12:
                            break
                    if len(found) >= 12:
                        break
                if found:
                    return _weighted_choice(found, rng)
                return None

            candidate = _try_fill(check_perp=True)
            if candidate is None:
                candidate = _try_fill(check_perp=False)

        if candidate is None:
            # Expand DEF directions then retry
            added_dirs = expand_def_directions(grid)
            if added_dirs > 0:
                candidate = _try_fill(check_perp=True) or _try_fill(check_perp=False)

            if candidate is None:
                added = _inject_defs_with_words(check_perp=False)
                if added > 0:
                    def_retry = 0
                    continue
                # Fallback: add more DEF cells once
                if def_retry >= 1:
                    added = _inject_defs_with_words(check_perp=True)
                    if added > 0:
                        def_retry = 0
                        continue
                    log("STOP: aucun mot placable.")
                    break

                before = sum(
                    1 for y in range(HEIGHT) for x in range(WIDTH) if grid.get(x, y).type == "DEF"
                )
                try:
                    auto_place_defs(grid, seed=77 + word_attempt)
                except Exception:
                    log("STOP: aucun mot placable.")
                    break
                after = sum(
                    1 for y in range(HEIGHT) for x in range(WIDTH) if grid.get(x, y).type == "DEF"
                )
                if after > before:
                    def_retry += 1
                    continue

                added = _inject_defs_with_words(check_perp=True)
                if added > 0:
                    def_retry = 0
                    continue
                log("STOP: aucun mot placable.")
                break

        # Apply candidate
        slot = candidate["slot"]
        res = try_place_word_progressive(
            grid,
            slot,
            candidate["word"],
            dict_index,
            predefs,
            check_perpendicular=candidate.get("check_perp", True),
            relax_end_def_checks=True,
        )
        if not res:
            word_attempt += 1
            stagnation += 1
            if stagnation >= max_stagnation:
                log("STOP: stagnation.")
                break
            continue
        _placement, added_def = res
        used_slots.add((slot.start_x, slot.start_y, slot.direction))
        used_words.add(candidate["word"])
        _track_place_on(grid, cell_use_count, placements, candidate["word"], slot, candidate["mandatory"])
        if candidate.get("mandatory") and candidate.get("queue_index") is not None:
            queue_index = candidate["queue_index"] + 1
        _print_place(steps + 1, candidate["word"], candidate["mandatory"], slot, added_def, log)
        emit_grid()
        steps += 1
        def_retry = 0
        stagnation = 0

    added = _fill_unfilled_defs()
    if added > 0:
        log(f"DEF-FILL placed={added}")

    rep_placed, rep_replaced = _repair_unfilled_def_dirs()
    if rep_placed > 0 or rep_replaced > 0:
        log(f"REPAIR summary placed={rep_placed} replaced={rep_replaced}")

    added2 = _fill_unfilled_defs()
    if added2 > 0:
        log(f"DEF-FILL placed={added2}")

    log(f"DONE: placed={steps} unplaced={len(unplaced)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mots fleches - console GO")
    parser.add_argument("--input", default="mots_oblig.txt", help="Mandatory words file")
    parser.add_argument("--dict", default="francais.txt", help="Dictionary file")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--max-mandatory", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    run(args.input, args.dict, args.seed, args.max_mandatory, args.max_steps)


if __name__ == "__main__":
    main()
