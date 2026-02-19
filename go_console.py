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

    def write_state() -> None:
        try:
            payload = {"grid": grid.to_json(), "logs": log_buffer}
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

    def _try_add_def_at(x: int, y: int, check_perp: bool) -> tuple[Grid, set, set, set, list] | None:
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
            cell = trial.get(x, y)
            cell.type = "DEF"
            cell.letter = None
            cell.defs = [DefEntry(direction=d) for d in dirs]
            if def_run_too_long(trial, set()) or has_singleton_segments(trial):
                continue
            t_predefs.add((x, y))
            placements = []
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
                        placements.append((w, slot, added_def))
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
            if not ok:
                continue
            if def_run_too_long(trial, set()) or has_singleton_segments(trial):
                continue
            return trial, t_predefs, t_used_slots, t_used_words, placements
        return None

    def _inject_defs_with_words(check_perp: bool) -> int:
        nonlocal grid, predefs, used_slots, used_words, steps
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
            grid, predefs, used_slots, used_words, placements = res
            dirs = ",".join([d.direction for d in grid.get(xx, yy).defs])
            detail = []
            for w, slot, added_def in placements:
                detail.append(f"{w}@({slot.start_x},{slot.start_y}){slot.direction}")
            log(f"INJECT OK ({xx},{yy}) dirs={dirs} words={'; '.join(detail)}")
            for w, slot, added_def in placements:
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
    log(f"DONE: placed={steps} unplaced={len(unplaced)}")



    queue_index = 0
    used_slots: set[tuple[int, int, str]] = set()
    used_words: set[str] = set()
    unplaced: list[str] = []
    word_attempt = 0
    def_retry = 0
    steps = 0

    rng = random.Random(1000 + seed)

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
        if candidate.get("mandatory") and candidate.get("queue_index") is not None:
            queue_index = candidate["queue_index"] + 1
        _print_place(steps + 1, candidate["word"], candidate["mandatory"], slot, added_def, log)
        emit_grid()
        steps += 1
        def_retry = 0
        stagnation = 0
        if time.time() - last_hb > 5:
            log(f"HB steps={steps} queue_idx={queue_index} used_slots={len(used_slots)}")
            last_hb = time.time()

    added = _fill_unfilled_defs()
    if added > 0:
        log(f"DEF-FILL placed={added}")
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
