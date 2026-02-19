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


def _slot_priority(slots, rng):
    rng.shuffle(slots)
    slots.sort(key=lambda s: (s.length, 0 if s.direction == "DOWN" else 1, rng.random()))
    return slots


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
                for slot in slots:
                    key = (slot.start_x, slot.start_y, slot.direction)
                    if key in used_slots:
                        continue
                    w = _pick_word_for_slot(slot, dict_index, used_words, rng)
                    if not w:
                        continue
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
                    return {
                        "word": w,
                        "slot": slot,
                        "added_def": added_def,
                        "mandatory": False,
                        "queue_index": None,
                        "check_perp": check_perp,
                    }
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
                # Fallback: add more DEF cells once
                if def_retry >= 1:
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
                # Fallback: add more DEF cells once
                if def_retry >= 1:
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
