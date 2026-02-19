import copy
import os
import random
import time
from typing import List

import streamlit as st

from mots_fleches import (
    Grid,
    WIDTH,
    HEIGHT,
    Slot,
    load_mandatory_words,
    select_mandatory_subset,
    load_env_fallback,
    load_dictionary,
    normalize_word,
    mandatory_placement_generator,
    predef_slot_lengths,
    select_words_for_slots,
    build_dictionary_index,
    place_mandatory_words_backtracking,
    place_mandatory_with_skeleton,
    progressive_place_next,
    build_llm_prompt,
    llm_attempt,
    auto_place_defs,
    apply_border_def_pattern,
    list_slots,
    apply_word_to_grid,
    validate_partial_structure,
    propose_word_llm,
    try_place_word_progressive,
)


def init_state() -> None:
    if "grid" not in st.session_state:
        st.session_state.grid = Grid(WIDTH, HEIGHT)
    if "mandatory_words" not in st.session_state:
        st.session_state.mandatory_words = []
    if "placements" not in st.session_state:
        st.session_state.placements = []
    if "reserved_defs" not in st.session_state:
        st.session_state.reserved_defs = set()
    if "unplaced" not in st.session_state:
        st.session_state.unplaced = []
    if "gen" not in st.session_state:
        st.session_state.gen = None
    if "log" not in st.session_state:
        st.session_state.log = []
    if "mandatory_subset" not in st.session_state:
        st.session_state.mandatory_subset = []
    if "dictionary" not in st.session_state:
        st.session_state.dictionary = set()
    if "slots" not in st.session_state:
        st.session_state.slots = []
    if "slot_index" not in st.session_state:
        st.session_state.slot_index = 0
    if "proposed_word" not in st.session_state:
        st.session_state.proposed_word = ""
    if "last_word_cells" not in st.session_state:
        st.session_state.last_word_cells = []
    if "llm_prompt_system" not in st.session_state:
        st.session_state.llm_prompt_system = ""
    if "llm_prompt_user" not in st.session_state:
        st.session_state.llm_prompt_user = ""
    if "llm_last_response" not in st.session_state:
        st.session_state.llm_last_response = None
    if "llm_last_error" not in st.session_state:
        st.session_state.llm_last_error = None
    if "llm_valid_grid" not in st.session_state:
        st.session_state.llm_valid_grid = None
    if "llm_attempts" not in st.session_state:
        st.session_state.llm_attempts = 0
    if "llm_history" not in st.session_state:
        st.session_state.llm_history = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "mandatory_text" not in st.session_state:
        st.session_state.mandatory_text = ""
    if "queue_words" not in st.session_state:
        st.session_state.queue_words = []
    if "queue_index" not in st.session_state:
        st.session_state.queue_index = 0
    if "predefs" not in st.session_state:
        st.session_state.predefs = set()
    if "used_slots" not in st.session_state:
        st.session_state.used_slots = set()
    if "used_words" not in st.session_state:
        st.session_state.used_words = set()
    if "dict_index" not in st.session_state:
        st.session_state.dict_index = None
    if "pending_word" not in st.session_state:
        st.session_state.pending_word = None
    if "word_attempt" not in st.session_state:
        st.session_state.word_attempt = 0
    if "def_retry" not in st.session_state:
        st.session_state.def_retry = 0


def log(msg: str) -> None:
    st.session_state.log.append(msg)


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
                return w
    return None


def _slot_priority(slots, rng):
    rng.shuffle(slots)
    slots.sort(key=lambda s: (s.length, 0 if s.direction == "DOWN" else 1, rng.random()))
    return slots


def _prepare_candidate() -> bool:
    if st.session_state.pending_word is not None:
        log("Un mot est deja prepare.")
        return True

    rng = random.Random(1000 + st.session_state.word_attempt)
    # Try mandatory words first
    i = st.session_state.queue_index
    while i < len(st.session_state.queue_words):
        word = st.session_state.queue_words[i]
        grid_copy = copy.deepcopy(st.session_state.grid)
        predefs_copy = set(st.session_state.predefs)
        used_slots_copy = set(st.session_state.used_slots)
        res = progressive_place_next(
            grid_copy,
            word,
            st.session_state.dict_index,
            predefs_copy,
            used_slots_copy,
            seed=42 + i + st.session_state.word_attempt,
        )
        if res:
            _placement, added_def, slot = res
            st.session_state.pending_word = {
                "word": word,
                "slot": {
                    "start_x": slot.start_x,
                    "start_y": slot.start_y,
                    "direction": slot.direction,
                    "length": slot.length,
                    "pattern": slot.pattern,
                },
                "added_def": added_def,
                "mandatory": True,
                "queue_index": i,
                "check_perp": True,
            }
            st.session_state.def_retry = 0
            log(
                f"Prepare: {word} {slot.direction} @ ({slot.start_x},{slot.start_y}) len={slot.length} "
                f"pattern={slot.pattern} (obligatoire)"
                + (f" +DEF ({added_def[0]},{added_def[1]})" if added_def else "")
            )
            return True
        st.session_state.unplaced.append(word)
        log(f"Non place (oblig): {word}")
        i += 1
    st.session_state.queue_index = i

    def _try_fill(check_perp: bool) -> bool:
        slots = _slot_priority(list_slots(st.session_state.grid), rng)
        for slot in slots:
            key = (slot.start_x, slot.start_y, slot.direction)
            if key in st.session_state.used_slots:
                continue
            w = _pick_word_for_slot(slot, st.session_state.dict_index, st.session_state.used_words, rng)
            if not w:
                continue
            # Preview placement to capture possible end DEF
            grid_copy = copy.deepcopy(st.session_state.grid)
            predefs_copy = set(st.session_state.predefs)
            res = try_place_word_progressive(
                grid_copy,
                slot,
                w,
                st.session_state.dict_index,
                predefs_copy,
                check_perpendicular=check_perp,
                relax_end_def_checks=True,
            )
            if not res:
                continue
            _placement, added_def = res
            st.session_state.pending_word = {
                "word": w,
                "slot": {
                    "start_x": slot.start_x,
                    "start_y": slot.start_y,
                    "direction": slot.direction,
                    "length": slot.length,
                    "pattern": slot.pattern,
                },
                "added_def": added_def,
                "mandatory": False,
                "queue_index": None,
                "check_perp": check_perp,
            }
            st.session_state.def_retry = 0
            log(
                f"Prepare: {w} {slot.direction} @ ({slot.start_x},{slot.start_y}) len={slot.length} "
                f"pattern={slot.pattern} (remplissage)"
                + (f" +DEF ({added_def[0]},{added_def[1]})" if added_def else "")
            )
            return True
        return False

    # Try non-mandatory fill words (strict, then relaxed)
    if _try_fill(check_perp=True):
        return True
    if _try_fill(check_perp=False):
        log("Remplissage relaxe (sans contrainte perpendiculaire).")
        return True

    # As fallback, add a few DEF cells to open new slots, then retry once
    if st.session_state.def_retry >= 1:
        log("Aucun mot placable.")
        return False
    before = sum(
        1 for y in range(HEIGHT) for x in range(WIDTH) if st.session_state.grid.get(x, y).type == "DEF"
    )
    try:
        auto_place_defs(st.session_state.grid, seed=77 + st.session_state.word_attempt)
    except Exception:
        log("Aucun mot placable.")
        return False
    after = sum(
        1 for y in range(HEIGHT) for x in range(WIDTH) if st.session_state.grid.get(x, y).type == "DEF"
    )
    if after > before:
        log(f"Ajout DEF: +{after - before}")
        st.session_state.def_retry += 1
        return _prepare_candidate()
    log("Aucun mot placable.")
    return False


def _apply_pending_and_prepare_next() -> bool:
    if st.session_state.pending_word is None:
        return _prepare_candidate()
    pf = st.session_state.pending_word
    slot = pf["slot"]
    slot_obj = Slot(
        start_x=slot["start_x"],
        start_y=slot["start_y"],
        direction=slot["direction"],
        length=slot["length"],
        pattern=slot["pattern"],
    )
    res = try_place_word_progressive(
        st.session_state.grid,
        slot_obj,
        pf["word"],
        st.session_state.dict_index,
        st.session_state.predefs,
        check_perpendicular=pf.get("check_perp", True),
        relax_end_def_checks=True,
    )
    if res:
        placement, _added_def = res
        st.session_state.placements.append(placement)
        st.session_state.used_slots.add((slot_obj.start_x, slot_obj.start_y, slot_obj.direction))
        st.session_state.used_words.add(pf["word"])
        if pf.get("mandatory") and pf.get("queue_index") is not None:
            st.session_state.queue_index = pf["queue_index"] + 1
        log(f"Applique: {pf['word']}")
    else:
        st.session_state.word_attempt += 1
        log("Application echouee, nouvelle tentative.")
    st.session_state.pending_word = None
    return _prepare_candidate()


def _run_go(max_steps: int = 500) -> None:
    steps = 0
    if not _prepare_candidate():
        log("Stop: aucun mot placable.")
        return
    while steps < max_steps and st.session_state.pending_word is not None:
        before = len(st.session_state.placements)
        ok = _apply_pending_and_prepare_next()
        after = len(st.session_state.placements)
        steps += 1
        if after == before and not ok:
            log("Stop: aucun mot placable.")
            return
    if steps >= max_steps:
        log("Stop: limite de steps atteinte.")
        return
    if st.session_state.pending_word is None:
        log("Stop: aucun mot placable.")


def render_grid_html(grid: Grid) -> str:
    highlight = set(st.session_state.get("last_word_cells", []))
    cell_px = 26
    rows = []
    rows.append(
        "<style>"
        "table.grid{border-collapse:collapse;font-family:Arial;font-size:12px}"
        "table.grid td{border:1px solid #999;width:%dpx;height:%dpx;text-align:center;vertical-align:middle}"
        "td.def{background:#e6e6e6;color:#333;font-size:10px}"
        "td.letter{background:#fff;font-weight:bold}"
        "td.hl{background:#ffeaa7}"
        "</style>" % (cell_px, cell_px)
    )
    rows.append("<table class='grid'>")
    for y in range(grid.height):
        rows.append("<tr>")
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type == "DEF":
                dirs = ""
                if c.defs:
                    dirs = ",".join(
                        ["R" if d.direction == "RIGHT" else "D" if d.direction == "DOWN" else "RD" for d in c.defs]
                    )
                cls = "def"
                if (x, y) in highlight:
                    cls = "def hl"
                rows.append(f"<td class='{cls}'>{dirs}</td>")
            elif c.type == "LETTER" and c.letter:
                cls = "letter"
                if (x, y) in highlight:
                    cls = "letter hl"
                rows.append(f"<td class='{cls}'>{c.letter}</td>")
            else:
                cls = ""
                if (x, y) in highlight:
                    cls = "hl"
                rows.append(f"<td class='{cls}'></td>")
        rows.append("</tr>")
    rows.append("</table>")
    return "\n".join(rows)


def step1() -> None:
    return


def step0_describe() -> None:
    return


def check_impossible_words(words: List[str]) -> List[str]:
    max_len = max(WIDTH, HEIGHT)
    return [w for w in words if len(w) > max_len]


def step2(file_bytes: bytes) -> None:
    path = ".tmp_mots_oblig.txt"
    with open(path, "wb") as f:
        f.write(file_bytes)
    st.session_state.mandatory_words = load_mandatory_words(path)


def step2_load_local() -> None:
    st.session_state.mandatory_words = load_mandatory_words("mots_oblig.txt")
    st.session_state.dictionary = load_dictionary("francais.txt")
    st.session_state.mandatory_text = "\n".join(st.session_state.mandatory_words)


def step3_init(seed: int, max_words: int) -> None:
    too_long = check_impossible_words(st.session_state.mandatory_words)
    if too_long:
        log(
            "Etape 3: impossible de placer certains mots (trop longs pour la grille): "
            + ", ".join(too_long[:10])
            + (" ..." if len(too_long) > 10 else "")
        )
        return
    st.session_state.grid = Grid(WIDTH, HEIGHT)
    subset = select_mandatory_subset(st.session_state.mandatory_words, max_words, seed)
    st.session_state.mandatory_subset = subset
    st.session_state.gen = mandatory_placement_generator(
        st.session_state.grid, subset, seed=seed
    )
    st.session_state.placements = []
    st.session_state.reserved_defs = set()
    st.session_state.unplaced = []
    log(f"Etape 3: placement de {len(subset)} mots obligatoires...")


def step3_tick() -> bool:
    if st.session_state.gen is None:
        return True
    try:
        info = next(st.session_state.gen)
    except StopIteration:
        log("Etape 3: termine.")
        return True
    st.session_state.placements = info["placements"]
    st.session_state.reserved_defs = info["reserved_defs"]
    st.session_state.unplaced = info["unplaced"]
    if info["placed_word"]:
        log(f"Place: {info['placed_word']}")
    if info["done"]:
        log(f"Etape 3: termine. Non places: {len(st.session_state.unplaced)}")
        return True
    return False


def step4_prepare_prompt() -> None:
    system, user = build_llm_prompt(
        st.session_state.grid,
        st.session_state.placements,
        st.session_state.reserved_defs,
        st.session_state.mandatory_subset or st.session_state.mandatory_words,
        st.session_state.unplaced,
    )
    st.session_state.llm_prompt_system = system
    st.session_state.llm_prompt_user = user


def step4_attempt(model: str) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.error("OPENAI_API_KEY manquante. Ajoute-la dans .env ou dans l'environnement.")
        return
    log("Etape 4: tentative LLM...")
    st.session_state.llm_attempts += 1
    grid_out, data, err = llm_attempt(
        st.session_state.grid,
        st.session_state.placements,
        st.session_state.reserved_defs,
        st.session_state.mandatory_subset or st.session_state.mandatory_words,
        st.session_state.unplaced,
        api_key,
        model,
    )
    st.session_state.llm_last_response = data
    st.session_state.llm_last_error = err
    st.session_state.llm_valid_grid = grid_out
    if err:
        log(f"Tentative {st.session_state.llm_attempts}: invalide -> {err}")
    else:
        log(f"Tentative {st.session_state.llm_attempts}: valide (pret a appliquer).")
    attempt_info = {
        "attempt": st.session_state.llm_attempts,
        "model": model,
        "ok": err is None,
        "error": err,
        "timestamp": time.time(),
    }
    st.session_state.llm_history.append(attempt_info)


def grid_stats(grid: Grid) -> str:
    letters = 0
    defs = 0
    empty = 0
    for y in range(grid.height):
        for x in range(grid.width):
            c = grid.get(x, y)
            if c.type == "LETTER":
                letters += 1
            elif c.type == "DEF":
                defs += 1
            else:
                empty += 1
    return f"lettres={letters}, DEF={defs}, vides={empty}"


def main() -> None:
    st.set_page_config(page_title="Mots fleches - Pas a pas", layout="wide")
    # Ensure .env is loaded when running via streamlit
    load_env_fallback()
    init_state()

    st.title("Mots fleches - Interface pas a pas (Streamlit)")

    # Auto-init: load words/dictionary, build grid, place mandatory DEFs
    if not st.session_state.initialized:
        st.session_state.mandatory_words = load_mandatory_words("mots_oblig.txt")
        st.session_state.dictionary = load_dictionary("francais.txt")
        st.session_state.mandatory_text = "\n".join(st.session_state.mandatory_words)
        dict_words = st.session_state.dictionary.union(st.session_state.mandatory_words)
        st.session_state.dict_index = build_dictionary_index(list(dict_words))
        base_grid = Grid(WIDTH, HEIGHT)
        apply_border_def_pattern(base_grid, seed=42)
        grid = base_grid
        for i in range(8):
            trial = copy.deepcopy(base_grid)
            try:
                auto_place_defs(trial, seed=42 + i)
                grid = trial
                break
            except Exception:
                continue
        predefs = {(x, y) for y in range(HEIGHT) for x in range(WIDTH) if grid.get(x, y).type == "DEF"}
        slot_lengths = predef_slot_lengths(grid, predefs)
        queue = select_words_for_slots(st.session_state.mandatory_words, slot_lengths, 5, seed=42)
        st.session_state.mandatory_words = queue
        st.session_state.grid = grid
        st.session_state.predefs = predefs
        st.session_state.mandatory_subset = queue
        st.session_state.queue_words = queue
        st.session_state.queue_index = 0
        st.session_state.used_slots = set()
        st.session_state.used_words = set()
        st.session_state.placements = []
        st.session_state.unplaced = []
        st.session_state.log = []
        st.session_state.pending_word = None
        st.session_state.word_attempt = 0
        st.session_state.initialized = True
    # Refresh dict_index if older session lacks bigram support
    if st.session_state.dict_index is not None and not hasattr(st.session_state.dict_index, "bigrams"):
        dict_words = st.session_state.dictionary.union(st.session_state.mandatory_words)
        st.session_state.dict_index = build_dictionary_index(list(dict_words))

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Controles")
        if st.button("GO"):
            _run_go()
            st.rerun()

        st.markdown("---")
        st.subheader("Log")
        st.text_area(" ", value="\n".join(st.session_state.log), height=300)

    with col2:
        st.subheader("Grille")
        st.markdown(render_grid_html(st.session_state.grid), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
