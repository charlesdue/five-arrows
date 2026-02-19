import json
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from mots_fleches import (
    Grid,
    WIDTH,
    HEIGHT,
    load_mandatory_words,
    mandatory_placement_generator,
    fill_with_llm,
    words_list,
    draw_pdf,
)


CELL_SIZE = 24
PADDING = 10


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mots fleches - Interface pas a pas")
        self.resizable(False, False)

        self.grid_obj = Grid(WIDTH, HEIGHT)
        self.mandatory_words = []
        self.placements = []
        self.reserved_defs = set()
        self.unplaced = []
        self.stop_flag = False
        self.step3_gen = None

        self._build_ui()
        self.bind("<Escape>", self._on_escape)

    def _build_ui(self) -> None:
        main = ttk.Frame(self)
        main.grid(row=0, column=0, padx=10, pady=10)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="n")

        right = ttk.Frame(main)
        right.grid(row=0, column=1, padx=10, sticky="n")

        self.canvas = tk.Canvas(
            right,
            width=WIDTH * CELL_SIZE + 2,
            height=HEIGHT * CELL_SIZE + 2,
            bg="white",
            highlightthickness=1,
        )
        self.canvas.grid(row=0, column=0)

        ttk.Button(left, text="Etape 1: Charger code", command=self.step1).grid(
            row=0, column=0, sticky="ew", pady=2
        )
        ttk.Button(left, text="Etape 2: Charger dico", command=self.step2).grid(
            row=1, column=0, sticky="ew", pady=2
        )
        ttk.Button(left, text="Etape 3: Placer obligatoires", command=self.step3).grid(
            row=2, column=0, sticky="ew", pady=2
        )
        ttk.Button(left, text="Etape 4: Remplissage", command=self.step4).grid(
            row=3, column=0, sticky="ew", pady=2
        )
        ttk.Button(left, text="Exporter PDFs/JSON", command=self.export_outputs).grid(
            row=4, column=0, sticky="ew", pady=8
        )

        self.status = ttk.Label(left, text="Pret.")
        self.status.grid(row=5, column=0, sticky="ew", pady=4)

        self.log = tk.Text(left, width=38, height=18)
        self.log.grid(row=6, column=0, pady=4)

        self._draw_grid()

    def _log(self, msg: str) -> None:
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def _on_escape(self, _evt=None) -> None:
        self.stop_flag = True
        self._log("Stop demande (ESC).")

    def _draw_grid(self) -> None:
        self.canvas.delete("all")
        for y in range(HEIGHT):
            for x in range(WIDTH):
                cell = self.grid_obj.get(x, y)
                x0 = x * CELL_SIZE + 1
                y0 = y * CELL_SIZE + 1
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                fill = "white"
                if cell.type == "DEF":
                    fill = "#e6e6e6"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#999")
                if cell.type == "LETTER" and cell.letter:
                    self.canvas.create_text(
                        x0 + CELL_SIZE / 2,
                        y0 + CELL_SIZE / 2,
                        text=cell.letter,
                        font=("Arial", 10, "bold"),
                    )
                elif cell.type == "DEF" and cell.defs:
                    # ASCII arrows: R, D, RD
                    txt = ",".join(
                        ["R" if d.direction == "RIGHT" else "D" if d.direction == "DOWN" else "RD" for d in cell.defs]
                    )
                    self.canvas.create_text(
                        x0 + 3,
                        y0 + 3,
                        text=txt,
                        anchor="nw",
                        font=("Arial", 6),
                    )

    def step1(self) -> None:
        self.status.config(text="Etape 1: code charge.")
        self._log("Code charge (module importe).")

    def step2(self) -> None:
        path = filedialog.askopenfilename(
            title="Choisir mots_oblig.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        self.mandatory_words = load_mandatory_words(path)
        self._log(f"Dico charge: {len(self.mandatory_words)} mots")
        self._log(" ".join(self.mandatory_words[:40]) + (" ..." if len(self.mandatory_words) > 40 else ""))
        self.status.config(text="Etape 2: dico charge.")

    def step3(self) -> None:
        if not self.mandatory_words:
            messagebox.showerror("Erreur", "Charge le dico d'abord (Etape 2).")
            return
        self.stop_flag = False
        self.grid_obj = Grid(WIDTH, HEIGHT)
        self.step3_gen = mandatory_placement_generator(self.grid_obj, self.mandatory_words, seed=42)
        self.status.config(text="Etape 3: placement en cours...")
        self._log("Placement des mots obligatoires...")
        self._step3_tick()

    def _step3_tick(self) -> None:
        if self.stop_flag:
            self.status.config(text="Etape 3: stoppe.")
            return
        try:
            info = next(self.step3_gen)
        except StopIteration:
            self.status.config(text="Etape 3: termine.")
            return
        self.placements = info["placements"]
        self.reserved_defs = info["reserved_defs"]
        self.unplaced = info["unplaced"]
        if info["placed_word"]:
            self._log(f"Place: {info['placed_word']}")
        if info["done"]:
            self.status.config(text=f"Etape 3: termine. Non places: {len(self.unplaced)}")
        self._draw_grid()
        if not info["done"]:
            self.after(10, self._step3_tick)

    def step4(self) -> None:
        if not self.mandatory_words:
            messagebox.showerror("Erreur", "Charge le dico d'abord (Etape 2).")
            return
        if not self.placements:
            messagebox.showerror("Erreur", "Place les mots obligatoires d'abord (Etape 3).")
            return
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            messagebox.showerror("Erreur", "OPENAI_API_KEY manquante (.env).")
            return
        self.stop_flag = False
        self.status.config(text="Etape 4: remplissage en cours...")
        self._log("Remplissage via LLM (cela peut prendre du temps)...")
        t = threading.Thread(target=self._fill_thread, daemon=True)
        t.start()

    def _fill_thread(self) -> None:
        try:
            filled = fill_with_llm(
                self.grid_obj,
                self.placements,
                self.reserved_defs,
                self.mandatory_words,
                self.unplaced,
                os.environ.get("OPENAI_API_KEY", ""),
                "gpt-5",
                max_attempts=10,
                seed=42,
            )
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erreur LLM", str(e)))
            self.after(0, lambda: self.status.config(text="Etape 4: erreur."))
            return
        if self.stop_flag:
            self.after(0, lambda: self._log("Remplissage annule (ESC)."))
            self.after(0, lambda: self.status.config(text="Etape 4: stoppe."))
            return
        self.grid_obj = filled
        self.after(0, self._draw_grid)
        self.after(0, lambda: self.status.config(text="Etape 4: termine."))
        self.after(0, lambda: self._log("Remplissage termine."))

    def export_outputs(self) -> None:
        outdir = filedialog.askdirectory(title="Choisir dossier de sortie")
        if not outdir:
            return
        os.makedirs(outdir, exist_ok=True)
        # JSON
        json_path = os.path.join(outdir, "grille.json")
        data = self.grid_obj.to_json()
        data["words"] = words_list(self.grid_obj)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
        # PDFs (if reportlab installed)
        try:
            draw_pdf(self.grid_obj, os.path.join(outdir, "grille_vide.pdf"), show_letters=False)
            draw_pdf(self.grid_obj, os.path.join(outdir, "grille_solution.pdf"), show_letters=True)
        except Exception as e:
            self._log(f"PDF non genere: {e}")
        self._log(f"Export OK: {outdir}")
        self.status.config(text="Export termine.")


if __name__ == "__main__":
    app = App()
    app.mainloop()
