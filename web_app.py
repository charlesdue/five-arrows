import http.server
import socketserver
import threading
import queue
import json

from go_console import run
from mots_fleches import Grid


PORT = 8000
LOG_QUEUE: "queue.Queue[str]" = queue.Queue()
LOG_BUFFER: list[str] = []
RUN_LOCK = threading.Lock()
RUNNING = False
GRID_STATE: Grid | None = None


def _enqueue(msg) -> None:
    if isinstance(msg, (tuple, list)):
        msg = " ".join(str(x) for x in msg)
    else:
        msg = str(msg)
    print(msg, flush=True)
    LOG_BUFFER.append(msg)
    if len(LOG_BUFFER) > 5000:
        del LOG_BUFFER[:-4000]
    LOG_QUEUE.put(msg)


def _grid_cb(grid: Grid) -> None:
    global GRID_STATE
    GRID_STATE = grid


def _run_job() -> None:
    global RUNNING
    try:
        run(
            "mots_oblig.txt",
            "francais.txt",
            seed=42,
            max_mandatory=5,
            max_steps=500,
            log_fn=_enqueue,
            grid_cb=_grid_cb,
        )
    finally:
        _enqueue("DONE")
        with RUN_LOCK:
            RUNNING = False


HTML_PAGE = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Mots fleches - GO</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      button { padding: 8px 14px; font-size: 14px; }
      .row { display: flex; gap: 12px; align-items: center; }
      .wrap { display: grid; grid-template-columns: 1fr 420px; gap: 16px; }
      #grid { border: 1px solid #ddd; padding: 8px; overflow: auto; }
      #log { white-space: pre-wrap; border: 1px solid #ddd; padding: 12px; height: 520px; overflow: auto; }
      #compact { margin-top: 16px; white-space: pre-wrap; border: 1px solid #ddd; padding: 12px; font-family: Consolas, monospace; }
      table.grid { border-collapse: collapse; font-size: 12px; }
      table.grid td { border: 1px solid #999; width: 26px; height: 26px; text-align: center; vertical-align: middle; }
      td.def { background: #e6e6e6; color: #333; font-size: 10px; }
      td.letter { background: #fff; font-weight: bold; }
      .status { color: #555; font-size: 13px; }
    </style>
  </head>
  <body>
    <h2>Mots fleches - GO</h2>
    <div class="row">
      <button id="go">GO</button>
      <div class="status" id="status">idle</div>
    </div>
    <div class="wrap">
      <div id="grid"></div>
      <div id="log"></div>
    </div>
    <div id="compact"></div>
    <script>
      const logEl = document.getElementById('log');
      const gridEl = document.getElementById('grid');
      const compactEl = document.getElementById('compact');
      const statusEl = document.getElementById('status');
      function renderGrid(data) {
        if (!data || !data.cells) return;
        const { width, height, cells } = data;
        let html = '<table class="grid">';
        for (let y = 0; y < height; y++) {
          html += '<tr>';
          for (let x = 0; x < width; x++) {
            const c = cells[y * width + x];
            if (c.type === 'DEF') {
              const dirs = (c.defs || []).map(d => d.direction === 'RIGHT' ? 'R' : d.direction === 'DOWN' ? 'D' : 'RD').join(',');
              html += `<td class="def">${dirs}</td>`;
            } else if (c.type === 'LETTER' && c.letter) {
              html += `<td class="letter">${c.letter}</td>`;
            } else {
              html += '<td></td>';
            }
          }
          html += '</tr>';
        }
        html += '</table>';
        gridEl.innerHTML = html;
      }
      function poll() {
        fetch('/grid').then(r => r.json()).then(renderGrid);
        fetch('/logs').then(r => r.json()).then(data => {
          if (!data || !data.lines) return;
          logEl.textContent = data.lines.join('\\n');
          logEl.scrollTop = logEl.scrollHeight;
          if (data.done) statusEl.textContent = 'done';
        });
        fetch('/compact').then(r => r.text()).then(txt => {
          if (txt) compactEl.textContent = txt;
        });
      }
      setInterval(poll, 1000);
      poll();
      document.getElementById('go').onclick = () => {
        logEl.textContent = '';
        statusEl.textContent = 'running';
        fetch('/start', { method: 'POST' }).then(r => {
          if (!r.ok) {
            statusEl.textContent = 'already running';
          }
        });
      };
    </script>
  </body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, content_type: str = "text/plain") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/grid":
            if GRID_STATE is None:
                self._send(200, b"{}", "application/json")
                return
            data = GRID_STATE.to_json()
            body = json.dumps(data).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if self.path == "/compact":
            if GRID_STATE is None:
                self._send(200, b"", "text/plain; charset=utf-8")
                return
            lines = []
            for y in range(GRID_STATE.height):
                row = []
                for x in range(GRID_STATE.width):
                    c = GRID_STATE.get(x, y)
                    if c.type == "DEF":
                        row.append("DEF")
                    elif c.type == "LETTER" and c.letter:
                        row.append(c.letter)
                    else:
                        row.append(".")
                lines.append(" | ".join(row))
            body = ("\n".join(lines)).encode("utf-8")
            self._send(200, body, "text/plain; charset=utf-8")
            return
        if self.path == "/logs":
            done = not RUNNING
            body = json.dumps({"lines": LOG_BUFFER, "done": done}).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if self.path.startswith("/events"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                self.wfile.write(b"retry: 1000\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            while True:
                try:
                    msg = LOG_QUEUE.get(timeout=1.0)
                    data = msg.replace("\n", " ")
                    self.wfile.write(f"data: {data}\\n\\n".encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    try:
                        self.wfile.write(b": keep-alive\\n\\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                except (BrokenPipeError, ConnectionResetError):
                    break
            return
        self._send(404, b"Not found")

    def do_POST(self) -> None:
        if self.path == "/start":
            global RUNNING
            with RUN_LOCK:
                if RUNNING:
                    self._send(409, b"Already running")
                    return
                # Clear old logs
                while not LOG_QUEUE.empty():
                    try:
                        LOG_QUEUE.get_nowait()
                    except queue.Empty:
                        break
                RUNNING = True
                t = threading.Thread(target=_run_job, daemon=True)
                t.start()
            self._send(200, b"OK")
            return
        self._send(404, b"Not found")


def main() -> None:
    with socketserver.ThreadingTCPServer(("", PORT), Handler) as httpd:
        print(f"Serving on http://localhost:{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
