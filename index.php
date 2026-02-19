<?php
// Simple PHP UI: start/stop python generator and view grid/logs
chdir(__DIR__);

$python = 'python';
$script = 'go_console.py';
$state = 'state.json';
$runlog = 'run.log';

$action = $_POST['action'] ?? null;
$output = '';

if ($action === 'start') {
    $cmd = $python . ' ' . $script;
    if (stripos(PHP_OS, 'WIN') === 0) {
        // Detached start on Windows, redirect output to run.log
        $cmdline = 'start "mots" /b cmd /c "' . $cmd . ' > ' . $runlog . ' 2>&1"';
        pclose(popen($cmdline, 'r'));
    } else {
        exec($cmd . ' > ' . $runlog . ' 2>&1 &');
    }
    $output = 'STARTED';
} elseif ($action === 'stop') {
    if (stripos(PHP_OS, 'WIN') === 0) {
        exec('taskkill /F /IM python.exe 2>nul');
    } else {
        exec('pkill -f ' . escapeshellarg($script));
    }
    $output = 'STOP REQUESTED';
}

$state_mtime = file_exists($state) ? date('H:i:s', filemtime($state)) : 'missing';
$runlog_mtime = file_exists($runlog) ? date('H:i:s', filemtime($runlog)) : 'missing';
?>
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Mots fleches - Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    button { padding: 8px 14px; font-size: 14px; }
    .wrap { display: grid; grid-template-columns: 1fr 360px; gap: 16px; }
    #grid { border: 1px solid #ddd; padding: 8px; overflow: auto; }
    #log { white-space: pre-wrap; border: 1px solid #ddd; padding: 12px; height: 520px; overflow: auto; }
    table.grid { border-collapse: collapse; font-size: 12px; }
    table.grid td { border: 1px solid #999; width: 26px; height: 26px; text-align: center; vertical-align: middle; }
    td.def { background: #e6e6e6; color: #333; font-size: 10px; }
    td.letter { background: #fff; font-weight: bold; }
    .meta { color: #666; font-size: 12px; margin-left: 8px; }
  </style>
</head>
<body>
  <h2>Mots fleches - Viewer</h2>
  <form method="post" style="margin-bottom:12px;">
    <button type="submit" name="action" value="start">Start</button>
    <button type="submit" name="action" value="stop">Stop</button>
    <span class="meta">status: <?= htmlspecialchars($output, ENT_QUOTES) ?></span>
    <span class="meta">state.json: <?= htmlspecialchars($state_mtime, ENT_QUOTES) ?></span>
    <span class="meta">run.log: <?= htmlspecialchars($runlog_mtime, ENT_QUOTES) ?></span>
  </form>
  <div class="wrap">
    <div id="grid"></div>
    <div id="log"></div>
  </div>
  <script>
    const gridEl = document.getElementById('grid');
    const logEl = document.getElementById('log');
    function render(data) {
      if (!data || !data.grid) return;
      const { width, height, cells } = data.grid;
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
      if (data.logs) {
        logEl.textContent = data.logs.join("\n");
        logEl.scrollTop = logEl.scrollHeight;
      }
    }
    function poll() {
      fetch('state.json?_=' + Date.now())
        .then(r => r.ok ? r.json() : null)
        .then(render)
        .catch(() => {});
    }
    setInterval(poll, 1000);
    poll();
  </script>
</body>
</html>
