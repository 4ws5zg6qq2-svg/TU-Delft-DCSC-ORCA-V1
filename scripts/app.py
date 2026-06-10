# uv run python scripts/server.py orca_core/models/v1/orcahand_right/config.yaml

import sys
import threading
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from common import add_hand_arguments, create_hand, connect_hand, shutdown_hand
from orca_core import OrcaJointPositions

_args = None
_hand = None
_stop_event = threading.Event()
_state = {"task": "idle", "thread": None}
_lock = threading.Lock()

NUM_STEPS = 8
STEP_SIZE = 0.02
HOLD = 0.2
TOUCH_HOLD = 0.2


def _pose(fractions: dict) -> OrcaJointPositions:
    pose = dict(_hand.config.neutral_position)
    for joint, frac in fractions.items():
        if joint not in _hand.config.joint_roms_dict:
            continue
        lo, hi = _hand.config.joint_roms_dict[joint]
        pose[joint] = lo + frac * (hi - lo)
    return OrcaJointPositions.from_dict(pose)


def _run_task(name: str, fn) -> bool:
    with _lock:
        if _state["thread"] and _state["thread"].is_alive():
            return False
        _stop_event.clear()

        def worker():
            _state["task"] = name
            try:
                fn()
            finally:
                _state["task"] = "idle"

        t = threading.Thread(target=worker, daemon=True)
        _state["thread"] = t
        t.start()
    return True


def _spread_fingers():
    wave_a = _pose({
    "thumb_mcp": 0.39, "thumb_pip": 0.37, "thumb_abd": 0.93, "thumb_dip": 0.14,
    "index_pip": 0.23, "index_mcp": 0.82, "index_abd": 0.73,
    "middle_mcp": 0.90, "middle_pip": 0.17, "middle_abd": 0.48,
    "ring_mcp": 0.89, "ring_pip": 0.06, "ring_abd": 0.23,
    "pinky_mcp": 0.64, "pinky_pip": 0.20, "pinky_abd": 0.05,
    "wrist": 0.93,
})
    wave_b = _pose({
    "thumb_mcp": 0.39, "thumb_pip": 0.37, "thumb_abd": 0.91, "thumb_dip": 0.15,
    "index_pip": 0.15, "index_mcp": 0.19, "index_abd": 0.85,
    "middle_mcp": 0.20, "middle_pip": 0.18, "middle_abd": 0.47,
    "ring_mcp": 0.17, "ring_pip": 0.15, "ring_abd": 0.25,
    "pinky_mcp": 0.19, "pinky_pip": 0.18, "pinky_abd": 0.02,
    "wrist": 0.16,
})
    
    _hand.register_position("wave_a", wave_a)
    _hand.register_position("wave_b", wave_b)
    for _ in range(5):
        if _stop_event.is_set():
            return
        _hand.set_named_position("wave_a", num_steps=NUM_STEPS, step_size=STEP_SIZE)
        if _stop_event.wait(HOLD):
            return
        _hand.set_named_position("wave_b", num_steps=NUM_STEPS, step_size=STEP_SIZE)
        if _stop_event.wait(HOLD):
            return


def _finger_touch_demo():
    poses = {
        "touch_index": _pose({
            "thumb_mcp": 0.75, "thumb_pip": 0.57, "thumb_abd": 0.05, "thumb_dip": 0.03,
            "index_pip": 0.48, "index_mcp": 0.94, "index_abd": 0.77,
            "middle_mcp": 0.18, "middle_pip": 0.14, "middle_abd": 0.44,
            "ring_mcp": 0.18, "ring_pip": 0.15, "ring_abd": 0.27,
            "pinky_mcp": 0.17, "pinky_pip": 0.14, "pinky_abd": 0.04,
            "wrist": 0.68,
        }),
        "touch_middle": _pose({
            "thumb_mcp": 0.35, "thumb_pip": 0.61, "thumb_abd": 0.42, "thumb_dip": 0.58,
            "index_pip": 0.16, "index_mcp": 0.19, "index_abd": 0.85,
            "middle_mcp": 0.81, "middle_pip": 0.50, "middle_abd": 0.51,
            "ring_mcp": 0.24, "ring_pip": 0.16, "ring_abd": 0.27,
            "pinky_mcp": 0.19, "pinky_pip": 0.19, "pinky_abd": 0.03,
            "wrist": 0.65,
        }),
        "touch_ring": _pose({
            "thumb_mcp": 0.37, "thumb_pip": 0.60, "thumb_abd": 0.12, "thumb_dip": 0.75,
            "index_pip": 0.16, "index_mcp": 0.19, "index_abd": 0.85,
            "middle_mcp": 0.20, "middle_pip": 0.18, "middle_abd": 0.51,
            "ring_mcp": 0.72, "ring_pip": 0.70, "ring_abd": 0.25,
            "pinky_mcp": 0.22, "pinky_pip": 0.19, "pinky_abd": 0.03,
            "wrist": 0.65,
        }),
        "touch_pinky": _pose({
            "thumb_mcp": 0.33, "thumb_pip": 0.51, "thumb_abd": 0.02, "thumb_dip": 0.92,
            "index_pip": 0.16, "index_mcp": 0.19, "index_abd": 0.85,
            "middle_mcp": 0.20, "middle_pip": 0.18, "middle_abd": 0.51,
            "ring_mcp": 0.17, "ring_pip": 0.16, "ring_abd": 0.25,
            "pinky_mcp": 0.67, "pinky_pip": 0.56, "pinky_abd": 0.48,
            "wrist": 0.68,
        }),
    }
    for name, pose in poses.items():
        _hand.register_position(name, pose)

    for name in ("touch_index", "touch_middle", "touch_ring", "touch_pinky"):
        if _stop_event.is_set():
            return
        _hand.set_named_position(name, num_steps=NUM_STEPS, step_size=STEP_SIZE)
        if _stop_event.wait(TOUCH_HOLD):
            return
        _hand.set_neutral_position(num_steps=NUM_STEPS, step_size=STEP_SIZE)
        if _stop_event.wait(TOUCH_HOLD):
            return
    if not _stop_event.is_set():
        _hand.set_neutral_position(num_steps=NUM_STEPS, step_size=STEP_SIZE)


def _go_neutral():
    _hand.set_neutral_position(num_steps=NUM_STEPS, step_size=STEP_SIZE)


def _tryout():
    peace = _pose({
    "thumb_mcp": 0.63, "thumb_pip": 0.57, "thumb_abd": 0.18, "thumb_dip": 0.62,
    "index_pip": 0.16, "index_mcp": 0.20, "index_abd": 0.26,
    "middle_mcp": 0.18, "middle_pip": 0.17, "middle_abd": 0.67,
    "ring_mcp": 1.00, "ring_pip": 1.03, "ring_abd": 0.72,
    "pinky_mcp": 0.93, "pinky_pip": 0.77, "pinky_abd": 0.53,
    "wrist": 0.69,
})

    _hand.register_position("peace_sign", peace)
    _hand.set_named_position("peace_sign", num_steps=NUM_STEPS, step_size=STEP_SIZE)
    _stop_event.wait(HOLD)


HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ORCA Hand</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #111;
            color: #fff;
            font-family: system-ui, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 2rem 1rem;
            gap: 1.25rem;
        }
        h1 { font-size: 1.4rem; letter-spacing: 0.1em; color: #888; }
        #status {
            background: #1e1e1e;
            border-radius: 0.75rem;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            color: #4af;
            width: 100%;
            max-width: 420px;
            text-align: center;
        }
        .btn {
            width: 100%;
            max-width: 420px;
            padding: 1.5rem;
            font-size: 1.15rem;
            font-weight: 600;
            border: none;
            border-radius: 1rem;
            cursor: pointer;
            transition: opacity 0.15s;
            letter-spacing: 0.02em;
        }
        .btn:disabled { opacity: 0.35; cursor: not-allowed; }
        #btn-main    { background: #2a7a4a; color: #fff; }
        #btn-abd     { background: #2060a0; color: #fff; }
        #btn-tryout  { background: #8a5020; color: #fff; }
        #btn-neutral { background: #4a4a4a; color: #fff; }
        #btn-stop    { background: #8a2020; color: #fff; }
    </style>
</head>
<body>
    <h1>ORCA Hand</h1>
    <div id="status">Idle</div>
    <button class="btn" id="btn-main"   onclick="run('main-demo')">Spread Fingers</button>
    <button class="btn" id="btn-abd"    onclick="run('abduction-demo')">Finger Touch Demo</button>
    <button class="btn" id="btn-tryout"  onclick="run('tryout')">Peace Sign</button>
    <button class="btn" id="btn-neutral" onclick="run('neutral')">Neutral Position</button>
    <button class="btn" id="btn-stop"    onclick="stop()">Stop</button>
    <script>
        async function run(name) {
            await fetch('/run/' + name, { method: 'POST' });
        }
        async function stop() {
            await fetch('/stop', { method: 'POST' });
        }
        async function poll() {
            try {
                const r = await fetch('/status');
                const d = await r.json();
                const busy = d.task !== 'idle';
                document.getElementById('status').textContent = busy ? 'Running: ' + d.task : 'Idle';
                ['btn-main', 'btn-abd', 'btn-tryout', 'btn-neutral'].forEach(id => {
                    document.getElementById(id).disabled = busy;
                });
            } catch {}
        }
        setInterval(poll, 800);
        poll();
    </script>
</body>
</html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _hand
    _hand = create_hand(_args.config_path, use_mock=_args.mock)
    connect_hand(_hand)
    _hand.init_joints(force_calibrate=_args.mock)
    yield
    _stop_event.set()
    shutdown_hand(_hand)


app = FastAPI(lifespan=lifespan)

DEMOS = {
    "main-demo": _spread_fingers,
    "abduction-demo": _finger_touch_demo,
    "tryout": _tryout,
    "neutral": _go_neutral,
}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/run/{name}")
async def run(name: str):
    if name not in DEMOS:
        return JSONResponse({"error": "unknown demo"}, status_code=404)
    started = _run_task(name, DEMOS[name])
    return {"started": started}


@app.post("/stop")
async def stop_task():
    _stop_event.set()
    return {"stopped": True}


@app.get("/status")
async def status():
    return {"task": _state["task"]}


def main():
    global _args
    parser = argparse.ArgumentParser(description="ORCA Hand web control server")
    add_hand_arguments(parser)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    _args = parser.parse_args()
    uvicorn.run(app, host=_args.host, port=_args.port)


if __name__ == "__main__":
    main()
