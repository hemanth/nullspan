"""HTTP `/v1/systemone` server for `gnumanth/nullspan` (TypeSafe / JevBench / Typed Decisions wire format).

Exposes:
  - GET  /health        -> {"status": "ok", "model": "gnumanth/nullspan"}
  - POST /v1/systemone  -> {"model": ..., "answers": {...}, "usage": {"input_tokens": ..., "output_tokens": 0}}

Compatible out-of-the-box with:
  - `fstandhartinger/jevbench` (`--adapter typesafe --endpoint http://127.0.0.1:8010 --key-env ''`)
  - `LocalLLaMA/typed-decisions` (`bench/score_with_jobs.py`)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from systemone_decision import NullspanSystemOneEngine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="gnumanth/nullspan", help="Local path or HF repo ID (e.g. gnumanth/nullspan)")
    ap.add_argument("--revision", default=None, help="Optional immutable Hugging Face commit SHA")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    engine = NullspanSystemOneEngine.load_from_dir(args.model_dir, device=args.device, revision=args.revision)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/health"):
                body = json.dumps({"status": "ok", "model": "gnumanth/nullspan", "device": engine.device}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if not self.path.startswith("/v1/systemone"):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                state = payload.get("state", "")
                questions = payload.get("questions") or {}
                out = engine.decide(state=state, questions=questions)
                if payload.get("model"):
                    out["model"] = payload["model"]
                raw = json.dumps(out).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception as exc:
                err = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)

        def log_message(self, format: str, *args: object) -> None:
            pass

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"serving": "gnumanth/nullspan", "endpoint": f"http://{args.host}:{args.port}/v1/systemone", "device": engine.device}), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
