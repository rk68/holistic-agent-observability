from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from typing import Optional


def _cmd_serve(host: str, port: int, reload: bool) -> int:
    try:
        import uvicorn  # type: ignore
        # Prefer import string path so reload works reliably
        uvicorn.run(
            "observability_backend.trace_summary_server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"[glass-agent] Failed to start server: {exc}", file=sys.stderr)
        return 1


def _cmd_ask(prompt: str) -> int:
    try:
        from agent.factory import ask  # local import to avoid import cost on --help
        out = ask(prompt)
        print(out)
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"[glass-agent] ask failed: {exc}", file=sys.stderr)
        return 2


def _cmd_demo() -> int:
    try:
        from agent.demo import main as demo_main
        asyncio.run(demo_main())
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"[glass-agent] demo failed: {exc}", file=sys.stderr)
        return 3


def _cmd_up(host: str, port: int, ui_port: int, reload: bool) -> int:
    root = os.getcwd()
    ui_dir = os.path.join(root, "observability")
    pkg_json = os.path.join(ui_dir, "package.json")
    if not os.path.exists(pkg_json):
        print("[glass-agent] Could not find observability/package.json. Run from repo root.", file=sys.stderr)
        return 4

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "observability_backend.trace_summary_server:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        backend_cmd.append("--reload")

    # Pass port to Vite; works for `vite` and most wrapper scripts
    frontend_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--port",
        str(ui_port),
    ]

    print(f"[glass-agent] Starting backend on http://{host}:{port} …")
    be_proc = subprocess.Popen(backend_cmd, cwd=root)

    # Give backend a moment to start so UI can proxy if it needs
    time.sleep(0.8)

    print(f"[glass-agent] Starting frontend on http://127.0.0.1:{ui_port} …")
    fe_proc = subprocess.Popen(frontend_cmd, cwd=ui_dir)

    children = [be_proc, fe_proc]

    def _shutdown():
        for p in children:
            try:
                if p.poll() is None:
                    if os.name == "nt":
                        p.terminate()
                    else:
                        p.send_signal(signal.SIGINT)
            except Exception:
                pass
        # Wait briefly, then force kill any stragglers
        time.sleep(1.5)
        for p in children:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

    # Graceful Ctrl+C
    try:
        if os.name != "nt":
            signal.signal(signal.SIGINT, lambda *a, **k: None)
        while True:
            codes = [p.poll() for p in children]
            if any(c is not None for c in codes):
                # If one died, stop the other
                _shutdown()
                # Return first non-None code
                for c in codes:
                    if c is not None:
                        return c or 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        _shutdown()
        return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="glass-agent", description="Glass Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_srv = sub.add_parser("serve", help="Run the trace summary backend (FastAPI)")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--reload", action="store_true", help="Enable hot reload")

    p_ask = sub.add_parser("ask", help="Ask the agent a question")
    p_ask.add_argument("prompt", help="User question to send to the agent")

    sub.add_parser("demo", help="Interactive REPL demo")

    p_up = sub.add_parser("up", help="Run backend and frontend together")
    p_up.add_argument("--host", default="127.0.0.1")
    p_up.add_argument("--port", type=int, default=8000, help="Backend port")
    p_up.add_argument("--ui-port", type=int, default=5173, help="Frontend (Vite) port")
    p_up.add_argument("--reload", action="store_true", help="Enable backend hot reload")

    args = parser.parse_args(argv)

    if args.command == "serve":
        return _cmd_serve(args.host, args.port, args.reload)
    if args.command == "ask":
        return _cmd_ask(args.prompt)
    if args.command == "demo":
        return _cmd_demo()
    if args.command == "up":
        return _cmd_up(args.host, args.port, args.ui_port, args.reload)

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
