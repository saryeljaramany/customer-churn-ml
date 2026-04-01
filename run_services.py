import os
import signal
import subprocess
import sys
import time


def start_process(
    command: list[str], env: dict[str, str], *, send_blank_line: bool = False
) -> subprocess.Popen:
    stdin = subprocess.PIPE if send_blank_line else subprocess.DEVNULL
    process = subprocess.Popen(command, env=env, stdin=stdin, text=True)
    if send_blank_line and process.stdin is not None:
        process.stdin.write("\n")
        process.stdin.flush()
    return process


def main() -> int:
    env = os.environ.copy()
    api_port = env.get("API_PORT", "8000")
    dashboard_port = env.get("DASHBOARD_PORT", "8501")
    api_host = env.get("API_HOST", "127.0.0.1")
    dashboard_host = env.get("DASHBOARD_HOST", "127.0.0.1")

    env.setdefault("API_BASE_URL", f"http://{api_host}:{api_port}")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.api:app",
        "--host",
        api_host,
        "--port",
        api_port,
    ]
    dashboard_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.address",
        dashboard_host,
        "--server.port",
        dashboard_port,
        "--browser.gatherUsageStats",
        "false",
    ]

    api_proc = start_process(api_cmd, env)
    dashboard_proc = start_process(dashboard_cmd, env, send_blank_line=True)

    processes = [api_proc, dashboard_proc]

    def shutdown(*_args: object) -> None:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            for proc in processes:
                if proc.poll() is not None:
                    shutdown()
                    return proc.returncode or 0
            time.sleep(0.5)
    finally:
        shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
