"""Three isolated ARQ processes in the worker service, with coordinated shutdown."""

import signal
import subprocess
import time


def main() -> None:
    processes = [
        subprocess.Popen(["arq", f"app.workers.{name}.WorkerSettings"])
        for name in ("mail", "ingestion", "chat")
    ]
    stopping = False

    def stop(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping and all(process.poll() is None for process in processes):
            time.sleep(0.5)
    finally:
        failed = not stopping
        stop(signal.SIGTERM, None)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
