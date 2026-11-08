"""Spawns the number of workers specified in the command line arguments."""

import argparse
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int, help="Number of worker processes to start.")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be at least one")

    processes = []
    try:
        for i in range(1, args.count + 1):
            processes.append(subprocess.Popen([
                "atlas", "worker", "--wait-for-leases", "--id", f"worker-{i}",
            ]))
        exit_codes = [process.wait() for process in processes]
        return int(any(code != 0 for code in exit_codes))
    except KeyboardInterrupt:
        return 130
    except OSError as error:
        parser.exit(1, f"Cannot start workers: {error}\n")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
