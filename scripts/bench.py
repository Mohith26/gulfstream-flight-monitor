"""Benchmark monitor throughput: samples/sec and full-flight scan time.

Writes results/bench.json. Numbers are single-threaded pure Python plus
numpy predicate evaluation, measured on the machine noted in the output.
"""

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aerowatch.monitor import Monitor
from aerowatch.profiles import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    flights = generate_dataset(args.flights, args.seed)
    n_samples = sum(f.n_samples for f in flights)
    mon = Monitor()
    # warmup
    mon.scan(flights[0])

    best = None
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        for f in flights:
            mon.scan(f)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)

    per_flight_ms = [None] * len(flights)
    for i, f in enumerate(flights):
        t0 = time.perf_counter()
        mon.scan(f)
        per_flight_ms[i] = (time.perf_counter() - t0) * 1000.0

    out = {
        "seed": args.seed,
        "n_flights": args.flights,
        "n_samples_total": n_samples,
        "repeats": args.repeats,
        "best_batch_time_s": best,
        "samples_per_sec": n_samples / best,
        "mean_flight_scan_ms": sum(per_flight_ms) / len(per_flight_ms),
        "max_flight_scan_ms": max(per_flight_ms),
        "mean_flight_samples": n_samples / len(flights),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "note": "single-threaded, pure Python state machine with numpy predicates",
    }
    path = os.path.join(os.path.dirname(__file__), "..", "results", "bench.json")
    with open(path, "w") as fjson:
        json.dump(out, fjson, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
