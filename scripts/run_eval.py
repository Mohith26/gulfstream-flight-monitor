"""Run the monitor over a seeded synthetic dataset and score it.

Writes results/eval.json with per-type precision/recall/F1, detection
latency statistics, and dataset composition.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aerowatch import metrics
from aerowatch.monitor import TYPES, Monitor
from aerowatch.profiles import generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slack", type=float, default=5.0)
    args = ap.parse_args()

    flights = generate_dataset(args.flights, args.seed)
    n_episodes = sum(len(f.labels) for f in flights)
    n_clean = sum(1 for f in flights if not f.labels)
    n_samples = sum(f.n_samples for f in flights)
    n_marginal = sum(1 for f in flights for l in f.labels if l.marginal)

    mon = Monitor()
    per_flight = []
    for f in flights:
        events = mon.scan(f)
        per_flight.append(metrics.score_flight(f.labels, events, slack_s=args.slack))

    total = metrics.merge_scores(per_flight)
    agg = metrics.overall(total)

    def score_dict(s):
        return {
            "tp": s.tp,
            "fp": s.fp,
            "fn": s.fn,
            "precision": s.precision,
            "recall": s.recall,
            "f1": s.f1,
            "latency": metrics.latency_stats(s.latencies),
        }

    out = {
        "seed": args.seed,
        "n_flights": args.flights,
        "n_clean_flights": n_clean,
        "n_samples": n_samples,
        "n_labeled_episodes": n_episodes,
        "n_marginal_episodes": n_marginal,
        "match_slack_s": args.slack,
        "per_type": {ftype: score_dict(total[ftype]) for ftype in TYPES},
        "overall": score_dict(agg),
    }
    path = os.path.join(os.path.dirname(__file__), "..", "results", "eval.json")
    with open(path, "w") as fjson:
        json.dump(out, fjson, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
