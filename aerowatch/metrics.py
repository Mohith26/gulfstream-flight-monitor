"""Event-level scoring of detections against ground-truth labels.

Matching rule: a detection matches a label when both have the same type and
their time intervals overlap after widening the label by slack_s seconds on
each side. Matching is greedy in time order and one to one: each label can
absorb at most one detection and vice versa.

Latency for a matched pair is confirm_t - label.t0: how long after the
episode began the monitor actually flagged it (debounce included).
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .monitor import TYPES, Event
from .profiles import Label


@dataclass
class TypeScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    latencies: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)


def score_flight(labels: List[Label], events: List[Event], slack_s: float = 5.0
                 ) -> Dict[str, TypeScore]:
    scores = {ftype: TypeScore() for ftype in TYPES}
    for ftype in TYPES:
        labs = sorted([l for l in labels if l.type == ftype], key=lambda l: l.t0)
        evs = sorted([e for e in events if e.type == ftype], key=lambda e: e.start_t)
        used = [False] * len(evs)
        for lab in labs:
            hit = None
            for j, ev in enumerate(evs):
                if used[j]:
                    continue
                if ev.overlaps(lab.t0, lab.t1, slack=slack_s):
                    hit = j
                    break
            if hit is None:
                scores[ftype].fn += 1
            else:
                used[hit] = True
                scores[ftype].tp += 1
                scores[ftype].latencies.append(evs[hit].confirm_t - lab.t0)
        scores[ftype].fp += used.count(False)
    return scores


def merge_scores(per_flight: List[Dict[str, TypeScore]]) -> Dict[str, TypeScore]:
    total = {ftype: TypeScore() for ftype in TYPES}
    for scores in per_flight:
        for ftype, s in scores.items():
            total[ftype].tp += s.tp
            total[ftype].fp += s.fp
            total[ftype].fn += s.fn
            total[ftype].latencies.extend(s.latencies)
    return total


def overall(total: Dict[str, TypeScore]) -> TypeScore:
    agg = TypeScore()
    for s in total.values():
        agg.tp += s.tp
        agg.fp += s.fp
        agg.fn += s.fn
        agg.latencies.extend(s.latencies)
    return agg


def latency_stats(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"n": 0}
    arr = np.array(latencies)
    return {
        "n": int(len(arr)),
        "mean_s": float(np.mean(arr)),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "max_s": float(np.max(arr)),
    }
