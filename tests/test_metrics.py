"""Event-level scoring tests."""

import pytest

from aerowatch.metrics import latency_stats, merge_scores, overall, score_flight
from aerowatch.monitor import Event
from aerowatch.profiles import Label


def test_perfect_match():
    labels = [Label("overspeed_vmo", 100.0, 120.0)]
    events = [Event("overspeed_vmo", 100.5, 102.0, 119.5)]
    s = score_flight(labels, events)["overspeed_vmo"]
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == 1.0
    assert s.latencies == [pytest.approx(2.0)]


def test_missed_label_counts_fn():
    labels = [Label("bank_limit", 50.0, 60.0)]
    s = score_flight(labels, [])["bank_limit"]
    assert (s.tp, s.fp, s.fn) == (0, 0, 1)
    assert s.recall == 0.0


def test_spurious_event_counts_fp():
    events = [Event("pitch_limit", 10.0, 11.5, 20.0)]
    s = score_flight([], events)["pitch_limit"]
    assert (s.tp, s.fp, s.fn) == (0, 1, 0)
    assert s.precision == 0.0


def test_wrong_type_does_not_match():
    labels = [Label("overspeed_vmo", 100.0, 120.0)]
    events = [Event("overspeed_mmo", 100.5, 102.0, 119.5)]
    scores = score_flight(labels, events)
    assert scores["overspeed_vmo"].fn == 1
    assert scores["overspeed_mmo"].fp == 1


def test_no_match_outside_slack():
    labels = [Label("bank_limit", 100.0, 110.0)]
    events = [Event("bank_limit", 130.0, 131.5, 140.0)]
    s = score_flight(labels, events, slack_s=5.0)["bank_limit"]
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)


def test_one_to_one_matching():
    labels = [Label("bank_limit", 100.0, 110.0)]
    events = [
        Event("bank_limit", 101.0, 102.5, 105.0),
        Event("bank_limit", 107.0, 108.5, 109.0),
    ]
    s = score_flight(labels, events)["bank_limit"]
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)


def test_merge_and_overall():
    a = score_flight([Label("bank_limit", 0.0, 10.0)],
                     [Event("bank_limit", 1.0, 2.5, 9.0)])
    b = score_flight([Label("pitch_limit", 0.0, 10.0)], [])
    total = merge_scores([a, b])
    agg = overall(total)
    assert (agg.tp, agg.fp, agg.fn) == (1, 0, 1)
    assert agg.recall == pytest.approx(0.5)


def test_latency_stats():
    st = latency_stats([1.0, 2.0, 3.0, 4.0])
    assert st["n"] == 4
    assert st["mean_s"] == pytest.approx(2.5)
    assert st["median_s"] == pytest.approx(2.5)
    assert st["max_s"] == pytest.approx(4.0)
    assert latency_stats([]) == {"n": 0}
