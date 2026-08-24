"""Monitor tests: per-type signatures, debounce, hysteresis, merge, determinism."""

import numpy as np
import pytest

from aerowatch.monitor import TYPES, Event, Monitor, detect_events, merge_events
from aerowatch.profiles import generate_flight

DT = 0.5


@pytest.mark.parametrize("ftype", TYPES)
def test_single_injection_detected_as_exactly_that_type(ftype):
    f = generate_flight(31, inject_types=[ftype], marginal_frac=0.0)
    assert [l.type for l in f.labels] == [ftype]
    events = Monitor().scan(f)
    assert len(events) == 1, "expected exactly one event, got %r" % events
    ev = events[0]
    assert ev.type == ftype
    lab = f.labels[0]
    assert ev.overlaps(lab.t0, lab.t1, slack=5.0)


@pytest.mark.parametrize("seed", [201, 202, 203])
def test_clean_flight_produces_no_events(seed):
    f = generate_flight(seed, inject_types=[])
    assert Monitor().scan(f) == []


def test_monitor_deterministic():
    f = generate_flight(55)
    e1 = Monitor().scan(f)
    e2 = Monitor().scan(f)
    assert [(e.type, e.start_t, e.confirm_t, e.end_t) for e in e1] == [
        (e.type, e.start_t, e.confirm_t, e.end_t) for e in e2
    ]


def _series(pattern):
    trig = np.array([c == "T" for c in pattern])
    clear = np.array([c == "C" for c in pattern])
    t = np.arange(len(pattern)) * DT
    return trig, clear, t


def test_debounce_rejects_short_blip():
    trig, clear, t = _series("CCTTTCCCCC")  # 3 trigger samples < min_duration 4
    assert detect_events(trig, clear, t, "x", 4, 4) == []


def test_debounce_accepts_exact_min_duration():
    trig, clear, t = _series("CCTTTTCCCC")
    evs = detect_events(trig, clear, t, "x", 4, 4)
    assert len(evs) == 1
    assert evs[0].start_t == pytest.approx(2 * DT)
    assert evs[0].confirm_t == pytest.approx(5 * DT)
    assert evs[0].end_t == pytest.approx(5 * DT)


def test_hysteresis_band_keeps_event_open():
    # After confirmation the signal drops into the band between clear and
    # trigger ("." samples): the event must stay open, not close or restart.
    trig, clear, t = _series("TTTT....TTCCCCC")
    evs = detect_events(trig, clear, t, "x", 4, 4)
    assert len(evs) == 1
    assert evs[0].end_t == pytest.approx(9 * DT)


def test_release_requires_consecutive_clear():
    # Clear run of 3 samples is interrupted, so the event closes only later.
    trig, clear, t = _series("TTTTCCCTCCCCC")
    evs = detect_events(trig, clear, t, "x", 4, 4)
    assert len(evs) == 1
    assert evs[0].end_t == pytest.approx(7 * DT)


def test_detection_latency_is_debounce_time():
    trig, clear, t = _series("CCCCTTTTTTTTCCCCC")
    evs = detect_events(trig, clear, t, "x", 4, 4)
    assert len(evs) == 1
    assert evs[0].confirm_t - evs[0].start_t == pytest.approx((4 - 1) * DT)


def test_two_separate_events():
    trig, clear, t = _series("TTTTTCCCCCCCCCCCCCCCCCCCCTTTTTCCCCC")
    evs = detect_events(trig, clear, t, "x", 4, 4)
    assert len(evs) == 2


def test_merge_close_events():
    evs = [Event("x", 0.0, 1.5, 10.0), Event("x", 15.0, 16.5, 20.0)]
    merged = merge_events(evs, merge_gap_s=10.0)
    assert len(merged) == 1
    assert merged[0].start_t == 0.0
    assert merged[0].end_t == 20.0
    assert merged[0].confirm_t == 1.5


def test_no_merge_across_large_gap():
    evs = [Event("x", 0.0, 1.5, 10.0), Event("x", 40.0, 41.5, 50.0)]
    assert len(merge_events(evs, merge_gap_s=10.0)) == 2


def test_open_event_at_end_of_data_is_reported():
    trig, clear, t = _series("CCCCTTTTTT")
    evs = detect_events(trig, clear, t, "x", 4, 4)
    assert len(evs) == 1
    assert evs[0].end_t == pytest.approx(9 * DT)


def test_events_sorted_by_start_time():
    f = generate_flight(77, inject_types=["unstable_approach", "overspeed_mmo"],
                        marginal_frac=0.0)
    events = Monitor().scan(f)
    starts = [e.start_t for e in events]
    assert starts == sorted(starts)
