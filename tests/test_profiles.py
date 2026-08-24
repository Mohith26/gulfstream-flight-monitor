"""Profile generator tests: determinism, label sanity, phase structure."""

import numpy as np

from aerowatch.monitor import TYPES
from aerowatch.profiles import (
    APPROACH,
    CLIMB,
    CRUISE,
    DESCENT,
    GROUND,
    ROLLOUT,
    generate_dataset,
    generate_flight,
)


def test_same_seed_identical():
    a = generate_flight(123)
    b = generate_flight(123)
    assert np.array_equal(a.cas_kt, b.cas_kt)
    assert np.array_equal(a.mach, b.mach)
    assert np.array_equal(a.vs_fpm, b.vs_fpm)
    assert np.array_equal(a.agl_ft, b.agl_ft)
    assert [(l.type, l.t0, l.t1) for l in a.labels] == [
        (l.type, l.t0, l.t1) for l in b.labels
    ]


def test_different_seed_differs():
    a = generate_flight(123)
    b = generate_flight(124)
    assert a.n_samples != b.n_samples or not np.array_equal(a.cas_kt, b.cas_kt)


def test_labels_inside_flight_and_valid_types():
    f = generate_flight(5, inject_types=list(TYPES))
    assert f.labels, "expected at least one injected episode"
    for lab in f.labels:
        assert lab.type in TYPES
        assert 0.0 <= lab.t0 < lab.t1 <= f.duration_s


def test_clean_flight_has_no_labels():
    f = generate_flight(9, inject_types=[])
    assert f.labels == []


def test_phases_present_and_ordered():
    f = generate_flight(7, inject_types=[])
    for ph in (GROUND, CLIMB, CRUISE, DESCENT, APPROACH, ROLLOUT):
        assert np.any(f.phase == ph)
    firsts = [int(np.argmax(f.phase == ph)) for ph in
              (GROUND, CLIMB, CRUISE, DESCENT, APPROACH, ROLLOUT)]
    assert firsts == sorted(firsts)


def test_agl_non_negative():
    f = generate_flight(11, inject_types=[])
    assert np.all(f.agl_ft >= 0.0)


def test_dataset_deterministic_and_mixed():
    d1 = generate_dataset(10, seed=42)
    d2 = generate_dataset(10, seed=42)
    assert [f.n_samples for f in d1] == [f.n_samples for f in d2]
    assert [len(f.labels) for f in d1] == [len(f.labels) for f in d2]
    assert any(not f.labels for f in d1)
    assert any(f.labels for f in d1)
