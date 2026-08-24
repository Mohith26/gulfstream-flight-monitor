"""Seeded synthetic flight profile generator with labeled injected exceedances.

Each flight is a multi-phase profile (ground roll, climb, cruise, descent,
approach, rollout) sampled at a fixed rate with Gaussian sensor noise per
channel. Exceedance episodes are injected by overriding the truth signal
over a window and recording a ground-truth label with the window times.

Two episode difficulty classes are generated:
  clear     amplitude well above the trigger threshold, 15 to 30 s
  marginal  amplitude about 2 noise sigma above the threshold, 4 to 8 s,
            which stresses the debounce logic and produces realistic misses

Everything is driven by numpy's seeded Generator, so a given seed always
reproduces the identical flight, labels included.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from . import airspeeds, isa
from .monitor import TYPES

DT = 0.5  # seconds per sample

# Phase ids
GROUND, CLIMB, CRUISE, DESCENT, APPROACH, ROLLOUT = range(6)

NOISE = {
    "cas_kt": 1.2,
    "mach": 0.003,
    "vs_fpm": 80.0,
    "pitch_deg": 0.4,
    "bank_deg": 0.5,
    "agl_ft": 15.0,
}


@dataclass
class Label:
    type: str
    t0: float
    t1: float
    marginal: bool = False


@dataclass
class Flight:
    seed: int
    t: np.ndarray
    agl_ft: np.ndarray
    cas_kt: np.ndarray
    mach: np.ndarray
    vs_fpm: np.ndarray
    pitch_deg: np.ndarray
    bank_deg: np.ndarray
    phase: np.ndarray
    labels: List[Label] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.t)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])


def _build_altitude(rng) -> tuple:
    """Integrate a piecewise vertical profile. Returns (agl_ft, phase)."""
    cruise_alt = float(rng.integers(330, 411)) * 100.0
    cruise_s = float(rng.uniform(480.0, 900.0))
    agl = [0.0]
    phase = [GROUND]
    # ground roll
    ground_n = int(30.0 / DT)
    for _ in range(ground_n - 1):
        agl.append(0.0)
        phase.append(GROUND)
    # climb
    alt = 0.0
    while alt < cruise_alt:
        if alt < 10000.0:
            vs = rng.uniform(2300.0, 2600.0)
        elif alt < 20000.0:
            vs = rng.uniform(1800.0, 2100.0)
        else:
            vs = rng.uniform(1300.0, 1500.0)
        alt = min(cruise_alt, alt + vs / 60.0 * DT)
        agl.append(alt)
        phase.append(CLIMB)
    # cruise
    for _ in range(int(cruise_s / DT)):
        agl.append(cruise_alt)
        phase.append(CRUISE)
    # descent to 1500 ft
    while alt > 1500.0:
        if alt > 20000.0:
            vs = rng.uniform(2100.0, 2400.0)
        elif alt > 10000.0:
            vs = rng.uniform(1900.0, 2100.0)
        elif alt > 3000.0:
            vs = rng.uniform(1400.0, 1600.0)
        else:
            vs = rng.uniform(650.0, 750.0)
        alt = max(1500.0, alt - vs / 60.0 * DT)
        agl.append(alt)
        phase.append(DESCENT)
    # approach 1500 ft to 50 ft
    while alt > 50.0:
        vs = rng.uniform(650.0, 750.0)
        alt = max(0.0, alt - vs / 60.0 * DT)
        agl.append(alt)
        phase.append(APPROACH)
    # rollout
    for _ in range(int(20.0 / DT)):
        agl.append(0.0)
        phase.append(ROLLOUT)
    return np.array(agl), np.array(phase), cruise_alt


def _baseline_cas(agl_ft, phase, cruise_mach) -> np.ndarray:
    """Speed schedule in KCAS as a function of phase and altitude."""
    alt_m = agl_ft * isa.FT
    cas_cruise_ms = airspeeds.cas_from_mach(cruise_mach, np.maximum(alt_m, 0.0))
    cas_cruise = np.asarray(cas_cruise_ms) / airspeeds.KT
    cas = np.zeros_like(agl_ft)

    ground = phase == GROUND
    n_ground = int(np.sum(ground))
    if n_ground:
        cas[ground] = np.linspace(0.0, 160.0, n_ground)

    climb = phase == CLIMB
    c_alt = agl_ft[climb]
    c = np.where(
        c_alt < 10000.0,
        250.0,
        np.where(
            c_alt < 12000.0,
            250.0 + (c_alt - 10000.0) / 2000.0 * 50.0,
            np.minimum(300.0, cas_cruise[climb]),
        ),
    )
    cas[climb] = c

    cruise = phase == CRUISE
    cas[cruise] = cas_cruise[cruise]

    descent = phase == DESCENT
    d_alt = agl_ft[descent]
    d = np.where(
        d_alt > 12000.0,
        np.minimum(295.0, cas_cruise[descent]),
        np.where(
            d_alt > 10000.0,
            250.0 + (d_alt - 10000.0) / 2000.0 * 45.0,
            np.where(
                d_alt > 3000.0,
                250.0,
                180.0 + (d_alt - 1500.0) / 1500.0 * 70.0,
            ),
        ),
    )
    cas[descent] = d

    approach = phase == APPROACH
    a_alt = agl_ft[approach]
    cas[approach] = np.where(
        a_alt > 1000.0, 140.0 + (a_alt - 1000.0) / 500.0 * 40.0, 140.0
    )

    rollout = phase == ROLLOUT
    n_roll = int(np.sum(rollout))
    if n_roll:
        cas[rollout] = np.linspace(130.0, 20.0, n_roll)
    return cas


def _baseline_pitch(phase) -> np.ndarray:
    return np.select(
        [phase == GROUND, phase == CLIMB, phase == CRUISE, phase == DESCENT,
         phase == APPROACH, phase == ROLLOUT],
        [1.0, 11.0, 2.5, -2.0, 1.5, 0.5],
    )


def _add_turns(rng, bank, phase):
    """Superimpose one to three gentle sine-shaped turns, max 22 degrees."""
    eligible = np.flatnonzero((phase == CLIMB) | (phase == CRUISE) | (phase == DESCENT))
    n_turns = int(rng.integers(1, 4))
    for _ in range(n_turns):
        dur = int(rng.uniform(30.0, 60.0) / DT)
        if len(eligible) <= dur:
            continue
        start = int(rng.choice(eligible[:-dur]))
        peak = rng.uniform(12.0, 22.0) * rng.choice([-1.0, 1.0])
        bank[start : start + dur] += peak * np.sin(
            np.linspace(0.0, np.pi, dur)
        )
    return bank


def _contiguous_runs(mask):
    """Yield (start, end) index pairs of contiguous True runs (end exclusive)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], splits + 1))
    ends = np.concatenate((splits + 1, [len(idx)]))
    return [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, ends)]


def _pick_window(rng, mask, dur_samples, used, pad_samples):
    """Pick a random window of dur_samples inside mask, avoiding used windows."""
    runs = [r for r in _contiguous_runs(mask) if r[1] - r[0] >= dur_samples]
    if not runs:
        return None
    for _ in range(30):
        r0, r1 = runs[int(rng.integers(0, len(runs)))]
        i0 = int(rng.integers(r0, r1 - dur_samples + 1))
        i1 = i0 + dur_samples
        if all(i1 + pad_samples <= u0 or i0 - pad_samples >= u1 for u0, u1 in used):
            used.append((i0, i1))
            return i0, i1
    return None


# Injection amplitudes. "clear" is far above the trigger threshold,
# "marginal" is about 2 noise sigma above it.
_AMPLITUDE = {
    "overspeed_vmo": {"clear": 348.0, "marginal": 342.5},
    "overspeed_mmo": {"clear": 0.862, "marginal": 0.856},
    "high_sink_low_alt": {"clear": -2600.0, "marginal": -2160.0},
    "pitch_limit": {"clear": 28.0, "marginal": 25.8},
    "bank_limit": {"clear": 52.0, "marginal": 46.0},
    "unstable_approach": {"clear": 172.0, "marginal": 162.5},
}


def _eligible_mask(ftype, agl, phase):
    if ftype == "overspeed_vmo":
        return (phase == DESCENT) & (agl > 10000.0) & (agl < 20000.0)
    if ftype == "overspeed_mmo":
        return phase == CRUISE
    if ftype == "high_sink_low_alt":
        return (agl > 1300.0) & (agl < 1950.0) & ((phase == DESCENT) | (phase == APPROACH))
    if ftype == "pitch_limit":
        return (phase == CLIMB) & (agl > 3000.0)
    if ftype == "bank_limit":
        return phase == CRUISE
    if ftype == "unstable_approach":
        return (phase == APPROACH) & (agl > 350.0) & (agl < 950.0)
    raise ValueError("unknown exceedance type %r" % ftype)


def generate_flight(
    seed: int,
    inject_types: Optional[Sequence[str]] = None,
    marginal_frac: float = 0.2,
) -> Flight:
    """Generate one flight. inject_types None means choose randomly, [] means clean."""
    rng = np.random.default_rng(seed)
    agl, phase, cruise_alt = _build_altitude(rng)
    n = len(agl)
    t = np.arange(n) * DT
    cruise_mach = float(rng.uniform(0.78, 0.82))

    cas = _baseline_cas(agl, phase, cruise_mach)
    pitch = _baseline_pitch(phase)
    bank = _add_turns(rng, np.zeros(n), phase)
    vs = np.gradient(agl, DT) * 60.0  # ft/min

    if inject_types is None:
        k = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
        inject_types = list(rng.choice(TYPES, size=k, replace=False))

    labels: List[Label] = []
    used: List[tuple] = []
    pad = int(10.0 / DT)
    pending_mach_windows = []
    for ftype in inject_types:
        marginal = bool(rng.random() < marginal_frac)
        dur_s = rng.uniform(4.0, 8.0) if marginal else rng.uniform(15.0, 30.0)
        if ftype in ("high_sink_low_alt", "unstable_approach"):
            dur_s = min(dur_s, 25.0)
        dur = max(2, int(dur_s / DT))
        win = _pick_window(rng, _eligible_mask(ftype, agl, phase), dur, used, pad)
        if win is None:
            continue
        i0, i1 = win
        amp = _AMPLITUDE[ftype]["marginal" if marginal else "clear"]
        if ftype == "overspeed_vmo":
            cas[i0:i1] = amp
        elif ftype == "overspeed_mmo":
            pending_mach_windows.append((i0, i1, amp))
        elif ftype == "high_sink_low_alt":
            vs[i0:i1] = amp
        elif ftype == "pitch_limit":
            pitch[i0:i1] = amp
        elif ftype == "bank_limit":
            bank[i0:i1] = amp * rng.choice([-1.0, 1.0])
        elif ftype == "unstable_approach":
            if rng.random() < 0.5:
                cas[i0:i1] = amp  # too fast on approach
            else:
                vs[i0:i1] = -1500.0 if not marginal else -1290.0  # too steep
        labels.append(Label(ftype, float(t[i0]), float(t[i1 - 1]), marginal))

    # Mach follows CAS and altitude, then direct Mach injections are applied.
    alt_m = np.maximum(agl, 0.0) * isa.FT
    mach = np.asarray(airspeeds.mach_from_cas(cas * airspeeds.KT, alt_m))
    for i0, i1, amp in pending_mach_windows:
        mach[i0:i1] = amp
        cas[i0:i1] = (
            np.asarray(airspeeds.cas_from_mach(amp, alt_m[i0:i1])) / airspeeds.KT
        )

    # Sensor noise
    airborne = agl > 0.0
    cas = cas + rng.normal(0.0, NOISE["cas_kt"], n)
    mach = mach + rng.normal(0.0, NOISE["mach"], n)
    vs = vs + rng.normal(0.0, NOISE["vs_fpm"], n)
    pitch = pitch + rng.normal(0.0, NOISE["pitch_deg"], n)
    bank = bank + rng.normal(0.0, NOISE["bank_deg"], n)
    agl_noisy = np.where(airborne, np.maximum(agl + rng.normal(0.0, NOISE["agl_ft"], n), 1.0), 0.0)

    return Flight(
        seed=seed,
        t=t,
        agl_ft=agl_noisy,
        cas_kt=np.maximum(cas, 0.0),
        mach=np.clip(mach, 0.0, 0.99),
        vs_fpm=vs,
        pitch_deg=pitch,
        bank_deg=bank,
        phase=phase,
        labels=labels,
    )


def generate_dataset(n_flights: int, seed: int, clean_frac: float = 0.25):
    """Generate a list of flights; about clean_frac of them have no injections."""
    rng = np.random.default_rng(seed)
    flights = []
    for i in range(n_flights):
        fseed = int(seed + 1000 + i)
        clean = bool(rng.random() < clean_frac)
        flights.append(generate_flight(fseed, inject_types=[] if clean else None))
    return flights
