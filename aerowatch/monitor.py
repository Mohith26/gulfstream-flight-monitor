"""Flight data exceedance monitor.

Streaming, per-sample state machine per exceedance type with:
  debounce    a condition must hold for min_duration consecutive samples
              before an event is confirmed
  hysteresis  separate trigger and clear predicates, with the clear
              threshold offset from the trigger threshold so noise near the
              limit does not chatter
  release     the clear predicate must hold for release consecutive samples
              before an open event closes
  merge       closed events of the same type separated by less than
              merge_gap_s seconds are merged into one event

Limits are GENERIC PLACEHOLDERS for a mid-size transport style aircraft.
They are not the limits of any real aircraft type.

Event times:
  start_t    when the trigger condition first became true (onset)
  confirm_t  when debounce was satisfied and the monitor actually flagged
             the event; detection latency is measured from label start to
             confirm_t
  end_t      last sample at which the trigger condition held
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

TYPES = (
    "overspeed_vmo",
    "overspeed_mmo",
    "high_sink_low_alt",
    "pitch_limit",
    "bank_limit",
    "unstable_approach",
)


@dataclass
class Limits:
    """Generic placeholder operating limits (not any real aircraft type)."""

    vmo_kt: float = 340.0        # max operating CAS, kt
    vmo_clear_kt: float = 337.0
    mmo: float = 0.85            # max operating Mach
    mmo_clear: float = 0.845
    sink_fpm: float = 2000.0     # max sink rate below sink_gate_agl_ft, fpm
    sink_clear_fpm: float = 1800.0
    sink_gate_agl_ft: float = 2000.0
    pitch_up_deg: float = 25.0
    pitch_up_clear_deg: float = 24.0
    pitch_down_deg: float = -10.0
    pitch_down_clear_deg: float = -9.0
    bank_deg: float = 45.0
    bank_clear_deg: float = 43.0
    appr_gate_agl_ft: float = 1000.0
    appr_ref_kt: float = 140.0   # placeholder approach reference speed
    appr_speed_tol_kt: float = 20.0
    appr_speed_clear_kt: float = 17.0
    appr_sink_fpm: float = 1200.0
    appr_sink_clear_fpm: float = 1100.0
    ground_agl_ft: float = 50.0  # below this the aircraft is treated as landing/ground


@dataclass
class Event:
    type: str
    start_t: float
    confirm_t: float
    end_t: float

    def overlaps(self, t0: float, t1: float, slack: float = 0.0) -> bool:
        return self.start_t <= t1 + slack and self.end_t >= t0 - slack


@dataclass
class DetectorSpec:
    name: str
    trigger: Callable[[Dict[str, np.ndarray]], np.ndarray]
    clear: Callable[[Dict[str, np.ndarray]], np.ndarray]
    min_duration: int = 4      # samples the trigger must hold to confirm
    release: int = 4           # samples the clear must hold to close
    merge_gap_s: float = 10.0  # events closer than this are merged


def detect_events(
    trigger: np.ndarray,
    clear: np.ndarray,
    t: np.ndarray,
    name: str,
    min_duration: int,
    release: int,
) -> List[Event]:
    """Run the streaming state machine over boolean trigger/clear arrays."""
    events: List[Event] = []
    state = "idle"  # idle -> pending -> active
    onset = -1
    confirm = -1
    last_true = -1
    hold = 0
    calm = 0
    n = len(trigger)
    for i in range(n):
        if state == "idle":
            if trigger[i]:
                state = "pending"
                onset = i
                hold = 1
        elif state == "pending":
            if trigger[i]:
                hold += 1
                if hold >= min_duration:
                    state = "active"
                    confirm = i
                    last_true = i
                    calm = 0
            else:
                state = "idle"  # blip rejected by debounce
        else:  # active
            if trigger[i]:
                last_true = i
                calm = 0
            elif clear[i]:
                calm += 1
                if calm >= release:
                    events.append(Event(name, t[onset], t[confirm], t[last_true]))
                    state = "idle"
            else:
                # inside the hysteresis band: neither triggering nor clearing
                calm = 0
    if state == "active":
        events.append(Event(name, t[onset], t[confirm], t[last_true]))
    return events


def merge_events(events: List[Event], merge_gap_s: float) -> List[Event]:
    """Merge same-type events whose gap is smaller than merge_gap_s."""
    if not events:
        return []
    events = sorted(events, key=lambda e: e.start_t)
    merged = [events[0]]
    for ev in events[1:]:
        prev = merged[-1]
        if ev.start_t - prev.end_t < merge_gap_s:
            merged[-1] = Event(prev.type, prev.start_t, prev.confirm_t, ev.end_t)
        else:
            merged.append(ev)
    return merged


def build_detectors(limits: Limits) -> List[DetectorSpec]:
    lm = limits

    def airborne(ch):
        return ch["agl_ft"] > lm.ground_agl_ft

    return [
        DetectorSpec(
            "overspeed_vmo",
            trigger=lambda ch: (ch["cas_kt"] > lm.vmo_kt) & airborne(ch),
            clear=lambda ch: (ch["cas_kt"] < lm.vmo_clear_kt) | ~airborne(ch),
        ),
        DetectorSpec(
            "overspeed_mmo",
            trigger=lambda ch: (ch["mach"] > lm.mmo) & airborne(ch),
            clear=lambda ch: (ch["mach"] < lm.mmo_clear) | ~airborne(ch),
        ),
        DetectorSpec(
            "high_sink_low_alt",
            trigger=lambda ch: (ch["vs_fpm"] < -lm.sink_fpm)
            & (ch["agl_ft"] < lm.sink_gate_agl_ft)
            & airborne(ch),
            clear=lambda ch: (ch["vs_fpm"] > -lm.sink_clear_fpm)
            | (ch["agl_ft"] >= lm.sink_gate_agl_ft)
            | ~airborne(ch),
        ),
        DetectorSpec(
            "pitch_limit",
            trigger=lambda ch: (
                (ch["pitch_deg"] > lm.pitch_up_deg)
                | (ch["pitch_deg"] < lm.pitch_down_deg)
            )
            & airborne(ch),
            clear=lambda ch: (
                (ch["pitch_deg"] < lm.pitch_up_clear_deg)
                & (ch["pitch_deg"] > lm.pitch_down_clear_deg)
            )
            | ~airborne(ch),
        ),
        DetectorSpec(
            "bank_limit",
            trigger=lambda ch: (np.abs(ch["bank_deg"]) > lm.bank_deg) & airborne(ch),
            clear=lambda ch: (np.abs(ch["bank_deg"]) < lm.bank_clear_deg)
            | ~airborne(ch),
        ),
        DetectorSpec(
            "unstable_approach",
            trigger=lambda ch: (ch["agl_ft"] < lm.appr_gate_agl_ft)
            & airborne(ch)
            & (ch["vs_fpm"] < 0.0)
            & (
                (ch["cas_kt"] > lm.appr_ref_kt + lm.appr_speed_tol_kt)
                | (ch["vs_fpm"] < -lm.appr_sink_fpm)
            ),
            clear=lambda ch: (ch["agl_ft"] >= lm.appr_gate_agl_ft)
            | ~airborne(ch)
            | (
                (ch["cas_kt"] < lm.appr_ref_kt + lm.appr_speed_clear_kt)
                & (ch["vs_fpm"] > -lm.appr_sink_clear_fpm)
            ),
        ),
    ]


@dataclass
class Monitor:
    limits: Limits = field(default_factory=Limits)

    def scan(self, flight) -> List[Event]:
        """Scan one flight (profiles.Flight) and return merged events."""
        ch = {
            "cas_kt": flight.cas_kt,
            "mach": flight.mach,
            "vs_fpm": flight.vs_fpm,
            "agl_ft": flight.agl_ft,
            "pitch_deg": flight.pitch_deg,
            "bank_deg": flight.bank_deg,
        }
        out: List[Event] = []
        for spec in build_detectors(self.limits):
            trig = np.asarray(spec.trigger(ch), dtype=bool)
            clr = np.asarray(spec.clear(ch), dtype=bool)
            raw = detect_events(
                trig, clr, flight.t, spec.name, spec.min_duration, spec.release
            )
            out.extend(merge_events(raw, spec.merge_gap_s))
        return sorted(out, key=lambda e: e.start_t)
