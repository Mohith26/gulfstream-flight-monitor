# AeroWatch

I wanted to understand how flight data monitoring actually works under the hood, so I built a small version of the whole chain myself: a standard atmosphere and airspeed conversion library, a generator that produces realistic synthetic flight profiles with known injected exceedances, and a monitor that scans those flights and flags the exceedances. Because the generator labels every injected episode, I can score the monitor with real precision and recall numbers instead of just eyeballing plots.

## What is in here

- `aerowatch/isa.py`: International Standard Atmosphere (ISO 2533 / ICAO Doc 7488 model) with analytic layers up to 20 km geopotential, plus geometric/geopotential altitude conversion. Temperature, pressure, density, speed of sound, scalar or numpy array in, same shape out.
- `aerowatch/airspeeds.py`: CAS, EAS, TAS and Mach conversions using the subsonic compressible pitot relations, all routed through the ISA model.
- `aerowatch/profiles.py`: seeded synthetic flight generator. Multi-phase profiles (ground roll, climb, cruise, descent, approach, rollout) with Gaussian sensor noise, plus labeled injected exceedance episodes of six types.
- `aerowatch/monitor.py`: the exceedance monitor. Per-type streaming state machines with debounce, hysteresis (separate trigger and clear thresholds), a release counter, and merging of nearby events.
- `aerowatch/metrics.py`: event-level scoring. Greedy one to one matching of detections to labels, per-type precision/recall/F1, and detection latency statistics.
- `scripts/`: ISA validation against published tables, the labeled evaluation run, and a throughput benchmark. Each writes a JSON file into `results/`.

## The six exceedance types

Overspeed against VMO, overspeed against MMO, high sink rate at low altitude, pitch outside limits, bank angle limit, and unstable approach (too fast or too steep below the stabilization gate). The limits are generic placeholder numbers for a mid-size transport style aircraft, stated in `monitor.py`. They are deliberately not the limits of any real aircraft type.

## Why debounce and hysteresis matter

A naive threshold check on noisy data chatters: one event becomes twenty. The monitor requires the trigger condition to hold for several consecutive samples before it confirms an event, keeps the event open while the signal sits between the clear and trigger thresholds, and only closes it after the clear condition holds for several samples. Confirmed events of the same type that sit closer together than a gap window get merged. The evaluation set intentionally includes short, barely-over-the-limit episodes (about 2 noise sigma over threshold for 4 to 8 seconds) precisely to stress this logic.

## How I validated it

The ISA implementation is checked against 13 published table points from two independent sources: the ISO 2533 / ICAO Doc 7488 layer base values (0, 11000, 20000 m geopotential) and the Engineering ToolBox ISA table, which lists elevations as geometric altitude. On my run the worst pressure error across all 13 points is 0.015 percent and the worst density error is 0.039 percent, within the rounding of the published figures. Airspeed conversions are checked against hand-computed worked examples that are written out step by step in `tests/test_airspeeds.py`.

The monitor is scored on a seeded 200-flight dataset (1.32 million samples at 2 Hz) with 259 labeled episodes, 51 of them the hard marginal kind. Results from my run are in `results/eval.json`: precision 1.000, recall 0.996 overall, with one missed marginal unstable-approach episode. Median detection latency is 1.5 s, which is exactly the debounce time, and the worst case was 3.5 s on a marginal episode where noise reset the debounce counter.

## Running it

```
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install numpy pytest pytest-cov
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/validate_isa.py
.venv/bin/python scripts/run_eval.py --flights 200 --seed 42
.venv/bin/python scripts/bench.py
```

## Limitations

- The atmosphere model stops at 20 km and assumes ISA conditions everywhere. There is no temperature offset support, no humidity, and the airspeed relations are subsonic only (they raise on Mach 1 and above).
- The flights are synthetic. The generator produces plausible phase structure and noise, but the channels are not fully physically consistent with each other (an injected sink rate does not re-integrate the altitude trace), and nothing here reads real FDR or QAR formats.
- The near-perfect scores partly reflect that the same thresholds define both the injections and the detector. The marginal episode class is the honest part of the eval; a real-world monitor would face sensor dropouts, unit issues, and airframe-specific limits this project does not model.
- Limits are placeholders. Applying this to any real aircraft would mean replacing the whole `Limits` dataclass with certified numbers and re-tuning the debounce windows.
- Throughput numbers are from a single-threaded pure Python state machine with numpy predicate evaluation on one Apple silicon machine; treat them as machine specific.
