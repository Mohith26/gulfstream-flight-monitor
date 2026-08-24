# Results and benchmark notes

Everything below comes from runs on my machine: Apple silicon (arm64), Python 3.9.6, numpy 2.0.2, single thread. JSON artifacts for each run are committed in `results/`.

## ISA validation

Command:

```
.venv/bin/python scripts/validate_isa.py
```

13 published reference points across two sources (ISO 2533 / ICAO Doc 7488 layer bases at 0, 11000, 20000 m geopotential; Engineering ToolBox ISA table at 10 geometric altitudes from 1000 to 20000 m). Measured worst-case deviations:

- temperature: 0.050 K (the published table rounds to 0.1 K)
- pressure: 0.0149 percent relative
- density: 0.0394 percent relative
- all 13 points within the stated tolerances: yes

One thing I had to figure out during validation: the Engineering ToolBox table lists elevation as geometric altitude, not geopotential. The giveaway is its 216.8 K at 11000 m, which is the ISA temperature at the corresponding geopotential altitude of 10981 m. Feeding the tabulated elevations through the geometric to geopotential conversion first is what makes all 13 points line up.

## Exceedance detection

Command:

```
.venv/bin/python scripts/run_eval.py --flights 200 --seed 42
```

Dataset: 200 seeded flights, 50 of them clean, 1,323,540 samples total at 2 Hz, 259 labeled injected episodes of which 51 are marginal (about 2 noise sigma over threshold, 4 to 8 s duration). Matching slack 5 s, one to one greedy matching per type.

| type | tp | fp | fn | precision | recall | f1 |
|---|---|---|---|---|---|---|
| overspeed_vmo | 50 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| overspeed_mmo | 44 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| high_sink_low_alt | 43 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| pitch_limit | 42 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| bank_limit | 35 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| unstable_approach | 44 | 0 | 1 | 1.000 | 0.978 | 0.989 |
| overall | 258 | 0 | 1 | 1.000 | 0.996 | 0.998 |

The single miss is a marginal unstable-approach episode (about 1.1 sigma of margin on the sink-rate variant), where noise kept breaking the 4-sample debounce run. I consider that a correct trade: the same debounce is what keeps the 50 clean flights at zero false positives.

Detection latency over the 258 matched episodes, measured from label start to monitor confirmation (debounce included): mean 1.52 s, median 1.50 s, p95 1.50 s, max 3.50 s. The floor is 1.5 s by construction: 4 samples at 2 Hz must agree before the monitor commits.

## Throughput

Command:

```
.venv/bin/python scripts/bench.py
```

20 flights, 134,277 samples, best of 3 batch runs, single thread:

- 5.66 million samples/s through the full six-detector monitor
- mean full-flight scan: 1.23 ms for a 6,714-sample flight (about 56 minutes of flight data at 2 Hz)
- max flight scan: 1.45 ms

The state machine is a per-sample Python loop over numpy boolean arrays, so this number is mostly a statement about how cheap the predicates are after vectorization. Run to run I saw the samples/s figure move a few percent; the committed JSON is the run described here.

## Tests

```
.venv/bin/python -m pytest tests/ --color=no -q --cov=aerowatch --cov-report=term
```

77 tests pass, 98 percent line coverage over the `aerowatch` package (the misses are defensive branches in metrics and rarely-hit generator fallbacks). The suite covers the 13 ISA table points, the hand-worked airspeed fixtures, one signature test per exceedance type, debounce and hysteresis edge cases on hand-built boolean sequences, merge rules, generator determinism, and scoring logic.
