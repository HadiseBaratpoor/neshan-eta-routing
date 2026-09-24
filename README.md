# Time-dependent ETA prediction and routing

A routing engine that answers the question a navigation app is actually for:
**given where I am, where I'm going, and what today looks like — how long will
it take, which way should I go, and when should I leave?**

Built for the Neshan data-science challenge: a road graph, aggregated user
speeds, and a set of test cases to answer.

```
src, dest, departure minute, holiday flag, weather  ->  eta (seconds), route (node list)
```

---

## The short version

| | |
| --- | --- |
| **Speed model** | Hierarchical multiplicative profiles — 13.8% better than a per-edge median baseline on held-out days |
| **Routing** | Time-dependent Dijkstra: edges are priced at the moment you actually reach them, not at departure |
| **Departure planning** | Picking the best minute in a 2-hour window saves **8.5%** of travel time on average |
| **Tests** | 32 unit tests covering unit conversion, FIFO correctness, optimality, and submission format |

The interesting finding: **choosing *when* to leave is worth several times more
than choosing *which way* to drive.** Re-routing saves ~1.4% on the trips where
it changes the path; shifting the departure minute saves 8.5% on a typical
trip, and 14.2% at the 90th percentile.

---

## Quick start

```bash
pip install -r requirements.txt

# No data? Generate a synthetic city with the same schema.
python scripts/make_synthetic_data.py

python scripts/explore_data.py        # EDA + figures
python scripts/train_speed_model.py   # fit + held-out validation
python scripts/predict_submission.py  # fill test_cases.csv
python scripts/evaluate.py            # score against ground truth
pytest tests/ -q
```

### Using the real challenge data

Drop `network.gpickle`, `dataset.csv` and `test_cases.csv` into `data/raw/`,
replacing the generated files. The loaders normalise column order, dtypes, and
directed/undirected graphs.

> **Security:** Python pickle files can execute arbitrary code when loaded.
> Inspect and trust graph/model pickle files before use. CLI commands that load
> a model supplied with `--model` require `--trust-pickle`; library callers must
> pass `trusted_pickle=True` to `load_graph`. Never use either acknowledgement
> for downloaded or otherwise untrusted files.

> The repository ships a **synthetic data generator** because the challenge
> files are not public. It reproduces the real structure — rush-hour peaks
> that sweep across the city, weather effects, a congested downtown core,
> ~3% missing cells and ~1% injected outliers — so the pipeline runs end to
> end on a clean clone and the tests have something deterministic to assert
> against. Every number in this README comes from an actual run on it.

---

## Ask it a question

```bash
# When should I leave, between 06:00 and 11:00?
python scripts/plan_trip.py --src 0 --dest 195 --window 360 660 --step 20
```

```
0 -> 195  (sunny, working day)
----------------------------------------------------------------
Departure sweep 06:00 - 11:00 every 20 min
  best: leave at 11:00 -> 37.5 min | worst: leave at 08:20 -> 41.8 min | saving 4.3 min (10.4%)

  depart   travel   arrive
  06:00     38.5m  06:38  ##########################
  06:20     39.6m  06:59  ###########################
  06:40     40.6m  07:20  ###########################
  07:00     41.5m  07:41  ############################
  07:20     41.8m  08:01  ############################
  ...
```

```bash
# One trip, and what the naive approach would have cost
python scripts/plan_trip.py --src 0 --dest 195 --depart 480 --weather 1 --compare-static

# The meeting is at 09:00 — when do I have to leave?
python scripts/plan_trip.py --src 0 --dest 195 --arrive-by 540
```

---

## How it works

### 1. Cleaning

Aggregated probe speeds are dirty in two ways, so two filters run in sequence:
physically impossible values (outside 1–130 km/h) are dropped outright, then a
**median-absolute-deviation** filter removes statistical outliers within each
`(edge, weather, holiday)` group. The MAD is used instead of a standard
deviation because a handful of extreme values inflates a standard deviation
enough to hide themselves.

### 2. Speed model

Rather than a black-box regressor over a high-cardinality `edge_id` — which
memorises and cannot extrapolate to unseen edge-condition pairs — speed is
decomposed multiplicatively:

```
speed = city(time, weather, holiday) x edge_level x edge_shape(time) x residual
```

Edge factors are learned from **condition-normalised** observations, so an edge
only ever seen in sunny weather still produces a sensible snowy estimate. Gaps
are filled by **circular Gaussian smoothing** over time bins (midnight wraps
onto 23:59), and sparse cells are **shrunk towards their parent level** with a
weight `n / (n + k)` where `k` adapts to the dataset's density.

Validation holds out **whole days**, not random rows. A random split leaks —
the same edge-minute-weather cell recurs across days — and would report a far
too optimistic error.

### 3. Routing

Time-dependent Dijkstra, where a node's label is its **arrival time** and each
edge is priced at the moment the traveller actually reaches it:

```
arrival(v) = arrival(u) + tau(u, v, arrival(u))
```

Optimality needs two guarantees, and both are enforced rather than assumed:
weights are strictly positive by construction (`length / speed`, speed clamped
positive), and the profile is **FIFO-corrected** so that leaving later can
never mean arriving earlier — an interpolated profile can otherwise violate
this at a sharp rush-hour edge and silently break the algorithm.

---

## Results

Speed prediction, held-out days:

| Model | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| Hierarchical profile model | **6.46 km/h** | 8.46 km/h | 19.9% |
| Per-edge median baseline | 7.49 km/h | 9.16 km/h | 23.1% |

ETA and routing, 300 test cases:

| Metric | Time-dependent | Static baseline |
| --- | --- | --- |
| ETA MAE | 128.1 s | 130.0 s |
| ETA MAPE | 5.56% | 5.72% |
| Mean true travel time | 1,789.0 s | 1,792.7 s |

Routes differ on 15.3% of cases; on those the time-dependent route is 24.4 s
faster.

Departure planning, 2-hour window:

| Metric | Value |
| --- | --- |
| Mean saving, best vs. worst departure | **8.5%** |
| 90th percentile | 14.2% |

**On reading the routing gain honestly.** It is real but modest, and that is a
property of the test city rather than the algorithm. On a uniform grid most
detours are near-equivalent. An **oracle experiment** — substituting the true
generating function for the fitted model — raises the gain only to ~0.6%, which
bounds what *any* speed model could achieve on this network. A real city with
ring roads and river crossings offers far more for the same algorithm to
exploit. The departure-time result needs no such caveat: it exploits temporal
structure, which the grid has in abundance.

Full discussion in [`docs/method.md`](docs/method.md).

---

## Submission format

The brief is strict: column names, column count and row count must not change,
and only `eta` and `route` are filled in. `submission.py` writes
`route` as `[15,45,78,35]` (both endpoints included, no spaces) and `eta` as an
integer number of seconds, then **validates the result before writing** —
row count, column set, positive ETAs, bracket format, and that every route's
endpoints match its `src`/`dest`. `predict_submission.py` refuses to write a
file that fails any of those checks.

---

## Layout

```
src/eta_router/
    config.py        all tunables, with the reasoning for each
    data.py          defensive loaders for graph / dataset / test cases
    cleaning.py      physical bounds + MAD outlier removal
    speed_model.py   hierarchical multiplicative speed profiles
    travel_time.py   speed -> seconds, FIFO enforcement
    routing.py       time-dependent Dijkstra + static baseline
    departure.py     "when should I leave?"
    evaluation.py    day-wise splits, ETA and route metrics
    submission.py    exact output format + validation
    synthetic.py     schema-exact synthetic city generator

scripts/
    make_synthetic_data.py   generate data/raw/
    explore_data.py          EDA + figures
    train_speed_model.py     fit + validate + save
    predict_submission.py    answer every test case
    evaluate.py              score vs. ground truth and baseline
    plan_trip.py             interactive CLI

tests/                       32 unit tests
docs/method.md               full write-up
```

---

## Requirements

Python 3.10+, `networkx`, `pandas`, `numpy`, `scikit-learn`, `scipy`,
`matplotlib`, `pytest`. No GPU, no training loop — the whole pipeline runs in
under a minute on a laptop.
