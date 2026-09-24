# Method

How the system turns a road graph and a pile of aggregated user speeds into
an ETA and a route, and why each choice was made.

---

## 1. The problem

Given a source node, a destination node, a departure minute, a holiday flag
and a weather code, produce:

* **eta** — estimated travel time in **seconds**
* **route** — the list of nodes traversed, including both endpoints

Two sub-problems sit underneath, and they are not equally hard:

1. **Estimate speed** for every `(edge, minute, weather, holiday)` — including
   the many combinations never observed.
2. **Find the best path** when edge costs change while the traveller is
   still driving.

The second is a solved algorithmic problem once the first is done properly.
Almost all of the achievable accuracy lives in step 1.

---

## 2. Data preparation

### Units

Speeds are km/h, lengths are metres, and the answer is seconds:

```
seconds = length_m / (speed_kmh / 3.6)
```

This conversion lives in exactly one function, `travel_time.seconds_from_speed`,
and is pinned by hand-checked unit tests. A factor-of-3.6 error is invisible
in a notebook and fatal in scoring.

### Outlier removal

Aggregated probe speeds are dirty in two distinct ways, so two filters run in
sequence:

**Physical bounds.** Anything outside `[1, 130]` km/h is discarded outright.
A 0.2 km/h reading is a parked vehicle, a 260 km/h reading is GPS drift or a
map-matching error. On the reference dataset this removes ~0.7% of rows.

**Robust statistical filtering.** Within each `(edge_id, weather, is_holiday)`
group, points whose *modified z-score* exceeds 3.5 are dropped:

```
modified_z = |x - median| / (1.4826 * MAD)
```

The median and MAD are used rather than the mean and standard deviation
because a few extreme values inflate a standard deviation enough to hide
themselves — the classic masking effect. Groups smaller than 8 observations
skip this step, since a MAD computed from 3 points is meaningless. A special
case handles zero-MAD groups (a street where nearly everyone drives the same
speed) so that a constant group is not flagged as entirely anomalous.

Together these remove ~1.8% of rows on the reference data and leave a clean
speed range of roughly 7–100 km/h.

---

## 3. Speed model

### Why not one big regressor

The obvious move is gradient boosting on
`(edge_id, minute, weather, is_holiday)`. It is the wrong tool here:

* `edge_id` becomes a categorical with hundreds or thousands of levels, and
  with a few hundred rows each the model memorises rather than generalises —
  the overfitting the brief explicitly warns about;
* it has no principled answer for an edge-condition pair that never appears
  in training, which is common in this dataset;
* it gives no interpretable structure — a traffic engineer cannot read a
  daily profile out of it.

### The estimator used

A **multiplicative profile decomposition**:

```
speed(edge, bin, weather, holiday)
    = city(bin, weather, holiday)     # when the whole city slows
    x edge_level(edge)                # is this street fast or slow
    x edge_shape(edge, bin)           # its own rush-hour rhythm
    x residual(edge, bin, weather, holiday)
```

Each factor is fitted on the residual left by the factors above it, so the
levels complement rather than compete.

The important detail is that the edge factors are learned from
**condition-normalised** observations. Every speed is first divided by the
city profile for *its own* weather and holiday state, which strips those
effects out. What remains is "how does this street compare to the city right
now" — a quantity that transfers. An edge observed only on sunny working days
therefore still produces a sensible snowy-holiday estimate, because the snow
effect comes from the city level where there is plenty of data.

### Robustness, and the imputation the brief asks for

Three mechanisms, each doing a specific job:

**Medians everywhere.** Every factor is a median, so residual outliers that
survived cleaning cannot drag an estimate.

**Circular Gaussian smoothing.** Each daily profile is convolved with a
Gaussian kernel over time bins, wrapping at midnight so 23:55 and 00:05 are
neighbours. This is the imputation step: an unobserved bin borrows from its
neighbours in time rather than returning `NaN`. Bin width is 10 minutes and
the kernel σ is 1.5 bins.

**Shrinkage towards the parent level.** A cell with little evidence is pulled
towards the level above it:

```
w = n / (n + k)
estimate = w * own_median + (1 - w) * parent_estimate
```

`k` is **set adaptively at fit time** as a multiple of the observations behind
a typical cell, rather than being a fixed constant. This matters: a `k` tuned
on a 1.4M-row export would barely regularise a 20k-row one. The tunable is the
multiplier (1.5, selected on held-out days), and it transfers across dataset
sizes.

The result is a smoother, not a fitter. There is no capacity to memorise
individual rows, which is precisely why held-out error stays close to
training error.

### Validation: day-wise, not random

The split holds out **whole days**. A random row split would leak badly — the
same `(edge, minute, weather)` cell recurs across days, so a random validation
row almost always has a near-duplicate in training, and the reported error
would be far too optimistic. Holding out whole days is the honest analogue of
"predict tomorrow".

---

## 4. Routing

### Why classical Dijkstra is wrong here

The brief's closing hint: classical Dijkstra freezes edge weights at departure
time. A route chosen at 08:00 prices *every* edge with 08:00 traffic, even the
one reached at 08:40. On a congested network that is exactly when the error
is largest.

### Time-dependent Dijkstra

The label of a node is its **arrival time**, and an edge is priced at the
moment the traveller actually reaches its tail:

```
arrival(v) = arrival(u) + tau(u, v, arrival(u))
```

This remains a correct label-setting algorithm — the first time a node leaves
the priority queue its arrival time is optimal — provided two conditions hold.

**Positive weights.** Travel times are `length / speed` with speed clamped to
a positive range, so weights are strictly positive. The brief warns about
negative weights breaking Dijkstra; here they are impossible by construction.

**FIFO (non-overtaking).** Leaving later must never mean arriving earlier.
Real traffic is close to FIFO, but an *interpolated* profile can violate it at
a sharp rush-hour edge, which would break optimality. `TimeDependentGraph`
precomputes arrival times on the departure grid and enforces monotonicity with
a running maximum, including a second pass for the midnight wrap. The
correction is small and always conservative — it never invents a faster trip.

The unit test `test_static_baseline_never_beats_time_dependent` asserts the
consequence directly, and `test_matches_networkx_when_weights_are_constant`
checks the implementation against `networkx` on random graphs with a
time-invariant profile.

### Performance

Edge travel-time profiles are precomputed once per `(weather, holiday)`
scenario as a dense `n_edges x n_bins` array. Test cases are grouped by
scenario, so the expensive model lookups happen once per scenario rather than
once per Dijkstra relaxation. Queries run in well under a millisecond on the
reference network.

---

## 5. Departure-time planning

The problem statement ends on this question: *given today's conditions, when
should I leave?* It is answered by `departure.py`, and it is where the system
delivers the most value.

`departure_curve` sweeps candidate departure minutes across a window, running
the full time-dependent router at each, and returns the travel time for every
one. `latest_departure_for_arrival` solves the inverse problem — *the meeting
is at 09:00, when must I leave?* — by scanning backwards from the deadline.

On the reference data, choosing the best minute in a two-hour window saves a
mean of **8.5%** of travel time (14.2% at the 90th percentile). Re-routing
saves ~1.4% on the trips where it changes the path. **Choosing when to leave
is worth several times more than choosing where to drive** — and only a
time-dependent model can answer it at all.

---

## 6. Results on the reference data

Generated by `scripts/train_speed_model.py` and `scripts/evaluate.py`.

### Speed prediction (held-out days)

| Model | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| Hierarchical profile model | **6.46 km/h** | 8.46 km/h | 19.9% |
| Per-edge median (baseline) | 7.49 km/h | 9.16 km/h | 23.1% |

13.8% better than the naive baseline. The residual error is dominated by the
irreducible noise in the aggregated speeds themselves.

### ETA and routing (300 test cases)

| Metric | Time-dependent | Static baseline |
| --- | --- | --- |
| ETA MAE | 128.1 s | 130.0 s |
| ETA MAPE | 5.56% | 5.72% |
| Mean true travel time | 1,789.0 s | 1,792.7 s |

Routes differ on 15.3% of cases; on those the time-dependent route is 24.4 s
faster on average.

### Departure planning (2-hour window)

| Metric | Value |
| --- | --- |
| Mean saving, best vs. worst departure | **8.5%** |
| 90th percentile saving | 14.2% |

### Reading these numbers honestly

The routing gain is real but modest, and that is a property of the test city,
not a defect in the algorithm. On a uniform grid most detours are close to
equivalent, so there is little for a smarter router to exploit. This was
verified with an **oracle experiment**: substituting the true generating
function for the fitted model raises the routing gain only to ~0.6% — an upper
bound on what *any* speed model could achieve on this network. The gap between
the fitted model (0.2%) and the oracle (0.6%) is the part attributable to
prediction error; the rest is the network's own lack of route diversity.

On a real city with genuine alternatives — ring roads, river crossings,
highway-versus-surface choices — the same algorithm has far more to work with.
The departure-time result, by contrast, is large even on a uniform grid,
because it exploits temporal rather than spatial structure.

---

## 7. What would be tried next

* **Turn penalties and traffic signals.** Intersection delay is a real share
  of urban travel time and is entirely absent from a pure edge model.
* **A\* with a landmark heuristic.** The free-flow travel time per edge is
  already computed and is an admissible lower bound; ALT preprocessing would
  cut expanded nodes substantially on a city-scale graph.
* **Quantile profiles.** Predicting the 80th percentile as well as the median
  turns a point ETA into "12–17 minutes", which is what a user actually needs.
* **Incident detection.** The current model is purely historical. Detecting a
  live deviation from the profile is what separates a planner from a live
  navigator.
