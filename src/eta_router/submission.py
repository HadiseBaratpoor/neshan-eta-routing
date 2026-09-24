"""Writing ``test_cases.csv`` back in exactly the format the grader expects.

The brief is strict about this and it is the cheapest way to lose every
point: column names, column count and row count must not change. Only ``eta``
and ``route`` are filled in.

``route`` is written as ``[15,45,78,35]`` - a bracketed, comma-separated list
with no spaces, including both endpoints - and ``eta`` as an integer number of
seconds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUBMISSION_COLUMNS = ("src", "dest", "route_start_t", "is_holiday", "weather", "eta", "route")


def format_route(path: list) -> str:
    """``[15,45,78,35]`` - the exact literal format from the brief."""

    return "[" + ",".join(str(int(node)) for node in path) + "]"


def parse_route(text: str) -> list[int]:
    """Inverse of :func:`format_route`, tolerant of spaces."""

    import re

    value = str(text).strip()
    if not re.fullmatch(r"\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*\]", value):
        raise ValueError(f"malformed route: {text!r}")
    return [int(part.strip()) for part in value[1:-1].split(",")]


def build_submission(
    test_cases: pd.DataFrame,
    etas: list[int],
    routes: list[list[int]],
) -> pd.DataFrame:
    """Attach answers to the original frame without reordering or dropping rows."""

    if len(etas) != len(test_cases) or len(routes) != len(test_cases):
        raise ValueError(
            f"answer count mismatch: {len(test_cases)} test cases, "
            f"{len(etas)} etas, {len(routes)} routes"
        )
    out = test_cases.copy()
    out["eta"] = [int(round(float(e))) for e in etas]
    out["route"] = [format_route(r) for r in routes]
    return out


def write_submission(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write the CSV with a plain header and no index column."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def validate_submission(frame: pd.DataFrame, expected_rows: int, graph=None) -> list[str]:
    """Return a list of problems; an empty list means the file is submittable."""

    problems: list[str] = []

    if len(frame) != expected_rows:
        problems.append(f"row count changed: expected {expected_rows}, found {len(frame)}")

    missing = [c for c in SUBMISSION_COLUMNS if c not in frame.columns]
    if missing:
        problems.append(f"missing column(s): {missing}")

    extra = [c for c in frame.columns if c not in SUBMISSION_COLUMNS]
    if extra:
        problems.append(f"unexpected extra column(s): {extra}")

    if "eta" in frame.columns:
        etas = pd.to_numeric(frame["eta"], errors="coerce")
        if etas.isna().any():
            problems.append(f"{int(etas.isna().sum())} eta values are missing or non-numeric")
        elif (etas < 0).any():
            problems.append(f"{int((etas < 0).sum())} eta values are negative")

    parsed_routes = {}
    if "route" in frame.columns:
        for i, value in enumerate(frame["route"].astype(str)):
            try:
                nodes = parse_route(value)
            except (TypeError, ValueError) as exc:
                problems.append(f"row {i}: malformed route: {exc}")
                continue
            if not nodes:
                problems.append(f"row {i}: route is empty")
                continue
            parsed_routes[i] = nodes

    if {"src", "dest", "route"}.issubset(frame.columns):
        for i, row in frame.iterrows():
            nodes = parsed_routes.get(i)
            if not nodes:
                continue
            src, dest = int(row["src"]), int(row["dest"])
            if nodes[0] != src or nodes[-1] != dest:
                problems.append(
                    f"row {i}: route endpoints {nodes[0]}..{nodes[-1]} "
                    f"do not match src/dest {row['src']}..{row['dest']}"
                )
                continue
            if src == dest:
                if nodes != [src] or float(row["eta"]) != 0:
                    problems.append(f"row {i}: source==destination requires eta 0 and route [{src}]")
            elif float(row["eta"]) <= 0:
                problems.append(f"row {i}: non-trivial route requires a positive eta")
            if graph is not None:
                invalid = [(u, v) for u, v in zip(nodes[:-1], nodes[1:]) if not graph.has_edge(u, v)]
                if invalid:
                    problems.append(f"row {i}: route contains graph edge(s) that do not exist: {invalid[:3]}")

    return problems
