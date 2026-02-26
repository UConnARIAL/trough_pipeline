#!/usr/bin/env python3
"""
check_tile_tcn_gpkg_layers_crs.py

helper util to check crs is valid

Checks GeoPackages under:
  <root>/<tile_dir>/<pattern>   (default pattern: *_TCN.gpkg)

Avoids summary gpkg by excluding filenames containing 'summary' (case-insensitive).

Validates:
  1) expected layers/tables exist (inferred from --sample)
  2) spatial CRS is consistent within each gpkg (features+tiles)
  3) spatial CRS matches the sample gpkg's spatial CRS
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

SPATIAL_TYPES = {"features", "tiles"}


@dataclass
class SRSInfo:
    srs_id: int
    organization: Optional[str]
    organization_coordsys_id: Optional[int]
    definition: Optional[str]

    @property
    def label(self) -> str:
        if self.organization and self.organization_coordsys_id is not None:
            return f"{self.organization}:{self.organization_coordsys_id}"
        return f"SRS_ID:{self.srs_id}"


@dataclass
class CheckResult:
    gpkg_path: str
    tile_dir: str
    ok: bool

    expected_layers: str
    present_layers: str
    missing_layers: str
    extra_layers: str

    expected_spatial_crs: str
    gpkg_spatial_crs: str
    spatial_srs_ids_found: str
    spatial_crs_consistent: bool
    spatial_crs_matches_sample: bool

    error: str = ""


def parse_types(s: str) -> Set[str]:
    return {p.strip() for p in s.split(",") if p.strip()}


def connect_sqlite(gpkg_path: Path) -> sqlite3.Connection:
    uri = f"file:{gpkg_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def fetch_contents(conn: sqlite3.Connection, include_types: Set[str]) -> List[Tuple[str, str, Optional[int]]]:
    q = "SELECT table_name, data_type, srs_id FROM gpkg_contents"
    rows = conn.execute(q).fetchall()
    return [(t, dt, srs) for (t, dt, srs) in rows if dt in include_types]


def fetch_srs_info(conn: sqlite3.Connection, srs_id: int) -> Optional[SRSInfo]:
    q = """
    SELECT srs_id, organization, organization_coordsys_id, definition
    FROM gpkg_spatial_ref_sys
    WHERE srs_id = ?
    """
    row = conn.execute(q, (srs_id,)).fetchone()
    if not row:
        return None
    return SRSInfo(
        srs_id=int(row[0]),
        organization=row[1],
        organization_coordsys_id=(int(row[2]) if row[2] is not None else None),
        definition=row[3],
    )


def describe_srs(conn: sqlite3.Connection, srs_id: Optional[int]) -> str:
    if srs_id is None:
        return ""
    info = fetch_srs_info(conn, int(srs_id))
    return info.label if info else f"SRS_ID:{srs_id}"


def expected_from_sample(sample_gpkg: Path, include_types: Set[str]) -> Tuple[List[str], Optional[str]]:
    with connect_sqlite(sample_gpkg) as conn:
        contents = fetch_contents(conn, include_types)
        expected_layers = sorted([t for (t, _, _) in contents])

        spatial_srs_ids: Set[int] = set()
        for _, data_type, srs_id in contents:
            if data_type in SPATIAL_TYPES and srs_id is not None:
                spatial_srs_ids.add(int(srs_id))

        if len(spatial_srs_ids) == 1:
            sid = next(iter(spatial_srs_ids))
            expected_spatial_crs_label = describe_srs(conn, sid)
        else:
            expected_spatial_crs_label = None

        return expected_layers, expected_spatial_crs_label


def iter_tile_gpkgs(root: Path, pattern: str, exclude_substrings: List[str], rglob: bool) -> Iterable[Path]:
    """
    Yields gpkg files matching pattern within each immediate tile_dir under root.

    - If rglob=False:  <root>/<tile_dir>/<pattern>
    - If rglob=True:   <root>/<tile_dir>/**/<pattern>

    Excludes filenames containing any exclude_substrings (case-insensitive).
    """
    excludes = [s.lower() for s in exclude_substrings]

    for tile_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        it = tile_dir.rglob(pattern) if rglob else tile_dir.glob(pattern)
        for gpkg in it:
            if not gpkg.is_file():
                continue
            name_l = gpkg.name.lower()
            if any(x in name_l for x in excludes):
                continue
            yield gpkg


def check_one_gpkg(
    gpkg_path: Path,
    expected_layers: List[str],
    expected_spatial_crs_label: Optional[str],
    include_types: Set[str],
    fail_on_extra: bool,
) -> CheckResult:
    tile_dir_name = gpkg_path.parent.name  # works for your root/tile_dir/<file> layout
    try:
        with connect_sqlite(gpkg_path) as conn:
            contents = fetch_contents(conn, include_types)
            present_layers_list = sorted([t for (t, _, _) in contents])

            expected_set = set(expected_layers)
            present_set = set(present_layers_list)

            missing = sorted(expected_set - present_set)
            extra = sorted(present_set - expected_set)

            spatial_srs_ids: Set[int] = set()
            for _, data_type, srs_id in contents:
                if data_type in SPATIAL_TYPES and srs_id is not None:
                    spatial_srs_ids.add(int(srs_id))

            spatial_crs_consistent = (len(spatial_srs_ids) <= 1)

            gpkg_spatial_crs_label = ""
            if len(spatial_srs_ids) == 1:
                sid = next(iter(spatial_srs_ids))
                gpkg_spatial_crs_label = describe_srs(conn, sid)

            spatial_crs_matches_sample = True
            if expected_spatial_crs_label is not None:
                spatial_crs_matches_sample = (gpkg_spatial_crs_label == expected_spatial_crs_label)

            ok = (len(missing) == 0) and spatial_crs_consistent and spatial_crs_matches_sample
            if fail_on_extra and len(extra) > 0:
                ok = False

            return CheckResult(
                gpkg_path=str(gpkg_path),
                tile_dir=tile_dir_name,
                ok=ok,
                expected_layers=";".join(expected_layers),
                present_layers=";".join(present_layers_list),
                missing_layers=";".join(missing),
                extra_layers=";".join(extra),
                expected_spatial_crs=(expected_spatial_crs_label or ""),
                gpkg_spatial_crs=gpkg_spatial_crs_label,
                spatial_srs_ids_found=";".join(str(x) for x in sorted(spatial_srs_ids)),
                spatial_crs_consistent=spatial_crs_consistent,
                spatial_crs_matches_sample=spatial_crs_matches_sample,
                error="",
            )
    except Exception as e:
        return CheckResult(
            gpkg_path=str(gpkg_path),
            tile_dir=tile_dir_name,
            ok=False,
            expected_layers=";".join(expected_layers),
            present_layers="",
            missing_layers="",
            extra_layers="",
            expected_spatial_crs=(expected_spatial_crs_label or ""),
            gpkg_spatial_crs="",
            spatial_srs_ids_found="",
            spatial_crs_consistent=False,
            spatial_crs_matches_sample=False,
            error=str(e),
        )


def write_csv(out_csv: Path, results: List[CheckResult]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else list(asdict(CheckResult(
        gpkg_path="", tile_dir="", ok=False,
        expected_layers="", present_layers="", missing_layers="", extra_layers="",
        expected_spatial_crs="", gpkg_spatial_crs="", spatial_srs_ids_found="",
        spatial_crs_consistent=False, spatial_crs_matches_sample=False, error=""
    )).keys())

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing tile subdirs")
    ap.add_argument("--sample", required=True, help="Sample *_TCN.gpkg used to infer expected layers + CRS")
    ap.add_argument("--pattern", default="*_TCN_summary.gpkg", help="Glob pattern within each tile_dir (default: *_TCN.gpkg)")
    ap.add_argument("--rglob", action="store_true", help="If set, search recursively within each tile_dir")
    ap.add_argument("--exclude", default="summary", help="Comma-separated substrings to exclude (default: summary)")
    ap.add_argument("--types", default="features,attributes,tiles", help="gpkg_contents data_type values to treat as layers")
    ap.add_argument("--fail-on-extra", action="store_true", help="Fail if extra layers beyond sample exist")
    ap.add_argument("--out", default="", help="Output CSV path (default: <root>/tcn_sum_layer_crs_report.csv)")
    ap.add_argument("--log", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    root = Path(args.root).expanduser().resolve()
    sample = Path(args.sample).expanduser().resolve()
    include_types = parse_types(args.types)
    excludes = [s.strip() for s in args.exclude.split(",") if s.strip()]

    if not root.exists():
        raise SystemExit(f"--root does not exist: {root}")
    if not sample.exists():
        raise SystemExit(f"--sample does not exist: {sample}")

    expected_layers, expected_spatial_crs_label = expected_from_sample(sample, include_types)

    gpkgs = list(iter_tile_gpkgs(root, args.pattern, excludes, args.rglob))
    logging.info("Found %d matching gpkg files using pattern '%s' (rglob=%s)", len(gpkgs), args.pattern, args.rglob)
    if len(gpkgs) == 0:
        raise SystemExit(f"No matches under {root} using pattern '{args.pattern}' (try --rglob if nested).")

    out_csv = Path(args.out).expanduser().resolve() if args.out else (root / "tcn_layer_crs_report.csv")

    results: List[CheckResult] = []
    for gpkg in gpkgs:
        r = check_one_gpkg(
            gpkg_path=gpkg,
            expected_layers=expected_layers,
            expected_spatial_crs_label=expected_spatial_crs_label,
            include_types=include_types,
            fail_on_extra=args.fail_on_extra,
        )
        results.append(r)
        if not r.ok:
            logging.warning(
                "FAIL %s | missing=%s | extra=%s | CRS=%s (expected=%s) | err=%s",
                gpkg, r.missing_layers, r.extra_layers, r.gpkg_spatial_crs, r.expected_spatial_crs, r.error
            )

    write_csv(out_csv, results)
    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    logging.info("Done. Checked %d files: OK=%d FAIL=%d", len(results), ok_count, fail_count)
    logging.info("CSV report: %s", out_csv)

    if fail_count > 0:
        raise SystemExit(2)

if __name__ == "__main__":
    main()

"""
python check_gpkgs_crs.py --root /scratch2/projects/PDG_shared/AK_TCN_gpkgs --sample /scratch2/projects/PDG_shared/AK_TCN_gpkgs/46_20/46_20_TCN_summary.gpkg
python check_gpkgs_crs.py --root /scratch2/projects/PDG_shared/AK_TCN_gpkgs --sample /scratch2/projects/PDG_shared/AK_TCN_gpkgs/43_17/ArcticMosaic_43_17_1_1_TCN.gpkg


sqlite3 /scratch2/projects/PDG_shared/AK_TCN_gpkgs/49_19/ArcticMosaic_49_19_5_4_TCN.gpkg "SELECT table_name,data_type,srs_id FROM gpkg_contents ORDER BY data_type,table_name;"

sqlite3 /scratch2/projects/PDG_shared/AK_TCN_gpkgs/49_19/ArcticMosaic_49_19_5_4_TCN.gpkg "SELECT table_name,srs_id FROM gpkg_geometry_columns ORDER BY table_name;"

sqlite3 /scratch2/projects/PDG_shared/AK_TCN_gpkgs/49_19/ArcticMosaic_49_19_5_4_TCN.gpkg "SELECT table_name,srs_id FROM gpkg_tile_matrix_set ORDER BY table_name;"

"""
