#!/usr/bin/env python3
"""
fix_crs_from_report_csv.py

Fix CRS metadata in GeoPackages listed in a CSV report produced by the checker.

This updates CRS *metadata only* (no reprojection) to match the CRS in --sample.

Targets (for each gpkg in CSV):
  - gpkg_contents.srs_id for data_type in ('features','tiles')
  - gpkg_geometry_columns.srs_id (if table exists)
  - gpkg_tile_matrix_set.srs_id (if table exists)
Also ensures gpkg_spatial_ref_sys contains the expected CRS row (copied from sample).

Default: dry-run. Use --apply to modify files.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class SRSRow:
    srs_id: int
    srs_name: str
    organization: str
    organization_coordsys_id: Optional[int]
    definition: str
    description: str

    @property
    def label(self) -> str:
        if self.organization and self.organization_coordsys_id is not None:
            return f"{self.organization}:{self.organization_coordsys_id}"
        return f"SRS_ID:{self.srs_id}"


def connect_sqlite(path: Path, rw: bool) -> sqlite3.Connection:
    if rw:
        return sqlite3.connect(str(path))
    uri = f"file:{path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def load_sample_srs(sample_gpkg: Path) -> SRSRow:
    """
    Determines the sample CRS by looking for a non-null/non-0 srs_id in:
      1) gpkg_geometry_columns
      2) gpkg_tile_matrix_set
      3) gpkg_contents (features/tiles)
    Then returns the corresponding gpkg_spatial_ref_sys row.
    """
    with connect_sqlite(sample_gpkg, rw=False) as conn:
        srs_id: Optional[int] = None

        if table_exists(conn, "gpkg_geometry_columns"):
            r = conn.execute(
                "SELECT DISTINCT srs_id FROM gpkg_geometry_columns "
                "WHERE srs_id IS NOT NULL AND srs_id != 0 LIMIT 1"
            ).fetchone()
            if r:
                srs_id = int(r[0])

        if srs_id is None and table_exists(conn, "gpkg_tile_matrix_set"):
            r = conn.execute(
                "SELECT DISTINCT srs_id FROM gpkg_tile_matrix_set "
                "WHERE srs_id IS NOT NULL AND srs_id != 0 LIMIT 1"
            ).fetchone()
            if r:
                srs_id = int(r[0])

        if srs_id is None:
            r = conn.execute(
                "SELECT DISTINCT srs_id FROM gpkg_contents "
                "WHERE data_type IN ('features','tiles') AND srs_id IS NOT NULL AND srs_id != 0 LIMIT 1"
            ).fetchone()
            if r:
                srs_id = int(r[0])

        if srs_id is None:
            raise SystemExit(f"Could not infer sample CRS from {sample_gpkg}")

        row = conn.execute(
            "SELECT srs_id, srs_name, organization, organization_coordsys_id, definition, description "
            "FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
            (srs_id,),
        ).fetchone()
        if not row:
            raise SystemExit(f"Sample gpkg_spatial_ref_sys missing srs_id={srs_id}")

        return SRSRow(
            srs_id=int(row[0]),
            srs_name=row[1],
            organization=row[2],
            organization_coordsys_id=(int(row[3]) if row[3] is not None else None),
            definition=row[4],
            description=row[5],
        )


def ensure_srs_row(conn: sqlite3.Connection, srs: SRSRow, apply: bool) -> bool:
    exists = conn.execute(
        "SELECT 1 FROM gpkg_spatial_ref_sys WHERE srs_id = ? LIMIT 1",
        (srs.srs_id,),
    ).fetchone()
    if exists:
        return False
    if not apply:
        return True
    conn.execute(
        "INSERT INTO gpkg_spatial_ref_sys (srs_id, srs_name, organization, organization_coordsys_id, definition, description) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (srs.srs_id, srs.srs_name, srs.organization, srs.organization_coordsys_id, srs.definition, srs.description),
    )
    return True


def update_crs_metadata(conn: sqlite3.Connection, srs_id: int, mode: str, apply: bool) -> Tuple[int, int, int]:
    """
    Returns (n_contents, n_geom, n_tiles): number of rows updated (or that would be updated in dry-run).
    mode:
      - fill: only set where NULL or 0
      - force: overwrite any value != srs_id
    """
    def needs(cur: Optional[int]) -> bool:
        if mode == "fill":
            return cur is None or int(cur) == 0
        # force
        return cur is None or int(cur) != srs_id

    n_contents = n_geom = n_tiles = 0

    # gpkg_contents
    rows = conn.execute(
        "SELECT table_name, srs_id FROM gpkg_contents WHERE data_type IN ('features','tiles')"
    ).fetchall()
    for table_name, cur in rows:
        if needs(cur):
            n_contents += 1
            if apply:
                conn.execute(
                    "UPDATE gpkg_contents SET srs_id=? WHERE table_name=?",
                    (srs_id, table_name),
                )

    # gpkg_geometry_columns
    if table_exists(conn, "gpkg_geometry_columns"):
        rows = conn.execute("SELECT table_name, srs_id FROM gpkg_geometry_columns").fetchall()
        for table_name, cur in rows:
            if needs(cur):
                n_geom += 1
                if apply:
                    conn.execute(
                        "UPDATE gpkg_geometry_columns SET srs_id=? WHERE table_name=?",
                        (srs_id, table_name),
                    )

    # gpkg_tile_matrix_set
    if table_exists(conn, "gpkg_tile_matrix_set"):
        rows = conn.execute("SELECT table_name, srs_id FROM gpkg_tile_matrix_set").fetchall()
        for table_name, cur in rows:
            if needs(cur):
                n_tiles += 1
                if apply:
                    conn.execute(
                        "UPDATE gpkg_tile_matrix_set SET srs_id=? WHERE table_name=?",
                        (srs_id, table_name),
                    )

    return n_contents, n_geom, n_tiles


def read_bad_gpkgs_from_report(report_csv: Path) -> List[Path]:
    """
    Expects a column named 'gpkg_path'. Uses 'ok' if present; otherwise takes all rows.
    """
    out: List[Path] = []
    with report_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return out

        has_ok = "ok" in r.fieldnames
        if "gpkg_path" not in r.fieldnames:
            raise SystemExit(f"CSV missing required column 'gpkg_path'. Found: {r.fieldnames}")

        for row in r:
            gpkg = Path(row["gpkg_path"])
            if has_ok:
                ok_val = str(row.get("ok", "")).strip().lower()
                is_ok = ok_val in {"true", "1", "yes"}
                if is_ok:
                    continue
            out.append(gpkg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fix CRS metadata using a checker report CSV (no rescanning).")
    ap.add_argument("--report", required=True, help="CSV report produced by the checker (must contain gpkg_path)")
    ap.add_argument("--sample", required=True, help="Sample *_TCN.gpkg used to derive expected CRS")
    ap.add_argument("--mode", choices=["fill", "force"], default="fill",
                    help="fill: fix only NULL/0; force: overwrite mismatched CRS metadata")
    ap.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    ap.add_argument("--list", action="store_false", help="If only changed gpks list required (default: all details)")
    ap.add_argument("--backup", action="store_true", help="If --apply, create <file>.gpkg.bak first (once)")
    ap.add_argument("--log", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    report = Path(args.report).expanduser().resolve()
    sample = Path(args.sample).expanduser().resolve()
    if not report.exists():
        raise SystemExit(f"--report not found: {report}")
    if not sample.exists():
        raise SystemExit(f"--sample not found: {sample}")

    srs = load_sample_srs(sample)
    logging.info("Sample CRS: %s (srs_id=%d)", srs.label, srs.srs_id)

    targets = read_bad_gpkgs_from_report(report)
    logging.info("Targets from report (non-ok rows): %d", len(targets))
    if not targets:
        logging.info("Nothing to do.")
        return

    changed_files = 0
    skipped_missing = 0

    for gpkg in targets:
        if not gpkg.exists():
            skipped_missing += 1
            logging.warning("SKIP missing file: %s", gpkg)
            continue

        if args.apply and args.backup:
            bak = gpkg.with_suffix(gpkg.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(gpkg, bak)

        with connect_sqlite(gpkg, rw=args.apply) as conn:
            if args.apply:
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("BEGIN IMMEDIATE")

            srs_added = ensure_srs_row(conn, srs, apply=args.apply)
            n_c, n_g, n_t = update_crs_metadata(conn, srs.srs_id, mode=args.mode, apply=args.apply)

            if args.apply:
                conn.execute("COMMIT")

        would_change = srs_added or (n_c + n_g + n_t) > 0
        if would_change:
            tag = "UPDATED" if args.apply else "WOULD UPDATE"
            if args.list:
                logging.warning(gpkg)
            else:
                logging.warning("%s %s | add_srs=%s | gpkg_contents=%d gpkg_geometry_columns=%d gpkg_tile_matrix_set=%d",
                            tag, gpkg, srs_added, n_c, n_g, n_t)
            if args.apply:
                changed_files += 1

    if args.apply:
        logging.info("Done. Updated %d files. Missing skipped: %d", changed_files, skipped_missing)
    else:
        logging.info("Done (dry-run). Re-run with --apply to write changes. Missing skipped: %d", skipped_missing)


if __name__ == "__main__":
    main()

"""
python fix_gpkg_crs.py --report /scratch2/projects/PDG_shared/AK_TCN_gpkgs/tcn_layer_crs_report.csv --sample /scratch2/projects/PDG_shared/AK_TCN_gpkgs/43_17/ArcticMosaic_43_17_1_1_TCN.gpkg --mode force --log INFO
"""