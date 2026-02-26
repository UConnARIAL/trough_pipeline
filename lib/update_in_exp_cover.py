# -*- coding: utf-8 -*-
"""
Date-weighted (Gaussian) exposure scoring for TCN products.

This module computes a per-cutline "date quality" weight based on acquisition date
relative to a target peak date (mean_month/mean_day), using a Gaussian decay
controlled by sigma_days.

It then propagates that date weight into graph layers (nodes/edges/components)
as an "exposure" value based on spatial overlap with cutline polygons:

- Cutlines:
  - norm_input_exp      : Gaussian weight in [0, 1]
  - days_from_max_exp   : |Δdays| from peak date

- Nodes:
  - norm_input_exp      : mean of cutline weights for polygons that contain the node
  - input_exp_cnt       : count of contributing cutline polygons

- Edges / Components:
  - norm_input_exp      : fraction-weighted mean of cutline weights
                          where fraction = (overlap length) / (feature length)
  - input_exp_cnt       : count of contributing cutline overlaps (segments)

Notes:
- Uses spatial index (GeoPandas .sindex) for candidate cutline filtering.
- Uses OGR for in-place attribute updates in the GeoPackage.
- Does not modify geometries, only adds/updates attributes.

Project: Permafrost Discovery Gateway: Mapping and Analysing Trough Capilary Networks
PI      : Chandi Witharana
Authors : Michael Pimenta, Amal Perera
"""

from __future__ import annotations

import logging
import math
import calendar
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional, Dict

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry

from osgeo import ogr, gdal


# -------------------- helpers: dates & weights --------------------

def _normalize_mean_for_year(year, mean, on_nonexistent="floor"):
    """
    Normalize a "peak date" (mean) to a specific year.

    Args:
        year: Target year.
        mean: Either (month, day) tuple, datetime, or date.
        on_nonexistent: For invalid days (e.g., Feb 29 on non-leap year),
                        either "floor" (clamp to last day of month) or raise.

    Returns:
        A date() anchored to the requested year.
    """
    if isinstance(mean, tuple):
        m, dday = int(mean[0]), int(mean[1])
        # clamp to last valid day for month/year (handles Feb-29)
        last = (date(year + (1 if m == 12 else 0), (m % 12) + 1, 1) - datetime.resolution).day
        if dday > last:
            if on_nonexistent == "floor":
                dday = last
            else:
                raise ValueError(f"Mean {(m,dday)} invalid in year {year}")
        return date(year, m, dday)
    else:
        mu0 = mean.date() if isinstance(mean, datetime) else mean
        try:
            return mu0.replace(year=year)
        except ValueError:
            # e.g., Feb-29 -> Feb-28
            return date(year, 2, 28)


def gaussian_date_quality(acq_date, mean=(7, 15), sd_days=30.0, circular=False):
    """
    Compute Gaussian date quality score relative to a peak date.

    Args:
        acq_date: Acquisition date (date or datetime).
        mean: Peak date as (month, day) tuple or date/datetime.
        sd_days: Gaussian sigma in days (must be > 0).
        circular: If True, compute distance circularly over day-of-year
                  (useful when peak is near New Year).

    Returns:
        Float in (0, 1], with 1.0 at peak date and decaying with |Δdays|.
    """
    if isinstance(acq_date, datetime):
        acq_date = acq_date.date()
    if sd_days <= 0:
        raise ValueError("sd_days must be > 0")

    mu = _normalize_mean_for_year(acq_date.year, mean)

    if circular:
        doy = acq_date.timetuple().tm_yday
        mu_doy = mu.timetuple().tm_yday
        year_days = 366 if calendar.isleap(acq_date.year) else 365
        delta = abs(doy - mu_doy)
        d = min(delta, year_days - delta)
    else:
        d = abs((acq_date - mu).days)

    return math.exp(-0.5 * (d / float(sd_days))**2)


def _delta_days_rowwise(acq_dates, mean, circular=False):
    """
    Return |Δdays| per row as float, robust to mixed date-like inputs.

    Args:
        acq_dates: Iterable/Series of str/date/datetime/Timestamp/datetime64.
        mean: Peak date as (month, day) tuple or date/datetime.
        circular: If True, compute circular day-of-year distance.

    Returns:
        np.ndarray of floats (same length as input).
        Invalid/unparseable dates yield 0.0 (consistent with weight=0 fallback).
    """
    # coerce to pandas datetime
    s = pd.to_datetime(pd.Series(acq_dates), errors="coerce")
    mask = s.notna()
    out = np.zeros(len(s), dtype=float)
    if not mask.any():
        return out

    # per-row Python dates
    dates = s.dt.date.to_numpy()  # dtype=object (python date or None)
    years = np.array([d.year for d in dates[mask]], dtype=int)

    # per-row mean date anchored to each year
    mu_list = [_normalize_mean_for_year(y, mean) for y in years]

    if circular:
        doy = np.array([d.timetuple().tm_yday for d in dates[mask]], dtype=int)
        mu_doy = np.array([m.timetuple().tm_yday for m in mu_list], dtype=int)
        days_in_year = np.array([366 if calendar.isleap(y) else 365 for y in years], dtype=int)
        delta = np.abs(doy - mu_doy)
        d = np.minimum(delta, days_in_year - delta).astype(float)
    else:
        d = np.array([abs((dates[mask][k] - mu_list[k]).days) for k in range(len(mu_list))], dtype=float)

    out[mask.to_numpy()] = d
    return out


def _gaussian_from_delta(delta_days: np.ndarray, sigma_days: float) -> np.ndarray:
    """Vectorized Gaussian weight from |Δdays|."""
    return np.exp(-0.5 * (delta_days / float(sigma_days))**2)


# -------------------- helpers: geometry & updates --------------------

def _has_geom(geom: BaseGeometry) -> bool:
    """True if geometry is non-null and non-empty."""
    return (geom is not None) and (not getattr(geom, "is_empty", False))


def _ensure_field(layer: ogr.Layer, name: str, ogr_type: int) -> None:
    """
    Ensure attribute field `name` exists on OGR `layer`. Create it if missing.

    Works for GPKG with OGR Python bindings (GDAL 2/3/4).
    """
    defn = layer.GetLayerDefn()

    existing = set()
    n = defn.GetFieldCount()
    for i in range(n):
        fdefn = defn.GetFieldDefn(i)
        if fdefn is not None:
            existing.add(fdefn.GetName())

    if name in existing:
        return

    fdefn_new = ogr.FieldDefn(name, ogr_type)
    if ogr_type == ogr.OFTReal:
        # optional: set width/precision for nicer storage
        fdefn_new.SetWidth(24)
        fdefn_new.SetPrecision(8)

    rc = layer.CreateField(fdefn_new)
    if rc != 0:
        raise RuntimeError(f"Failed to create field '{name}' on layer '{layer.GetName()}'")

    try:
        layer.SyncToDisk()
    except Exception:
        pass


def _update_attrs_from_df(
    gpkg_path: str,
    layer_name: str,
    key_cols: Iterable[str],
    df_values: pd.DataFrame,
    write_cols: Iterable[str],
) -> None:
    """
    In-place attribute updates by (key_cols) -> write_cols using OGR; geometry untouched.

    Args:
        gpkg_path: GeoPackage path.
        layer_name: Target layer name.
        key_cols: Columns that uniquely identify features (must exist in both layer and df_values).
        df_values: DataFrame containing keys + columns to write.
        write_cols: Columns to write into the layer.
    """
    ds = ogr.Open(gpkg_path, update=1)
    if ds is None:
        raise RuntimeError(f"Cannot open for update: {gpkg_path}")
    try:
        lyr = ds.GetLayerByName(layer_name)
        if lyr is None:
            raise RuntimeError(f"Layer not found: {layer_name}")

        # Ensure destination fields exist
        for c in write_cols:
            _ensure_field(
                lyr,
                c,
                ogr.OFTReal if c.endswith("_exp") or c.endswith("_norm") else ogr.OFTInteger,
            )

        key_cols = list(key_cols)
        write_cols = list(write_cols)

        for k in key_cols:
            if k not in df_values.columns:
                raise KeyError(f"Missing key column '{k}' in values DataFrame for layer '{layer_name}'")

        # Build lookup: (key tuple) -> (values tuple)
        m: Dict[tuple, tuple] = {}
        for _, row in df_values.iterrows():
            key = tuple(row[k] for k in key_cols)
            m[key] = tuple(row[c] for c in write_cols)

        lyr.ResetReading()
        feat = lyr.GetNextFeature()
        updated = 0
        while feat:
            key = tuple(feat.GetField(k) for k in key_cols)
            vals = m.get(key)
            if vals is not None:
                for c, v in zip(write_cols, vals):
                    feat.SetField(c, float(v) if v is not None else None)
                if lyr.SetFeature(feat) != 0:
                    raise RuntimeError(f"Failed to write feature FID={feat.GetFID()} in '{layer_name}'")
                updated += 1
            feat = lyr.GetNextFeature()

        logging.info(f"[{layer_name}] updated {updated} features")
    finally:
        ds = None


# -------------------- main: precompute + exposures --------------------

def add_date_exposure_gaussian(
    gpkg_path: str,
    cutline_layer: str,
    date_field: str,
    mean_month: int,
    mean_day: int,
    sigma_days: float,
    nodes_layer: Optional[str] = None,
    edges_layer: Optional[str] = None,
    comps_layer: Optional[str] = None,
    edges_length_field: str = "length_m",
    comps_length_field: str = "total_length_m",
    target_crs: str = "EPSG:3338",
    circular: bool = False,
) -> None:
    """
    Precompute Gaussian date weight on cutlines, then compute per-feature exposures.

    - Cutlines: store per-feature "norm_input_exp" and "days_from_max_exp".
    - Nodes: mean(Dj) over containing cutlines (if multiple), else 0.
    - Edges/Components: fraction-weighted mean: sum(Fj*Dj)/sum(Fj),
      where Fj is overlap length / feature length.

    Also stores a count field "input_exp_cnt" on each graph layer.

    Args:
        gpkg_path: Target GeoPackage path.
        cutline_layer: Cutline polygon layer name.
        date_field: Date field name on cutlines (parsed by pandas.to_datetime).
        mean_month/mean_day: Peak date definition for Gaussian weighting.
        sigma_days: Gaussian sigma in days (controls decay from peak).
        nodes_layer/edges_layer/comps_layer: Graph layer names (optional).
        edges_length_field/comps_length_field: Not used in current logic (kept for compatibility).
        target_crs: CRS for length computations and consistent overlay.
        circular: If True, day-of-year circular distance for weighting.
    """
    gdal.UseExceptions()
    ogr.UseExceptions()

    # 1) Load cutlines and compute per-feature weights (store on the same layer)
    cut_raw = gpd.read_file(gpkg_path, layer=cutline_layer)
    if date_field not in cut_raw.columns:
        raise KeyError(f"'{date_field}' not found in layer '{cutline_layer}'")

    acq = pd.to_datetime(cut_raw[date_field], errors="coerce")
    if acq.isna().any():
        # keep NaNs => weight 0 and delta 0
        pass

    mean_tuple = (int(mean_month), int(mean_day))
    delta = _delta_days_rowwise(acq, mean_tuple, circular=circular)
    w = _gaussian_from_delta(delta, sigma_days)

    # write per-cutline attributes (recomputed per feature for safety)
    def _update_cutline_attrs():
        ds = ogr.Open(gpkg_path, update=1)
        if ds is None:
            raise RuntimeError(f"Cannot open for update: {gpkg_path}")
        try:
            lyr = ds.GetLayerByName(cutline_layer)
            if lyr is None:
                raise RuntimeError(f"Layer not found: {cutline_layer}")

            _ensure_field(lyr, "norm_input_exp", ogr.OFTReal)
            _ensure_field(lyr, "days_from_max_exp", ogr.OFTInteger)

            lyr.ResetReading()
            feat = lyr.GetNextFeature()
            written = 0
            while feat:
                val = feat.GetField(date_field)
                try:
                    ts = pd.to_datetime(val).date()
                    d = _delta_days_rowwise(pd.Series([ts]), mean_tuple, circular=circular)[0]
                    ww = _gaussian_from_delta(np.array([d]), sigma_days)[0]
                except Exception:
                    d = 0.0
                    ww = 0.0

                feat.SetField("norm_input_exp", float(ww))
                feat.SetField("days_from_max_exp", int(round(d)))
                if lyr.SetFeature(feat) != 0:
                    raise RuntimeError(f"Failed to update cutline feature FID={feat.GetFID()}")
                written += 1
                feat = lyr.GetNextFeature()

            logging.info(f"[{cutline_layer}] updated {written} features (norm_input_exp, days_from_max_exp)")
        finally:
            ds = None

    _update_cutline_attrs()

    # 2) Dissolve cutlines BY DATE (overlap-robust overlay math)
    #    prevents double counting when same-date polygons overlap
    cut = []

    mask_valid = (~acq.isna()) & cut_raw.geometry.notna() & (~cut_raw.geometry.is_empty)
    cut_work = cut_raw.loc[mask_valid].copy()
    if cut_work.empty:
        logging.error("[WARN] no valid cutlines to compute exposures against; skipping feature exposures.")
        return

    cut_work["__delta"] = _delta_days_rowwise(pd.to_datetime(cut_work[date_field]).dt.date, mean_tuple, circular=circular)
    cut_work["__Dw"] = _gaussian_from_delta(cut_work["__delta"].to_numpy(), sigma_days)

    # Reproject to target CRS for correct length computations
    if target_crs and (str(cut_work.crs) != str(target_crs)):
        cut_work = cut_work.to_crs(target_crs)

    for dval, sub in cut_work.groupby(date_field):
        geom = unary_union([g for g in sub.geometry if _has_geom(g)])
        if geom and not geom.is_empty:
            cut.append({
                "ACQDATE": pd.to_datetime(dval).date(),
                "geometry": geom,
                "date_w": float(sub["__Dw"].iloc[0]),
                "delta_days": float(sub["__delta"].iloc[0]),
            })

    cut = gpd.GeoDataFrame(cut, geometry="geometry", crs=cut_work.crs).reset_index(drop=True)
    if cut.empty:
        logging.error("[WARN] dissolved cutlines empty; skipping feature exposures.")
        return

    cut_geoms = list(cut.geometry.values)
    weights = cut["date_w"].to_numpy(dtype=float)
    deltas = cut["delta_days"].to_numpy(dtype=float)
    sidx = cut.sindex if len(cut) > 0 else None

    # ---------- Nodes ----------
    if nodes_layer:
        nodes = gpd.read_file(gpkg_path, layer=nodes_layer)
        if target_crs and (str(nodes.crs) != str(target_crs)):
            nodes = nodes.to_crs(target_crs)

        exp_nodes = np.zeros(len(nodes), dtype=float)
        cnt_nodes = np.zeros(len(nodes), dtype=np.int32)

        for i, pt in enumerate(nodes.geometry):
            if not _has_geom(pt):
                continue
            cand = list(sidx.intersection(pt.bounds)) if sidx is not None else list(range(len(cut)))
            if not cand:
                continue
            inside = [j for j in cand if pt.within(cut_geoms[j])]
            if not inside:
                continue

            cnt_nodes[i] = len(inside)
            Dj = weights[inside]
            exp_nodes[i] = float(Dj.mean())

        df_nodes = pd.DataFrame({
            "tile_id": nodes.get("tile_id", pd.Series([None] * len(nodes))).values,
            "file_id": nodes.get("file_id", pd.Series([None] * len(nodes))).values,
            "node_id": nodes["node_id"].values,
            "norm_input_exp": exp_nodes,
            "input_exp_cnt": cnt_nodes.astype(int),
        })
        _update_attrs_from_df(
            gpkg_path, nodes_layer,
            key_cols=["tile_id", "file_id", "node_id"],
            df_values=df_nodes,
            write_cols=["norm_input_exp", "input_exp_cnt"],
        )

    # ---------- Edges ----------
    if edges_layer:
        edges = gpd.read_file(gpkg_path, layer=edges_layer)
        if target_crs and (str(edges.crs) != str(target_crs)):
            edges = edges.to_crs(target_crs)

        denom_e = edges.geometry.length.to_numpy()
        denom_e = np.where(denom_e > 0, denom_e, np.nan)

        exp_edges = np.zeros(len(edges), dtype=float)
        cnt_edges = np.zeros(len(edges), dtype=np.int32)

        for i, geom in enumerate(edges.geometry):
            if not _has_geom(geom) or np.isnan(denom_e[i]):
                continue
            cand = list(sidx.intersection(geom.bounds)) if sidx is not None else list(range(len(cut)))
            if not cand:
                continue

            Ls, Dj = [], []
            for j in cand:
                if not geom.intersects(cut_geoms[j]):
                    continue
                L = geom.intersection(cut_geoms[j]).length
                if L <= 0:
                    continue
                Ls.append(L)
                Dj.append(weights[j])

            if not Ls:
                continue

            cnt_edges[i] = len(Ls)
            Ls = np.asarray(Ls, dtype=float)
            fracs = Ls / denom_e[i]
            sumF = float(fracs.sum())
            if sumF <= 0:
                continue

            Dj = np.asarray(Dj, dtype=float)
            exp_edges[i] = float((fracs * Dj).sum() / sumF)

        df_edges = pd.DataFrame({
            "tile_id": edges.get("tile_id", pd.Series([None] * len(edges))).values,
            "file_id": edges.get("file_id", pd.Series([None] * len(edges))).values,
            "edge_id": edges["edge_id"].values,
            "norm_input_exp": np.clip(exp_edges, 0.0, 1.0),
            "input_exp_cnt": cnt_edges.astype(int),
        })
        _update_attrs_from_df(
            gpkg_path, edges_layer,
            key_cols=["tile_id", "file_id", "edge_id"],
            df_values=df_edges,
            write_cols=["norm_input_exp", "input_exp_cnt"],
        )

    # ---------- Components ----------
    if comps_layer:
        comps = gpd.read_file(gpkg_path, layer=comps_layer)
        if target_crs and (str(comps.crs) != str(target_crs)):
            comps = comps.to_crs(target_crs)

        denom_c = comps.geometry.length.to_numpy()
        denom_c = np.where(denom_c > 0, denom_c, np.nan)

        exp_comps = np.zeros(len(comps), dtype=float)
        cnt_comps = np.zeros(len(comps), dtype=np.int32)

        for i, geom in enumerate(comps.geometry):
            if not _has_geom(geom) or np.isnan(denom_c[i]):
                continue
            cand = list(sidx.intersection(geom.bounds)) if sidx is not None else list(range(len(cut)))
            if not cand:
                continue

            Ls, Dj = [], []
            for j in cand:
                if not geom.intersects(cut_geoms[j]):
                    continue
                L = geom.intersection(cut_geoms[j]).length
                if L <= 0:
                    continue
                Ls.append(L)
                Dj.append(weights[j])

            if not Ls:
                continue

            cnt_comps[i] = len(Ls)
            Ls = np.asarray(Ls, dtype=float)
            fracs = Ls / denom_c[i]
            sumF = float(fracs.sum())
            if sumF <= 0:
                continue

            Dj = np.asarray(Dj, dtype=float)
            exp_comps[i] = float((fracs * Dj).sum() / sumF)

        df_comps = pd.DataFrame({
            "tile_id": comps.get("tile_id", pd.Series([None] * len(comps))).values,
            "file_id": comps.get("file_id", pd.Series([None] * len(comps))).values,
            "component_id": comps["component_id"].values,
            "norm_input_exp": np.clip(exp_comps, 0.0, 1.0),
            "input_exp_cnt": cnt_comps.astype(int),
        })
        _update_attrs_from_df(
            gpkg_path, comps_layer,
            key_cols=["tile_id", "file_id", "component_id"],
            df_values=df_comps,
            write_cols=["norm_input_exp", "input_exp_cnt"],
        )

        logging.info(f"✅ Added: norm_exp_coverage to nodes,edges,components to {gpkg_path}")


# -------------------- simple test main --------------------

def main():
    """
    Minimal sanity test:
    - Prints a few Gaussian quality values.
    - Runs exposure computation on a local test GPKG path.
    """
    print(gaussian_date_quality(date(2025, 7, 15), mean=(7, 15), sd_days=25))   # 1.0
    print(gaussian_date_quality(date(2025, 6, 15), mean=(7, 15), sd_days=25))   # ~0.486
    print(gaussian_date_quality(date(2025, 9, 15), mean=(7, 15), sd_days=25))   # ~0.135
    print(gaussian_date_quality(date(2025, 12, 31), mean=(1, 10), sd_days=15, circular=True))

    out_gpkg = "./test_tile_dir/58_16/ArcticMosaic_58_16_1_1_mask.gpkg"
    logging.info(f"exp compute path {out_gpkg}")

    add_date_exposure_gaussian(
        gpkg_path=out_gpkg,
        cutline_layer="InputMosaicCutlinesVector",
        date_field="ACQDATE",
        mean_month=7,
        mean_day=15,
        sigma_days=45,
        nodes_layer="GraphTheoraticNodes",
        edges_layer="GraphTheoraticEdges",
        comps_layer="GraphTheoraticComponents",
        edges_length_field="length_m",
        comps_length_field="total_length_m",
        target_crs="EPSG:3338",
    )

if __name__ == "__main__":
    main()

