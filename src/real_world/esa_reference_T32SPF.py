#!/usr/bin/env python3
# src/realworld/esa_reference_T32SPF.py
#
# Find REAL ESA-flagged Sentinel-2 L1C scenes over a single tile (default T32SPF),
# read their official quality indicators, and emit a labeled CSV that slots into
# the same defects-style schema used by the rest of the pipeline.
#
# Two modes:
#   --mode metadata  (default)  Reads only the product/granule quality XML from
#                               CDSE via the OData "Nodes" API. No bulk download
#                               (a few KB per scene instead of ~700 MB).
#   --mode full                 Downloads each .SAFE, unzips, parses locally, and
#                               (with --read-masks) counts MSK_QUALIT defect pixels.
#
# Ground-truth label produced per scene:
#   esa_flag        = PASSED / FAILED / UNKNOWN   (from the 5 OLQC quality checks)
#   failed_indicator= which of the 5 checks failed (";"-joined)
#   expected_reject = True when esa_flag == FAILED  -> your pipeline SHOULD reject it
#
# Auth: CDSE registration is free. Provide credentials via env vars
#   CDSE_USERNAME / CDSE_PASSWORD   (or --username/--password).
#
# Usage examples:
#   python -m src.realworld.esa_reference_T32SPF --start 2024-01-01 --max-products 200
#   python -m src.realworld.esa_reference_T32SPF --max-cloud 20        # isolate non-cloud defects
#   python -m src.realworld.esa_reference_T32SPF --mode full --read-masks --max-products 10
#   python -m src.realworld.esa_reference_T32SPF --write-db            # also write SQLite table
#
import os
import sys
import csv
import time
import json
import zipfile
import argparse
import sqlite3
import datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TILE = "T32SPF"
PRODUCT_TYPE = "S2MSI1C"          # Sentinel-2 Level-1C
COLLECTION = "SENTINEL-2"

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
ZIPPER   = "https://zipper.dataspace.copernicus.eu/odata/v1"   # Nodes + download
DOWNLOAD = ZIPPER                                                # kept for compat
TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")

# The 5 official OLQC quality checks (uppercase name -> normalized column)
QUALITY_CHECKS = {
    "SENSOR_QUALITY":      "sensor_quality",
    "GEOMETRIC_QUALITY":   "geometric_quality",
    "GENERAL_QUALITY":     "general_quality",
    "FORMAT_CORRECTNESS":  "format_correctness",
    "RADIOMETRIC_QUALITY": "radiometric_quality",
}

# MSK_QUALIT layer order (PB 04.00+): 1 lost-anc, 2 degraded-anc, 3 lost-MSI,
# 4 degraded-MSI, 5 defective, 6 no-data, 7 crosstalk, 8 saturated.
MASK_LAYERS = {"lost_packet": 3, "defective": 5, "nodata": 6, "saturated": 8}

CSV_COLUMNS = [
    "scene_name", "base_scene", "tile", "platform", "sensing_date",
    "processing_baseline", "product_uuid",
    "defect_family", "defect_type",
    "esa_flag", "failed_indicator", "expected_reject",
    "sensor_quality", "geometric_quality", "general_quality",
    "format_correctness", "radiometric_quality",
    "degraded_anc_pct", "degraded_msi_pct",
    "nodata_pct", "saturated_defective_pct", "cloud_cover_pct",
    "mask_defective_px", "mask_nodata_px",
    "mask_lost_packet_px", "mask_saturated_px",
    "source",
]


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _local(tag):
    """Strip XML namespace: '{ns}Foo' -> 'Foo'."""
    return tag.split("}")[-1]


class TokenManager:
    """
    Keeps a valid CDSE access token at all times.

    CDSE access tokens expire in 600 s (10 min). We refresh proactively at
    540 s using the refresh_token (valid 3600 s) so we never hit a
    mid-request expiry. If the refresh_token itself expires we fall back to
    a full password re-login.

    Usage:
        tm = TokenManager(username, password)
        headers = {"Authorization": f"Bearer {tm.get()}"}
    """

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self._access_token = None
        self._refresh_token = None
        self._issued_at = 0
        self._access_ttl = 600      # seconds (CDSE default)
        self._refresh_before = 60   # renew this many seconds before expiry
        self._login()

    def _login(self):
        data = {
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
        }
        r = requests.post(TOKEN_URL, data=data, timeout=60)
        if r.status_code == 401:
            body = ""
            try:
                body = r.json().get("error_description", "")
            except Exception:
                pass
            raise SystemExit(
                f"CDSE login failed (401). Server said: '{body}'\n"
                "Check username/password at dataspace.copernicus.eu."
            )
        r.raise_for_status()
        self._store(r.json())
        print(f"  [auth] logged in, token valid {self._access_ttl}s")

    def _refresh(self):
        data = {
            "client_id": "cdse-public",
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }
        r = requests.post(TOKEN_URL, data=data, timeout=60)
        if r.status_code in (400, 401):
            print("  [auth] refresh token expired, re-logging in...")
            self._login()
            return
        r.raise_for_status()
        self._store(r.json())
        print("  [auth] token refreshed")

    def _store(self, payload):
        self._access_token  = payload["access_token"]
        self._refresh_token = payload.get("refresh_token", self._refresh_token)
        self._access_ttl    = int(payload.get("expires_in", 600))
        self._issued_at     = time.time()

    def get(self):
        """Return a valid access token, refreshing silently when near expiry."""
        age = time.time() - self._issued_at
        if age >= (self._access_ttl - self._refresh_before):
            self._refresh()
        return self._access_token


# ----------------------------------------------------------------------
# Catalogue query (no auth required) — paginated
# ----------------------------------------------------------------------
def build_filter(start, end, max_cloud):
    parts = [
        f"Collection/Name eq '{COLLECTION}'",
        f"contains(Name,'_{TILE}_')",
        ("Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
         f"and att/OData.CSC.StringAttribute/Value eq '{PRODUCT_TYPE}')"),
        f"ContentDate/Start gt {start}T00:00:00.000Z",
        f"ContentDate/Start lt {end}T00:00:00.000Z",
    ]
    if max_cloud is not None:
        parts.append(
            "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {float(max_cloud)})")
    return " and ".join(parts)


def query_products(start, end, max_cloud, max_products):
    params = {
        "$filter": build_filter(start, end, max_cloud),
        "$orderby": "ContentDate/Start asc",
        "$expand": "Attributes",
        "$top": "100",
    }
    url = f"{CATALOGUE}/Products"
    out = []
    while url and len(out) < max_products:
        r = requests.get(url, params=params if "?" not in url else None, timeout=120)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
        params = None  # nextLink already carries the query
    return out[:max_products]


def attr_value(product, name):
    for a in product.get("Attributes", []):
        if a.get("Name") == name:
            return a.get("Value")
    return None


# ----------------------------------------------------------------------
# Nodes API (auth required) — fetch a file or list a folder without bulk download
# ----------------------------------------------------------------------
def _nodes_path(product_uuid, parts):
    seg = "".join(f"/Nodes({p})" for p in parts)
    return f"{ZIPPER}/Products({product_uuid}){seg}"


def list_nodes(tm, product_uuid, parts):
    url = _nodes_path(product_uuid, parts) + "/Nodes"
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {tm.get()}"})
    r = sess.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()
    data = r.json()
    # CDSE returns {"result":[...]} for Nodes listings
    items = data.get("result") or data.get("value") or []
    return [it.get("Name") or it.get("Id") for it in items if it.get("Name") or it.get("Id")]


def fetch_node_bytes(tm, product_uuid, parts):
    url = _nodes_path(product_uuid, parts) + "/$value"
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {tm.get()}"})
    r = sess.get(url, timeout=180, allow_redirects=True)
    r.raise_for_status()
    return r.content


# ----------------------------------------------------------------------
# Quality-XML parsers (work regardless of namespace / encoding variant)
# ----------------------------------------------------------------------
def parse_product_flags(xml_bytes):
    root = ET.fromstring(xml_bytes)
    flags = {v: None for v in QUALITY_CHECKS.values()}
    degraded = {"degraded_anc_pct": None, "degraded_msi_pct": None}

    for el in root.iter():
        tag = _local(el.tag)
        text = (el.text or "").strip().upper()

        # Encoding A: <quality_check checkType="SENSOR_QUALITY">PASSED</quality_check>
        if tag.lower() == "quality_check":
            ct = None
            for a, v in el.attrib.items():
                if _local(a).lower() in ("checktype", "name", "type"):
                    ct = (v or "").upper()
            if ct in QUALITY_CHECKS and text:
                flags[QUALITY_CHECKS[ct]] = text

        # Encoding B: <SENSOR_QUALITY_FLAG>PASSED</SENSOR_QUALITY_FLAG> (or no _FLAG)
        key = tag.upper().replace("_FLAG", "")
        if key in QUALITY_CHECKS and text:
            flags[QUALITY_CHECKS[key]] = text

        # Technical quality percentages
        if tag.upper() == "DEGRADED_ANC_DATA_PERCENTAGE" and text:
            degraded["degraded_anc_pct"] = _to_float(text)
        if tag.upper() == "DEGRADED_MSI_DATA_PERCENTAGE" and text:
            degraded["degraded_msi_pct"] = _to_float(text)

    return flags, degraded


def parse_tile_qi(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = {"nodata_pct": None, "saturated_defective_pct": None}
    for el in root.iter():
        tag = _local(el.tag).upper()
        text = (el.text or "").strip()
        if tag == "NODATA_PIXEL_PERCENTAGE" and text:
            out["nodata_pct"] = _to_float(text)
        if tag == "SATURATED_DEFECTIVE_PIXEL_PERCENTAGE" and text:
            out["saturated_defective_pct"] = _to_float(text)
    return out


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def summarize_flags(flags):
    vals = [v for v in flags.values() if v]
    failed = [name for name, v in flags.items() if v == "FAILED"]
    if not vals:
        esa_flag = "UNKNOWN"
    elif failed:
        esa_flag = "FAILED"
    else:
        esa_flag = "PASSED"
    return esa_flag, ";".join(failed)


# ----------------------------------------------------------------------
# Mask reading (full mode only; needs rasterio)
# ----------------------------------------------------------------------
def read_masks(safe_dir):
    counts = {f"mask_{k}_px": None for k in MASK_LAYERS}
    try:
        import numpy as np
        import rasterio
    except ImportError:
        return counts
    masks = list(Path(safe_dir).rglob("MSK_QUALIT_B04.jp2"))
    if not masks:
        return counts
    try:
        with rasterio.open(masks[0]) as src:
            arr = src.read()  # (bands, h, w)
            for name, layer in MASK_LAYERS.items():
                if arr.shape[0] >= layer:
                    counts[f"mask_{name}_px"] = int((arr[layer - 1] > 0).sum())
    except Exception as e:
        print(f"    (mask read failed: {e})")
    return counts


# ----------------------------------------------------------------------
# Per-product processing
# ----------------------------------------------------------------------
def find_granule_dir(tm, uuid, name):
    """Return the single GRANULE/<id> folder name via the Nodes API."""
    children = list_nodes(tm, uuid, [name, "GRANULE"])
    return children[0] if children else None


def process_metadata(tm, product):
    uuid = product["Id"]
    name = product["Name"]                     # ends with .SAFE
    rec = _base_record(product)

    # Product-level flags
    pmtd = fetch_node_bytes(tm, uuid, [name, "MTD_MSIL1C.xml"])
    flags, degraded = parse_product_flags(pmtd)
    rec.update(flags)
    rec.update(degraded)

    # Granule-level percentages (best effort)
    try:
        gran = find_granule_dir(tm, uuid, name)
        if gran:
            tmtd = fetch_node_bytes(tm, uuid, [name, "GRANULE", gran, "MTD_TL.xml"])
            rec.update(parse_tile_qi(tmtd))
    except Exception as e:
        print(f"    (granule QI skipped: {e})")

    _finalize(rec, flags)
    return rec


def process_full(tm, product, download_dir, read_masks_flag):
    uuid = product["Id"]
    name = product["Name"]
    rec = _base_record(product)

    safe_dir = download_product(tm, uuid, name, download_dir)

    pmtd = next(Path(safe_dir).glob("MTD_MSIL1C.xml"), None)
    if pmtd:
        flags, degraded = parse_product_flags(pmtd.read_bytes())
        rec.update(flags)
        rec.update(degraded)
    else:
        flags = {v: None for v in QUALITY_CHECKS.values()}

    tmtd = next(Path(safe_dir).rglob("MTD_TL.xml"), None)
    if tmtd:
        rec.update(parse_tile_qi(tmtd.read_bytes()))

    if read_masks_flag:
        rec.update(read_masks(safe_dir))

    _finalize(rec, flags)
    return rec


def download_product(tm, uuid, name, download_dir):
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    zip_path = Path(download_dir) / f"{name}.zip"
    safe_dir = Path(download_dir) / name
    if safe_dir.exists():
        return safe_dir

    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {tm.get()}"})
    url = f"{ZIPPER}/Products({uuid})/$value"
    r = sess.get(url, allow_redirects=True, stream=True, timeout=600)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(download_dir)
    zip_path.unlink(missing_ok=True)
    # the extracted root folder is the .SAFE
    return safe_dir if safe_dir.exists() else next(Path(download_dir).glob("*.SAFE"))


def _base_record(product):
    name = product["Name"]
    sensing = (product.get("ContentDate", {}) or {}).get("Start", "")
    return {
        "scene_name": name,
        "base_scene": name,
        "tile": TILE,
        "platform": attr_value(product, "platformShortName"),
        "sensing_date": sensing,
        "processing_baseline": attr_value(product, "processingBaseline"),
        "product_uuid": product["Id"],
        "defect_family": "ESA_REAL",
        "cloud_cover_pct": attr_value(product, "cloudCover"),
        "source": "ESA_OLQC",
        # default Nones for optional columns
        **{c: None for c in (
            "sensor_quality", "geometric_quality", "general_quality",
            "format_correctness", "radiometric_quality",
            "degraded_anc_pct", "degraded_msi_pct",
            "nodata_pct", "saturated_defective_pct",
            "mask_defective_px", "mask_nodata_px",
            "mask_lost_packet_px", "mask_saturated_px")},
    }


def _finalize(rec, flags):
    esa_flag, failed = summarize_flags(flags)
    rec["esa_flag"] = esa_flag
    rec["failed_indicator"] = failed
    rec["expected_reject"] = (esa_flag == "FAILED")
    rec["defect_type"] = (f"ESA_FAILED:{failed}" if esa_flag == "FAILED"
                          else f"ESA_{esa_flag}")


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def write_csv(records, out_csv):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    print(f"\nSaved CSV: {out_csv}  ({len(records)} scenes)")


def write_db(records, db_path="reports/eo_qc.db"):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS esa_reference (
            scene_name TEXT PRIMARY KEY,
            tile TEXT, platform TEXT, sensing_date TEXT,
            processing_baseline TEXT, product_uuid TEXT,
            esa_flag TEXT, failed_indicator TEXT, expected_reject BOOLEAN,
            sensor_quality TEXT, geometric_quality TEXT, general_quality TEXT,
            format_correctness TEXT, radiometric_quality TEXT,
            degraded_anc_pct REAL, degraded_msi_pct REAL,
            nodata_pct REAL, saturated_defective_pct REAL, cloud_cover_pct REAL,
            mask_defective_px INTEGER, mask_nodata_px INTEGER,
            mask_lost_packet_px INTEGER, mask_saturated_px INTEGER,
            source TEXT
        )
    """)
    cols = [c for c in CSV_COLUMNS if c not in ("base_scene", "defect_family", "defect_type")]
    placeholders = ",".join("?" for _ in cols)
    for rec in records:
        conn.execute(
            f"INSERT OR REPLACE INTO esa_reference ({','.join(cols)}) VALUES ({placeholders})",
            [rec.get(c) for c in cols])
    conn.commit()
    conn.close()
    print(f"Wrote {len(records)} rows to esa_reference in {db_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    today = dt.date.today().isoformat()
    ap = argparse.ArgumentParser(description=f"Label real ESA-flagged {PRODUCT_TYPE} scenes over {TILE}")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=today)
    ap.add_argument("--max-cloud", type=float, default=None,
                    help="Cloud-cover ceiling to isolate non-cloud defects (e.g. 20)")
    ap.add_argument("--max-products", type=int, default=200)
    ap.add_argument("--mode", choices=["metadata", "full"], default="metadata")
    ap.add_argument("--read-masks", action="store_true",
                    help="(full mode) count MSK_QUALIT defect pixels")
    ap.add_argument("--download-dir", default="data/esa_reference")
    ap.add_argument("--out-csv", default=f"reports/esa_reference_{TILE}.csv")
    ap.add_argument("--write-db", action="store_true")
    ap.add_argument("--username", default=os.environ.get("CDSE_USERNAME"))
    ap.add_argument("--password", default=os.environ.get("CDSE_PASSWORD"))
    args = ap.parse_args()

    if not args.username or not args.password:
        sys.exit("CDSE credentials required: set CDSE_USERNAME/CDSE_PASSWORD or pass --username/--password")

    print(f"Querying CDSE: {COLLECTION}/{PRODUCT_TYPE} tile {TILE} "
          f"{args.start}..{args.end} "
          f"{'(cloud<=%g)' % args.max_cloud if args.max_cloud is not None else '(no cloud filter)'}")
    products = query_products(args.start, args.end, args.max_cloud, args.max_products)
    print(f"Found {len(products)} products\n")
    if not products:
        return

    tm = TokenManager(args.username, args.password)

    records = []
    for i, p in enumerate(products, 1):
        name = p["Name"]
        print(f"[{i}/{len(products)}] {name}")
        try:
            if args.mode == "metadata":
                rec = process_metadata(tm, p)
            else:
                rec = process_full(tm, p, args.download_dir, args.read_masks)
            tag = rec["esa_flag"]
            extra = f" ({rec['failed_indicator']})" if rec["failed_indicator"] else ""
            print(f"    -> {tag}{extra}")
            records.append(rec)
        except Exception as e:
            print(f"    ERROR: {e}")

    write_csv(records, args.out_csv)
    if args.write_db:
        write_db(records)

    # Summary
    n_failed = sum(1 for r in records if r["esa_flag"] == "FAILED")
    n_unknown = sum(1 for r in records if r["esa_flag"] == "UNKNOWN")
    print(f"\n{'='*60}")
    print(f"ESA-FLAGGED SUMMARY (tile {TILE})")
    print(f"{'='*60}")
    print(f"Scanned : {len(records)}")
    print(f"FAILED  : {n_failed}   <- real defective scenes (use as positives)")
    print(f"PASSED  : {len(records) - n_failed - n_unknown}")
    print(f"UNKNOWN : {n_unknown}")
    if n_failed:
        from collections import Counter
        c = Counter()
        for r in records:
            for ind in (r["failed_indicator"] or "").split(";"):
                if ind:
                    c[ind] += 1
        print("Failed-indicator breakdown:")
        for ind, n in c.most_common():
            print(f"  {ind}: {n}")


if __name__ == "__main__":
    main()