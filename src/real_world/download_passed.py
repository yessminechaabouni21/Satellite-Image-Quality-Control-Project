# src/real_world/download_passed_sample.py
#
# Selects a stratified sample of ESA-PASSED scenes from esa_reference table
# and downloads them for use in the confusion matrix evaluation.
#
# Stratification: seasons x platform (S2A/S2B/S2C) so the sample is
# representative of the full 192-scene distribution.
#
# Run from repo root (use single quotes around password in PowerShell):
#   python -m src.real_world.download_passed_sample \
#       --username you@email.com --password 'yourpass' --n 40
#
import os, sys, time, zipfile, argparse, sqlite3, requests
from pathlib import Path
from datetime import datetime

DB_PATH      = "reports/eo_qc.db"
DOWNLOAD_DIR = "data/esa_reference"
ZIPPER       = "https://zipper.dataspace.copernicus.eu/odata/v1"
TOKEN_URL    = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
                "/protocol/openid-connect/token")


# ── token manager ─────────────────────────────────────────────────────────────
class TokenManager:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self._access_token = self._refresh_token = None
        self._issued_at = 0
        self._access_ttl = 600
        self._login()

    def _login(self):
        r = requests.post(TOKEN_URL, timeout=60, data={
            "client_id": "cdse-public", "grant_type": "password",
            "username": self.username, "password": self.password,
        })
        if r.status_code == 401:
            raise SystemExit(
                f"Login failed: {r.json().get('error_description', '')}\n"
                "Use single quotes around the password in PowerShell.")
        r.raise_for_status()
        self._store(r.json())
        print(f"  [auth] logged in (token valid {self._access_ttl}s)")

    def _refresh(self):
        r = requests.post(TOKEN_URL, timeout=60, data={
            "client_id": "cdse-public", "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        })
        if r.status_code in (400, 401):
            print("  [auth] refresh expired, re-logging in...")
            self._login()
            return
        r.raise_for_status()
        self._store(r.json())
        print("  [auth] token refreshed")

    def _store(self, p):
        self._access_token  = p["access_token"]
        self._refresh_token = p.get("refresh_token", self._refresh_token)
        self._access_ttl    = int(p.get("expires_in", 600))
        self._issued_at     = time.time()

    def get(self):
        if time.time() - self._issued_at >= self._access_ttl - 60:
            self._refresh()
        return self._access_token


# ── stratified sampler ────────────────────────────────────────────────────────
def get_season(date_str):
    """Return DJF/MAM/JJA/SON from a sensing_date string."""
    try:
        month = datetime.fromisoformat(date_str.replace("Z", "")).month
    except Exception:
        return "UNK"
    return {12: "DJF", 1: "DJF", 2: "DJF",
             3: "MAM", 4: "MAM", 5: "MAM",
             6: "JJA", 7: "JJA", 8: "JJA",
             9: "SON", 10: "SON", 11: "SON"}[month]

def get_platform(scene_name):
    """Extract S2A / S2B / S2C from scene name."""
    if scene_name.startswith("S2A"): return "S2A"
    if scene_name.startswith("S2B"): return "S2B"
    if scene_name.startswith("S2C"): return "S2C"
    return "UNK"

def stratified_sample(rows, n):
    """
    rows: list of (scene_name, product_uuid, sensing_date, cloud_cover_pct)
    Returns up to n rows, spread evenly across (season x platform) strata.
    Already-downloaded scenes are prioritised so we skip re-downloading.
    """
    from collections import defaultdict
    import random

    # Annotate each row with its stratum
    annotated = []
    for scene_name, uuid, sensing_date, cloud in rows:
        already = (Path(DOWNLOAD_DIR) / scene_name).exists()
        stratum = f"{get_season(sensing_date or '')}_{get_platform(scene_name)}"
        annotated.append((stratum, already, scene_name, uuid, sensing_date, cloud))

    # Sort: already-downloaded first (free), then by stratum
    annotated.sort(key=lambda x: (not x[1], x[0]))

    # Fill strata round-robin until we have n
    buckets = defaultdict(list)
    for row in annotated:
        buckets[row[0]].append(row)

    selected = []
    strata = sorted(buckets.keys())
    i = 0
    while len(selected) < n and any(buckets[s] for s in strata):
        s = strata[i % len(strata)]
        if buckets[s]:
            selected.append(buckets[s].pop(0))
        i += 1

    return selected


# ── downloader ────────────────────────────────────────────────────────────────
def download_scene(tm, uuid, name, out_dir):
    safe_dir = Path(out_dir) / name
    if safe_dir.exists():
        print(f"  already on disk — skipping")
        return safe_dir

    zip_path = Path(out_dir) / f"{name}.zip"
    url  = f"{ZIPPER}/Products({uuid})/$value"
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {tm.get()}"})

    r = sess.get(url, stream=True, timeout=600, allow_redirects=True)
    if r.status_code != 200:
        print(f"  FAILED — HTTP {r.status_code}: {r.text[:200]}")
        return None

    total      = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded/total*100:5.1f}%  "
                      f"{downloaded>>20} / {total>>20} MB",
                      end="", flush=True)
    print()

    print("  unzipping...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    zip_path.unlink(missing_ok=True)

    if not safe_dir.exists():
        found = list(Path(out_dir).rglob(name))
        if found:
            safe_dir = found[0]

    return safe_dir if safe_dir.exists() else None


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=os.environ.get("CDSE_USERNAME"))
    ap.add_argument("--password", default=os.environ.get("CDSE_PASSWORD"))
    ap.add_argument("--db",           default=DB_PATH)
    ap.add_argument("--download-dir", default=DOWNLOAD_DIR)
    ap.add_argument("--n",  type=int, default=40,
                    help="Number of PASSED scenes to download (default 40)")
    ap.add_argument("--max-cloud", type=float, default=60.0,
                    help="Skip PASSED scenes with cloud > this (default 60)")
    args = ap.parse_args()

    if not args.username or not args.password:
        sys.exit("Provide --username and --password "
                 "(or set CDSE_USERNAME/CDSE_PASSWORD).")

    if not Path(args.db).exists():
        sys.exit(f"DB not found: {args.db} — run esa_reference_T32SPF.py first.")

    conn = sqlite3.connect(args.db)

    # All PASSED scenes not already used as FAILED
    rows = conn.execute("""
        SELECT scene_name, product_uuid, sensing_date, cloud_cover_pct
        FROM   esa_reference
        WHERE  esa_flag = 'PASSED'
          AND  (cloud_cover_pct IS NULL OR cloud_cover_pct <= ?)
        ORDER  BY sensing_date
    """, (args.max_cloud,)).fetchall()
    conn.close()

    if not rows:
        sys.exit("No PASSED scenes found in DB. "
                 "Run esa_reference_T32SPF.py first.")

    print(f"\nPASSED scenes in DB (cloud ≤ {args.max_cloud}%): {len(rows)}")

    # Stratified sample
    sample = stratified_sample(rows, args.n)

    print(f"Selected {len(sample)} scenes for download "
          f"(stratified by season × platform):\n")
    print(f"  {'#':<3} {'Already':7} {'Stratum':12} {'Cloud%':7}  Scene")
    print(f"  {'-'*3} {'-'*7} {'-'*12} {'-'*7}  {'-'*50}")
    for i, (stratum, already, name, uuid, sensing_date, cloud) in \
            enumerate(sample, 1):
        flag = "yes" if already else "no"
        cloud_s = f"{cloud:.1f}" if cloud is not None else "?"
        print(f"  {i:<3} {flag:7} {stratum:12} {cloud_s:>7}%  {name[:55]}")

    Path(args.download_dir).mkdir(parents=True, exist_ok=True)
    tm = TokenManager(args.username, args.password)

    ok, failed = [], []
    need_download = [(s, n, u) for s, already, n, u, *_ in sample if not already]
    skip_count   = len(sample) - len(need_download)

    print(f"\nAlready on disk: {skip_count}  |  To download: {len(need_download)}\n")

    for i, (stratum, already, name, uuid, sensing_date, cloud) in \
            enumerate(sample, 1):
        print(f"\n[{i}/{len(sample)}] {name[:60]}")
        print(f"  stratum={stratum}  cloud={cloud}%")
        safe = download_scene(tm, uuid, name, args.download_dir)
        if safe:
            ok.append(name)
            print(f"  -> {safe}")
        else:
            failed.append(name)

    # ── summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Requested : {len(sample)}")
    print(f"Available : {len(ok)}")
    print(f"Failed    : {len(failed)}")
    if failed:
        print("Failed scenes:")
        for n in failed:
            print(f"  {n}")

    total_on_disk = len(list(Path(args.download_dir).glob("*.SAFE")))
    failed_on_disk = len([
        p for p in Path(args.download_dir).glob("*.SAFE")
        if "FAILED" in p.name  # rough check; will be joined via DB
    ])
    print(f"\nTotal .SAFE in {args.download_dir}: {total_on_disk}")
    print(f"\nNext step — run the confusion matrix:")
    print(f"  python -m src.real_world.confusion_matrix "
          f"--username ... --password '...'")


if __name__ == "__main__":
    main()