# src/real_world/download_failed_scenes.py
#
# Downloads ONLY the ESA-FAILED scenes already recorded in esa_reference table.
# Reads UUIDs directly from the DB — no re-querying the catalogue.
#
# Run from repo root:
#   python -m src.real_world.download_failed_scenes \
#       --username you@email.com --password 'yourpass'
#
import os, sys, time, zipfile, argparse, sqlite3, requests
from pathlib import Path

DB_PATH      = "reports/eo_qc.db"
DOWNLOAD_DIR = "data/esa_reference"
ZIPPER       = "https://zipper.dataspace.copernicus.eu/odata/v1"
TOKEN_URL    = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
                "/protocol/openid-connect/token")


# ── token manager (same as esa_reference_T32SPF.py) ──────────────────────────
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
                f"Login failed: {r.json().get('error_description','')}\n"
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
            self._login(); return
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


# ── download one product by UUID ──────────────────────────────────────────────
def download_scene(tm, uuid, name, out_dir):
    safe_dir = Path(out_dir) / name
    if safe_dir.exists():
        print(f"  already on disk: {safe_dir}")
        return safe_dir

    zip_path = Path(out_dir) / f"{name}.zip"
    url  = f"{ZIPPER}/Products({uuid})/$value"
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {tm.get()}"})

    print(f"  downloading ({url[:60]}...)")
    r = sess.get(url, stream=True, timeout=600, allow_redirects=True)
    if r.status_code != 200:
        print(f"  FAILED — HTTP {r.status_code}: {r.text[:200]}")
        return None

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):   # 1 MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:5.1f}%  {downloaded>>20} MB / {total>>20} MB",
                      end="", flush=True)
    print()

    print(f"  unzipping...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    zip_path.unlink(missing_ok=True)

    # CDSE zips sometimes nest the .SAFE one level deeper
    if not safe_dir.exists():
        found = list(Path(out_dir).rglob(f"{name}"))
        if found:
            safe_dir = found[0]

    print(f"  -> {safe_dir}")
    return safe_dir


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=os.environ.get("CDSE_USERNAME"))
    ap.add_argument("--password", default=os.environ.get("CDSE_PASSWORD"))
    ap.add_argument("--db",          default=DB_PATH)
    ap.add_argument("--download-dir", default=DOWNLOAD_DIR)
    ap.add_argument("--flag",        default="FAILED",
                    help="Which esa_flag to download: FAILED / PASSED / all")
    args = ap.parse_args()

    if not args.username or not args.password:
        sys.exit("Provide --username and --password (or set CDSE_USERNAME/CDSE_PASSWORD).")

    if not Path(args.db).exists():
        sys.exit(f"DB not found: {args.db}  — run esa_reference_T32SPF.py first.")

    conn = sqlite3.connect(args.db)

    # ── query the scenes to download ──
    if args.flag.lower() == "all":
        where = ""
    else:
        where = f"WHERE esa_flag = '{args.flag.upper()}'"

    rows = conn.execute(
        f"SELECT scene_name, product_uuid, esa_flag, failed_indicator "
        f"FROM esa_reference {where} ORDER BY sensing_date"
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No scenes with esa_flag='{args.flag}' found in {args.db}.")
        print("Tip: run esa_reference_T32SPF.py first to populate the table.")
        return

    print(f"\nScenes to download: {len(rows)}  (esa_flag={args.flag})")
    for name, uuid, flag, ind in rows:
        print(f"  {flag:6s}  {ind or '-':30s}  {name[:55]}")

    Path(args.download_dir).mkdir(parents=True, exist_ok=True)
    tm = TokenManager(args.username, args.password)

    ok, failed = [], []
    for i, (name, uuid, flag, ind) in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}] {name}")
        print(f"  esa_flag={flag}  indicator={ind or '-'}")
        safe = download_scene(tm, uuid, name, args.download_dir)
        if safe:
            ok.append(name)
        else:
            failed.append(name)

    print(f"\n{'='*60}")
    print(f"Downloaded OK : {len(ok)}")
    print(f"Failed        : {len(failed)}")
    if failed:
        print("Failed scenes:")
        for n in failed:
            print(f"  {n}")
    if ok:
        print(f"\nScenes are in: {Path(args.download_dir).resolve()}")
        print("Next step: run your pipeline on this folder:")
        print(f"  python -m src.real_world.run_pipeline_on_esa_reference")


if __name__ == "__main__":
    main()