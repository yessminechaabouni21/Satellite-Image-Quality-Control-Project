import sqlite3
from pathlib import Path

DB_PATH = "reports/eo_qc.db"


def _has_column(conn, table, col):
    return col in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def check_db():
    if not Path(DB_PATH).exists():
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    has_sev = _has_column(conn, "defects", "severity")
    has_base = _has_column(conn, "defects", "base_scene")
    has_fam = _has_column(conn, "defects", "defect_family")

    # ------------------------------------------------------------------
    print("=" * 78)
    print("CLEAN SCENES (latest 5)")
    print("=" * 78)
    for row in conn.execute("""
        SELECT scene_name, accepted, failed_filter, processed_at
        FROM scenes
        ORDER BY processed_at DESC
        LIMIT 5
    """):
        status = "PASS" if row[1] else "FAIL"
        fail = row[2] or "-"
        print(f"{status:5} | {fail:15} | {row[3]} | {row[0][:40]}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DEFECTS (latest 8)")
    print("=" * 78)
    sev_col = "severity" if has_sev else "NULL"
    base_col = "base_scene" if has_base else "NULL"
    for row in conn.execute(f"""
        SELECT defect_type, {sev_col}, caught, failed_filter, {base_col}
        FROM defects
        ORDER BY processed_at DESC
        LIMIT 8
    """):
        status = "CAUGHT" if row[2] else "MISSED"
        sev = "-" if row[1] is None else f"{row[1]:g}"
        fail = row[3] or "-"
        base = (row[4] or "-")[:34]
        print(f"{status:6} | sev={sev:>6} | {fail:15} | {base}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    clean = conn.execute("SELECT COUNT(*), SUM(accepted) FROM scenes").fetchone()
    if clean[0]:
        print(f"Clean scenes:   {clean[1]}/{clean[0]} accepted")

    defect = conn.execute("SELECT COUNT(*), SUM(caught) FROM defects").fetchone()
    if defect[0]:
        print(f"Defect runs:    {defect[1]}/{defect[0]} caught "
              f"({100*defect[1]/defect[0]:.0f}%)")
    else:
        print("Defect runs:    none logged yet")

    if has_base:
        n_base = conn.execute(
            "SELECT COUNT(DISTINCT base_scene) FROM defects").fetchone()[0]
        print(f"Base scenes used: {n_base}")

    if clean[0]:
        rejected = conn.execute(
            "SELECT COUNT(*) FROM scenes WHERE accepted=0").fetchone()[0]
        if rejected:
            print(f"\nClean rejection reasons (false positives):")
            for reason, count in conn.execute("""
                SELECT failed_filter, COUNT(*)
                FROM scenes WHERE accepted=0
                GROUP BY failed_filter
            """):
                print(f"  {reason}: {count}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("CATCH RATE BY DEFECT TYPE")
    print("=" * 78)
    for row in conn.execute("""
        SELECT defect_type, COUNT(*) AS total, SUM(caught) AS caught,
               ROUND(100.0 * SUM(caught) / COUNT(*), 0) AS rate
        FROM defects
        GROUP BY defect_type
        ORDER BY defect_type
    """):
        mark = "OK " if row[3] == 100 else "** "
        print(f"{mark}{row[0]:24s} | {row[2]}/{row[1]} | {row[3]:.0f}%")

    # ------------------------------------------------------------------
    # Sensitivity table: detection rate vs severity (the headline result)
    # ------------------------------------------------------------------
    if has_sev and has_fam:
        print("\n" + "=" * 78)
        print("DETECTION RATE vs SEVERITY  (sensitivity curve data)")
        print("=" * 78)
        rows = conn.execute("""
            SELECT defect_family, severity,
                   COUNT(*) AS n, SUM(caught) AS caught,
                   ROUND(100.0 * SUM(caught) / COUNT(*), 0) AS rate
            FROM defects
            WHERE severity IS NOT NULL
            GROUP BY defect_family, severity
            ORDER BY defect_family, severity
        """).fetchall()

        if not rows:
            print("  (no parametric defects logged yet — run test_defect.py)")
        else:
            current = None
            for fam, sev, n, caught, rate in rows:
                if fam != current:
                    print(f"\n{fam}:")
                    current = fam
                bar = "#" * int((rate or 0) / 10)
                print(f"  sev={sev:>8g} | {int(caught)}/{int(n)} | {rate:5.0f}% {bar}")

    conn.close()


if __name__ == "__main__":
    check_db()