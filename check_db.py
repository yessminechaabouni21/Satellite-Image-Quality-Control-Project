import sqlite3
from pathlib import Path

DB_PATH = "reports/eo_qc.db"

def check_db():
    if not Path(DB_PATH).exists():
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    print("=" * 70)
    print("CLEAN SCENES (latest 5)")
    print("=" * 70)
    for row in conn.execute("""
        SELECT scene_name, accepted, failed_filter, processed_at 
        FROM scenes 
        ORDER BY processed_at DESC 
        LIMIT 5
    """):
        status = "✅ PASS" if row[1] else "❌ FAIL"
        fail = row[2] or "-"
        print(f"{status:8} | {fail:15} | {row[3]} | {row[0][:40]}")

    print("\n" + "=" * 70)
    print("DEFECTS (latest 5)")
    print("=" * 70)
    for row in conn.execute("""
        SELECT defect_type, caught, failed_filter, processed_at 
        FROM defects 
        ORDER BY processed_at DESC 
        LIMIT 5
    """):
        status = "✅ CAUGHT" if row[1] else "❌ MISSED"
        fail = row[2] or "-"
        print(f"{status:10} | {fail:15} | {row[3]} | {row[0]}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    clean = conn.execute("SELECT COUNT(*), SUM(accepted) FROM scenes").fetchone()
    print(f"Clean scenes:   {clean[1]}/{clean[0]} accepted")
    
    defect = conn.execute("SELECT COUNT(*), SUM(caught) FROM defects").fetchone()
    print(f"Defects caught: {defect[1]}/{defect[0]} ({100*defect[1]/defect[0]:.0f}%)")

    if clean[0] > 0:
        print(f"\nClean rejection reasons:")
        for reason, count in conn.execute("""
            SELECT failed_filter, COUNT(*) 
            FROM scenes WHERE accepted=0 
            GROUP BY failed_filter
        """):
            print(f"  {reason}: {count}")

    print("\n" + "=" * 70)
    print("CATCH RATE BY DEFECT TYPE")
    print("=" * 70)
    for row in conn.execute("""
        SELECT defect_type, COUNT(*) as total, SUM(caught) as caught,
               ROUND(100.0 * SUM(caught) / COUNT(*), 0) as rate
        FROM defects 
        GROUP BY defect_type
    """):
        status = "✅" if row[3] == 100 else "⚠️"
        print(f"{status} {row[0]:20s} | {row[2]}/{row[1]} | {row[3]:.0f}%")

    conn.close()


if __name__ == "__main__":
    check_db()