# src/database.py
import sqlite3
from pathlib import Path
from datetime import datetime


DB_PATH = "reports/eo_qc.db"


def get_connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _ensure_columns(conn, table, columns):
    """Add missing columns to existing table."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for col, coltype in columns.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
    conn.commit()


def init_db():
    conn = get_connection()
    
    # Core tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scenes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_name TEXT UNIQUE NOT NULL,
            scene_path TEXT NOT NULL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accepted BOOLEAN NOT NULL,
            failed_filter TEXT,
            failure_reason TEXT,
            cloud_cover REAL,
            noise_std_ratio REAL,
            blur_variance REAL,
            stripe_score REAL
        );

        CREATE TABLE IF NOT EXISTS defects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_name TEXT UNIQUE NOT NULL,
            base_scene TEXT,
            defect_type TEXT NOT NULL,
            defect_family TEXT,
            severity REAL,
            caught BOOLEAN NOT NULL,
            failed_filter TEXT,
            failure_reason TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            noise_std_ratio REAL,
            blur_variance REAL,
            stripe_score REAL
        );

        CREATE INDEX IF NOT EXISTS idx_accepted ON scenes(accepted);
        CREATE INDEX IF NOT EXISTS idx_date ON scenes(processed_at);
    """)

    # Migrate old defects table
    _ensure_columns(conn, "defects", {
        "base_scene": "TEXT",
        "defect_family": "TEXT",
        "severity": "REAL",
    })

    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_defect_family ON defects(defect_family);
        CREATE INDEX IF NOT EXISTS idx_defect_severity ON defects(defect_family, severity);
    """)

    # ESA reference table for real-world validation
    conn.execute("""
        CREATE TABLE IF NOT EXISTS esa_reference (
            scene_name TEXT PRIMARY KEY,
            tile TEXT,
            platform TEXT,
            sensing_date TEXT,
            processing_baseline TEXT,
            product_uuid TEXT,
            esa_flag TEXT,
            failed_indicator TEXT,
            expected_reject BOOLEAN,
            sensor_quality TEXT,
            geometric_quality TEXT,
            general_quality TEXT,
            format_correctness TEXT,
            radiometric_quality TEXT,
            degraded_anc_pct REAL,
            degraded_msi_pct REAL,
            nodata_pct REAL,
            saturated_defective_pct REAL,
            cloud_cover_pct REAL,
            mask_defective_px INTEGER,
            mask_nodata_px INTEGER,
            mask_lost_packet_px INTEGER,
            mask_saturated_px INTEGER,
            source TEXT
        )
    """)
    
    conn.commit()
    return conn


def log_scene(conn, scene_path, result):
    scene_name = Path(scene_path).name
    results = result["results"]

    metrics = {}
    for fname, fres in results.items():
        metrics[fname] = fres.get("metrics", {})

    conn.execute("""
        INSERT OR REPLACE INTO scenes
        (scene_name, scene_path, accepted, failed_filter, failure_reason,
         cloud_cover, noise_std_ratio, blur_variance, stripe_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scene_name,
        str(scene_path),
        result["accepted"],
        result.get("failed_filter"),
        result.get("failure_reason"),
        metrics.get("MetadataFilter", {}).get("cloud_cover"),
        metrics.get("NoiseFilter", {}).get("noise_std_ratio"),
        metrics.get("BlurFilter", {}).get("laplacian_variance"),
        metrics.get("StripeFilter", {}).get("periodic_power_ratio"),
    ))
    conn.commit()


def log_defect(conn, scene_path, defect_type, result,
               base_scene=None, severity=None, defect_family=None):
    scene_name = Path(scene_path).name

    if base_scene is None:
        base_scene = scene_name

    results = result["results"]
    metrics = {}
    for fname, fres in results.items():
        metrics[fname] = fres.get("metrics", {})

    conn.execute("""
        INSERT OR REPLACE INTO defects
        (scene_name, base_scene, defect_type, defect_family, severity,
         caught, failed_filter, failure_reason,
         noise_std_ratio, blur_variance, stripe_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scene_name,
        base_scene,
        defect_type,
        defect_family,
        severity,
        not result["accepted"],
        result.get("failed_filter"),
        result.get("failure_reason"),
        metrics.get("NoiseFilter", {}).get("noise_std_ratio"),
        metrics.get("BlurFilter", {}).get("laplacian_variance"),
        metrics.get("StripeFilter", {}).get("periodic_power_ratio"),
    ))
    conn.commit()


def log_esa_reference(conn, record):
    """Log one ESA reference scene."""
    conn.execute("""
        INSERT OR REPLACE INTO esa_reference
        (scene_name, tile, platform, sensing_date, processing_baseline,
         product_uuid, esa_flag, failed_indicator, expected_reject,
         sensor_quality, geometric_quality, general_quality,
         format_correctness, radiometric_quality,
         degraded_anc_pct, degraded_msi_pct,
         nodata_pct, saturated_defective_pct, cloud_cover_pct,
         mask_defective_px, mask_nodata_px,
         mask_lost_packet_px, mask_saturated_px,
         source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("scene_name"),
        record.get("tile"),
        record.get("platform"),
        record.get("sensing_date"),
        record.get("processing_baseline"),
        record.get("product_uuid"),
        record.get("esa_flag"),
        record.get("failed_indicator"),
        record.get("expected_reject"),
        record.get("sensor_quality"),
        record.get("geometric_quality"),
        record.get("general_quality"),
        record.get("format_correctness"),
        record.get("radiometric_quality"),
        record.get("degraded_anc_pct"),
        record.get("degraded_msi_pct"),
        record.get("nodata_pct"),
        record.get("saturated_defective_pct"),
        record.get("cloud_cover_pct"),
        record.get("mask_defective_px"),
        record.get("mask_nodata_px"),
        record.get("mask_lost_packet_px"),
        record.get("mask_saturated_px"),
        record.get("source"),
    ))
    conn.commit()


def get_stats(conn):
    total = conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
    accepted = conn.execute("SELECT COUNT(*) FROM scenes WHERE accepted=1").fetchone()[0]

    reasons = conn.execute("""
        SELECT failed_filter, COUNT(*)
        FROM scenes WHERE accepted=0
        GROUP BY failed_filter
    """).fetchall()

    return {
        "total": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "rate": accepted / total if total else 0,
        "reasons": reasons,
    }


def get_defect_stats(conn):
    total = conn.execute("SELECT COUNT(*) FROM defects").fetchone()[0]
    caught = conn.execute("SELECT COUNT(*) FROM defects WHERE caught=1").fetchone()[0]

    types = conn.execute("""
        SELECT defect_type, COUNT(*) as total, SUM(caught) as caught
        FROM defects
        GROUP BY defect_type
    """).fetchall()

    return {
        "total": total,
        "caught": caught,
        "missed": total - caught,
        "rate": caught / total if total else 0,
        "by_type": types,
    }


def get_sensitivity_stats(conn):
    rows = conn.execute("""
        SELECT defect_family, severity,
               COUNT(*)            AS n_scenes,
               SUM(caught)         AS n_caught,
               1.0 * SUM(caught) / COUNT(*) AS detection_rate
        FROM defects
        WHERE severity IS NOT NULL
        GROUP BY defect_family, severity
        ORDER BY defect_family, severity
    """).fetchall()
    return rows


def get_esa_validation_stats(conn):
    """Compare pipeline results against ESA ground truth."""
    rows = conn.execute("""
        SELECT 
            e.scene_name,
            e.esa_flag,
            e.expected_reject,
            s.accepted as pipeline_accepted,
            CASE 
                WHEN e.expected_reject = 1 AND s.accepted = 0 THEN 'TRUE_POSITIVE'
                WHEN e.expected_reject = 0 AND s.accepted = 1 THEN 'TRUE_NEGATIVE'
                WHEN e.expected_reject = 1 AND s.accepted = 1 THEN 'FALSE_NEGATIVE'
                WHEN e.expected_reject = 0 AND s.accepted = 0 THEN 'FALSE_POSITIVE'
                ELSE 'NOT_PROCESSED'
            END as validation_result
        FROM esa_reference e
        LEFT JOIN scenes s ON e.scene_name = s.scene_name
    """).fetchall()
    return rows