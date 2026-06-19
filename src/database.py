import sqlite3
from pathlib import Path
from datetime import datetime


DB_PATH = "reports/eo_qc.db"


def get_connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
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
            defect_type TEXT NOT NULL,
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
        metrics.get("StripeFilter", {}).get("stripe_power_ratio")
    ))
    conn.commit()


def log_defect(conn, scene_path, defect_type, result):
    scene_name = Path(scene_path).name
    results = result["results"]
    
    metrics = {}
    for fname, fres in results.items():
        metrics[fname] = fres.get("metrics", {})
    
    conn.execute("""
        INSERT OR REPLACE INTO defects 
        (scene_name, defect_type, caught, failed_filter, failure_reason,
         noise_std_ratio, blur_variance, stripe_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scene_name,
        defect_type,
        not result["accepted"],
        result.get("failed_filter"),
        result.get("failure_reason"),
        metrics.get("NoiseFilter", {}).get("noise_std_ratio"),
        metrics.get("BlurFilter", {}).get("laplacian_variance"),
        metrics.get("StripeFilter", {}).get("stripe_power_ratio")
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
        "reasons": reasons
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
        "by_type": types
    }