# src/auto_pipeline.py
import time
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.run_all_scenes import run_all_scenes
from src.database import init_db, get_stats


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('reports/auto_pipeline.log', mode='a')
    ]
)
logger = logging.getLogger('auto_pipeline')

WATCH_DIR = "data/extracted"
CHECK_INTERVAL = 60  # seconds


def get_scene_count():
    return len(list(Path(WATCH_DIR).glob("*.SAFE")))


def get_scene_names():
    return {p.name for p in Path(WATCH_DIR).glob("*.SAFE")}


def main():
    logger.info("=" * 60)
    logger.info("EO QC Auto-Pipeline Started")
    logger.info(f"Watching: {WATCH_DIR}")
    logger.info(f"Check interval: {CHECK_INTERVAL}s")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    # Initialize database
    conn = init_db()
    conn.close()
    
    last_scenes = get_scene_names()
    logger.info(f"Initial scenes: {len(last_scenes)}")
    
    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            
            current_scenes = get_scene_names()
            new_scenes = current_scenes - last_scenes
            removed_scenes = last_scenes - current_scenes
            
            if removed_scenes:
                logger.info(f"Removed scenes: {len(removed_scenes)}")
            
            if new_scenes:
                logger.info(f"🔔 {len(new_scenes)} new scene(s) detected:")
                for s in sorted(new_scenes):
                    logger.info(f"  - {s}")
                
                try:
                    logger.info("Running pipeline...")
                    run_all_scenes()
                    
                    # Show stats
                    conn = init_db()
                    stats = get_stats(conn)
                    conn.close()
                    
                    logger.info(f"Pipeline complete: {stats['accepted']}/{stats['total']} accepted")
                    
                except Exception as e:
                    logger.error(f"Pipeline failed: {e}", exc_info=True)
                
                last_scenes = current_scenes
            else:
                logger.debug(f"No new scenes. Waiting...")
                
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
    finally:
        logger.info("Auto-pipeline shutdown complete")


if __name__ == "__main__":
    main()