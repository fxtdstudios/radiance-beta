import sqlite3
import os
import json
import logging
import time

logger = logging.getLogger("radiance.core.param_memory")

class RadianceParamHistoryTracker:
    """
    ◎ Radiance Parameter History Tracker
    
    A SQL-backed parameter memory node that records parameters,
    reviews historical diffs, and saves custom configuration defaults.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "node_name": ("STRING", {"default": "RadianceHDREncoder"}),
                "parameters_json": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "trigger": ("*",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("history_summary", "parameter_diff")
    FUNCTION = "record"
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Infrastructure"

    def __init__(self):
        # Establish database directory inside package root
        self.db_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.db_dir, "radiance_history.db")
        self._init_db()

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS param_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    node_name TEXT,
                    parameters_json TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning(f"[Param Memory] Failed to initialize SQLite database: {exc}")

    def record(self, node_name: str, parameters_json: str, trigger=None):
        timestamp = time.time()
        
        # 1. Parse current parameters
        try:
            current_params = json.loads(parameters_json)
        except Exception:
            current_params = {}
            logger.warning("[Param Memory] Invalid parameters JSON string.")
            
        # 2. Query previous entry for diffing
        prev_params = {}
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT parameters_json FROM param_history WHERE node_name = ? ORDER BY id DESC LIMIT 1",
                (node_name,)
            )
            row = cursor.fetchone()
            if row:
                prev_params = json.loads(row[0])
                
            # Insert current entry
            cursor.execute(
                "INSERT INTO param_history (timestamp, node_name, parameters_json) VALUES (?, ?, ?)",
                (timestamp, node_name, json.dumps(current_params))
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning(f"[Param Memory] Database read/write failed: {exc}")

        # 3. Calculate parameter diff
        diff_lines = []
        all_keys = set(prev_params.keys()).union(current_params.keys())
        
        for k in sorted(all_keys):
            prev_v = prev_params.get(k)
            curr_v = current_params.get(k)
            if prev_v != curr_v:
                diff_lines.append(f"  • {k}: {prev_v} ➔ {curr_v}")
                
        diff_str = "\n".join(diff_lines) if diff_lines else "No parameter changes detected since last run."
        
        # 4. Generate structured summary JSON
        summary = {
            "node_name": node_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "parameters": current_params,
            "changes_detected": len(diff_lines) > 0
        }
        
        logger.info(f"[Param Memory] Recorded configuration for {node_name} (Changes: {len(diff_lines)}).")
        return (json.dumps(summary, indent=2), diff_str)
