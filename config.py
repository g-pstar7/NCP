import json
import os
from datetime import datetime
from typing import List

CONFIG_FILE = "active_pairs.json"

def load_active_pairs() -> List[str]:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                pairs = data.get("pairs", [])
                if pairs:
                    return [p.strip().upper() for p in pairs]
        except Exception as e:
            print(f"[Config] Error loading pairs: {e}")
    # Fallback
    return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'NIGHTUSDT', '1000PEPEUSDT']

def save_active_pairs(pairs: List[str]):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({
                "pairs": pairs,
                "updated_at": datetime.now().isoformat()
            }, f, indent=2)
        print(f"[Config] Saved {len(pairs)} active pairs")
    except Exception as e:
        print(f"[Config] Error saving pairs: {e}")
