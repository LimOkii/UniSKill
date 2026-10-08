from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

from scripts.load_config import ALLOWED_KEYS, read_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts" / "config.yaml"
LOADER = ROOT / "scripts" / "load_config.py"


class TrainingConfigTest(unittest.TestCase):
    def test_shared_yaml_has_all_fields_and_no_committed_api_key(self):
        config = read_config(CONFIG)
        self.assertEqual(set(config), ALLOWED_KEYS)
        self.assertEqual(config["UNISKILL_CRITIC_API_KEY"], "")

    def test_environment_variable_takes_precedence(self):
        env = os.environ.copy()
        env["MODEL_PATH"] = "already-selected-model"
        env.pop("UNISKILL_CRITIC_MODEL", None)
        result = subprocess.run(
            [sys.executable, str(LOADER), str(CONFIG)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertNotIn("export MODEL_PATH=", result.stdout)
        self.assertIn("export UNISKILL_CRITIC_MODEL=", result.stdout)

    def test_both_launchers_read_the_same_yaml(self):
        for name in ("train_alfworld.sh", "train_webshop.sh"):
            with self.subTest(name=name):
                script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn('CONFIG_FILE="${UNISKILL_CONFIG_FILE:-$SCRIPT_DIR/config.yaml}"', script)
                self.assertIn('python3 "$SCRIPT_DIR/load_config.py" "$CONFIG_FILE"', script)


if __name__ == "__main__":
    unittest.main()
