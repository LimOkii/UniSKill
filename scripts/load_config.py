"""Read the flat string-valued scripts/config.yaml into shell-safe exports."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import shlex
import sys


ALLOWED_KEYS = frozenset(
    {
        "MODEL_PATH",
        "EMBEDDING_MODEL_PATH",
        "UNISKILL_CRITIC_API_URL",
        "UNISKILL_CRITIC_API_KEY",
        "UNISKILL_CRITIC_MODEL",
        "JAVA_HOME",
    }
)


def read_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = line.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        if not separator or key not in ALLOWED_KEYS or key in values:
            raise ValueError(f"{path}:{line_number}: invalid or duplicate configuration key")
        if raw_value.startswith(("'", '"')):
            value = ast.literal_eval(raw_value)
            if not isinstance(value, str):
                raise ValueError(f"{path}:{line_number}: expected a quoted string")
        else:
            value = raw_value
        values[key] = value
    return values


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: load_config.py PATH_TO_CONFIG_YAML", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    try:
        values = read_config(path)
    except (OSError, ValueError, SyntaxError) as error:
        print(f"Could not read training config: {error}", file=sys.stderr)
        return 2
    for key in sorted(ALLOWED_KEYS):
        if os.environ.get(key):
            continue
        print(f"export {key}={shlex.quote(values.get(key, ''))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
