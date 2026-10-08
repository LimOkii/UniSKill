from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def call_openai_compatible(prompt: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    config = load_config(config_path)
    url = config["api_url"]
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.get("temperature", 0),
    }
    if "max_tokens" in config:
        payload["max_tokens"] = config["max_tokens"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.get("timeout", 120)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        if body:
            raise urllib.error.HTTPError(
                exc.url,
                exc.code,
                f"{exc.reason}; response={body[:1000]}",
                exc.headers,
                None,
            ) from exc
        raise
    content = data["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("critic API returned empty message.content")
    return content


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config = {}
    with Path(config_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = _parse_value(value.strip())

    # Environment variables override the template without committing credentials.
    for key, variable in (
        ("api_key", "UNISKILL_CRITIC_API_KEY"),
        ("api_url", "UNISKILL_CRITIC_API_URL"),
        ("model", "UNISKILL_CRITIC_MODEL"),
    ):
        if os.environ.get(variable):
            config[key] = os.environ[variable]

    for key in ("api_url", "api_key", "model"):
        if not config.get(key):
            raise ValueError(f"critic config missing required key: {key}")
    return config


def _parse_value(value: str):
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
