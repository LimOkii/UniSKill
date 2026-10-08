from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from uniskill.critic.client import DEFAULT_CONFIG_PATH, call_openai_compatible, load_config


class CriticClientTest(unittest.TestCase):
    def test_request_uses_full_api_url_and_bearer_key(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "critic.yaml"
            config_path.write_text(
                'api_url: "https://api.example.com/v1/chat/completions"\n'
                'api_key: "example-key"\n'
                'model: "example-model"\n',
                encoding="utf-8",
            )
            requests = []

            def fake_urlopen(request, timeout):
                requests.append((request, timeout))
                response = {"choices": [{"message": {"content": "accepted"}}]}
                return io.BytesIO(json.dumps(response).encode("utf-8"))

            with patch.dict(os.environ, {}, clear=True), patch(
                "urllib.request.urlopen", side_effect=fake_urlopen
            ):
                content = call_openai_compatible("Review this skill", config_path)

        self.assertEqual(content, "accepted")
        request, timeout = requests[0]
        self.assertEqual(request.full_url, "https://api.example.com/v1/chat/completions")
        self.assertEqual(dict(request.header_items())["Authorization"], "Bearer example-key")
        self.assertEqual(timeout, 120)
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "example-model")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "Review this skill"}])

    def test_api_key_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "critic.yaml"
            config_path.write_text(
                'api_url: "https://api.example.com/v1/chat/completions"\n'
                'api_key: ""\n'
                'model: "example-model"\n',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "api_key"):
                    load_config(config_path)

    def test_environment_overrides_public_template(self):
        with patch.dict(
            os.environ,
            {
                "UNISKILL_CRITIC_API_URL": "https://api.example.com/v1/chat/completions",
                "UNISKILL_CRITIC_API_KEY": "example-key",
                "UNISKILL_CRITIC_MODEL": "example-model",
            },
            clear=True,
        ):
            config = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(config["api_url"], "https://api.example.com/v1/chat/completions")
        self.assertEqual(config["api_key"], "example-key")
        self.assertEqual(config["model"], "example-model")


if __name__ == "__main__":
    unittest.main()
