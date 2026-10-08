"""Expose the selected WebShop options to the Actor prompt."""
from __future__ import annotations

from typing import Any


def selected_options_from_env(env: Any) -> dict[str, str]:
    base_env = getattr(env, "unwrapped", env)
    server = getattr(base_env, "server", None)
    session_id = getattr(base_env, "session", None)
    sessions = getattr(server, "user_sessions", {}) if server is not None else {}
    session = sessions.get(session_id, {}) if session_id is not None else {}
    return {str(key): str(value) for key, value in dict(session.get("options", {})).items()}


def format_selected_options(options: dict[str, str] | None) -> str:
    if not options:
        return ""
    values = "; ".join(f"{key}={value}" for key, value in sorted(options.items()))
    return f"Current selected options: {values}."


class WebshopStateWorker:
    """Thin worker wrapper that adds selected options to each info dict."""

    def __init__(self, seed, env_kwargs):
        from agent_system.environments.env_package.webshop.envs import WebshopWorker

        worker_class = getattr(WebshopWorker, "_original_worker", WebshopWorker)
        self._worker = worker_class(seed, env_kwargs)

    def _add_selected_options(self, info):
        result = dict(info or {})
        result["selected_options"] = selected_options_from_env(self._worker.env)
        return result

    def step(self, action):
        obs, reward, done, info = self._worker.step(action)
        return obs, reward, done, self._add_selected_options(info)

    def reset(self, idx):
        obs, info = self._worker.reset(idx)
        return obs, self._add_selected_options(info)

    def render(self, mode_for_render):
        return self._worker.render(mode_for_render)

    def get_available_actions(self):
        return self._worker.get_available_actions()

    def get_goals(self):
        return self._worker.get_goals()

    def close(self):
        return self._worker.close()


def install_webshop_state_worker() -> None:
    from agent_system.environments.env_package.webshop import envs as env_module

    current = env_module.WebshopWorker
    if current is WebshopStateWorker:
        return
    WebshopStateWorker._original_worker = current
    env_module.WebshopWorker = WebshopStateWorker
