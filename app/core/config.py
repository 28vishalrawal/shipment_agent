"""Environment-based configuration. No secrets are ever hard-coded here."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- service identity (used in every structured log line) ---
    service_name: str = "shipment-delay-agent"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    build_version: str = "0.1.0"

    # --- LLM provider selection (provider-agnostic switch) ---
    llm_provider: Literal[
        "openai", "anthropic", "gemini", "azure_openai", "mock", "nemotron"
    ] = "openai"
    llm_model: str = "gpt-4o-mini"
    # OpenAI-compatible endpoint for self-hosted / gateway inference (vLLM, NIM,
    # Ollama, or a routing gateway). Must include the /v1 suffix. Leave unset to
    # talk to api.openai.com.
    llm_base_url: str = ""
    llm_api_key: str = Field(default="", repr=False)
    # Some gateways authenticate with a bespoke header (e.g. "x-api-key") rather
    # than the OpenAI SDK's default `Authorization: Bearer <key>`. When set, the
    # key is sent in this header instead.
    llm_api_key_header: str = ""
    # Set true when a gateway rejects any Authorization header it doesn't
    # recognise. Only meaningful alongside llm_api_key_header.
    llm_disable_bearer: bool = False
    # Reasoning models (Nemotron 3 and friends) emit a thinking trace. When vLLM
    # runs with --reasoning-parser it is split into a separate `reasoning_content`
    # field; without one it is prepended to message.content and breaks JSON
    # parsing. Enabled by default so structured output survives either setup.
    llm_strip_reasoning: bool = True
    # vLLM's JSON mode support varies by build; disable to fall back to
    # prompt-instructed JSON plus tolerant parsing.
    llm_use_json_mode: bool = True

    # --- Per-role model routing -------------------------------------------
    # The three LLM roles have different requirements: the ReAct agents need
    # reliable tool calling, mitigation needs the best reasoning (its narrative
    # goes to ops leadership, ~2 calls per run), and notification drafting wants
    # a small fast model (short customer-facing text, no reasoning needed).
    # Every field below falls back to the corresponding global setting when
    # blank, so leaving them unset preserves single-model behaviour exactly.
    llm_model_agents: str = ""
    llm_model_mitigation: str = ""
    llm_model_notification: str = ""
    # Only needed when a role is served by a different endpoint (e.g. drafting
    # on a local Ollama while mitigation goes to a gateway). Leave blank to
    # share the global endpoint and credentials.
    llm_base_url_agents: str = ""
    llm_base_url_mitigation: str = ""
    llm_base_url_notification: str = ""
    llm_api_key_agents: str = Field(default="", repr=False)
    llm_api_key_mitigation: str = Field(default="", repr=False)
    llm_api_key_notification: str = Field(default="", repr=False)
    llm_api_key_header_agents: str = ""
    llm_api_key_header_mitigation: str = ""
    llm_api_key_header_notification: str = ""

    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.2

    # Secrets are read from the environment only; defaults are empty.
    openai_api_key: str = Field(default="", repr=False)
    anthropic_api_key: str = Field(default="", repr=False)
    gemini_api_key: str = Field(default="", repr=False)
    azure_openai_endpoint: str = Field(default="", repr=False)
    azure_openai_api_key: str = Field(default="", repr=False)

    # --- analytics thresholds (deterministic gate parameters) ---
    support_floor: int = 200          # Gate 0: minimum orders per segment
    effect_size_min: float = 0.15     # Gate 2: |lift - 1|
    fdr_q: float = 0.05               # Gate 3: Benjamini-Hochberg q
    confound_margin: float = 0.15     # Gate 4: min lift vs every parent
    stability_var_max: float = 0.30   # Gate 5: max cross-half variation
    escalation_confidence: float = 0.75  # escalation gate threshold
    max_escalations: int = 2          # how many top findings may escalate per run

    # --- risk-scoring parameters (Lane A) ---
    shrinkage_k: int = 50             # empirical-Bayes shrinkage constant
    eta_percentile: int = 75          # revised ETA uses P75 transit
    triage_queue_cap: int = 0  # set 0 to draft for ALL at-risk orders (Goal 1)

    # --- security / reliability ---
    max_upload_mb: int = 50
    rate_limit_per_minute: int = 120
    jwt_secret: str = Field(default="change-me-in-prod", repr=False)
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # --- persistence ---
    database_url: str = "sqlite+aiosqlite:///./data/app.db"

    # --- feature flags ---
    require_human_review: bool = True
    enable_llm_notifications: bool = True

    # --- file-drop trigger ---
    # Directory watched for dropped batches. Blank disables the watcher entirely,
    # so tests, CLI runs and one-off API usage never spawn a background loop.
    trigger_inbox_dir: str = ""
    # Where a file is moved once its run finishes. Failures land in a "failed"
    # subdirectory of this path so a bad file is retained for inspection without
    # being retried forever.
    trigger_archive_dir: str = ""
    trigger_poll_s: int = 5
    # A file must report the same size across two consecutive polls before it is
    # read, so a large copy still in progress is not parsed half-written.
    trigger_stable_polls: int = 2

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()