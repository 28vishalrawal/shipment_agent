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
    llm_provider: Literal["openai", "anthropic", "gemini", "azure_openai", "mock"] = "openai"
    llm_model: str = "gpt-4o-mini"
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

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()