"""Single typed settings surface (the documented "knobs").

Everything tunable for cost-vs-freshness lives here. Load order:
    defaults  <  .env file  <  RTI_*  environment variables
Nested fields use a double-underscore delimiter, e.g.
    RTI_SAMPLING__BASE_INTERVAL_S=900
    RTI_FEATURES__DEVICE=cuda
Secrets (API tokens, DSNs, session cookies) come from env / .env only and are
never committed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["auto", "cuda", "cpu"]
StorageKind = Literal["sqlite", "postgres"]
EmbeddingBackend = Literal["deterministic", "real"]
RankingKey = Literal["health", "opportunity"]


class AppConfig(BaseModel):
    env: Literal["dev", "prod", "test"] = "dev"
    seed: int = 1729  # global seed for reproducibility
    data_dir: str = "data"
    # The niche used for whitespace / opportunity detection (config-driven).
    niche: str = "food/restaurant"
    # Zero-shot labels that define membership in the niche (scene/caption match).
    niche_keywords: list[str] = Field(
        default_factory=lambda: [
            "food",
            "restaurant",
            "recipe",
            "cooking",
            "cafe",
            "dish",
            "kitchen",
            "menu",
        ]
    )


class StorageConfig(BaseModel):
    backend: StorageKind = "sqlite"
    sqlite_path: str = "data/rti.db"
    # Postgres DSN, e.g. postgresql://user:pass@localhost:5432/rti
    postgres_dsn: SecretStr | None = None
    pool_min_size: int = 2
    pool_max_size: int = 10
    # Bulk-write batch size for COPY / executemany.
    write_batch_size: int = 1000


class CollectionConfig(BaseModel):
    # Active adapters (config-driven). 'sample' is always available offline.
    enabled_adapters: list[str] = Field(
        default_factory=lambda: ["sample"]
    )
    # Global token bucket shared across the polite scheduler.
    requests_per_minute: float = 30.0
    bucket_capacity: int = 10
    # Per-adapter hard daily request cap (budget).
    daily_request_cap: int = 5000
    # Randomized human-scale delay added per request (seconds, uniform).
    min_delay_s: float = 0.8
    max_delay_s: float = 2.5
    # Exponential backoff on transient errors.
    backoff_base_s: float = 2.0
    backoff_max_s: float = 300.0
    max_retries: int = 5
    # Owned-session adapter: HALT immediately on a login challenge/checkpoint.
    owned_session_halt_on_challenge: bool = True
    # Loop cadence. A collect pass over N tracked reels takes N * (polite delay),
    # so polling far faster than that just starves the rebuild tick (they share one
    # storage connection) and spams "max instances reached".
    collect_interval_s: int = 300
    rebuild_interval_s: int = 600
    # Graph API public discovery: hashtags (Facebook-Login tokens only) and
    # Business-Discovery seed accounts (public Business/Creator usernames).
    graph_hashtags: list[str] = Field(
        default_factory=lambda: ["food", "foodreels", "recipe", "restaurant", "cooking"]
    )
    graph_seed_usernames: list[str] = Field(default_factory=list)


class SamplingConfig(BaseModel):
    """Adaptive engagement re-sampling — the single biggest efficiency lever.

    next_interval = clamp(
        base * 2 ** (age_hours / age_halflife_h) / (1 + accel_weight * |accel_norm|),
        min_interval, max_interval)
    Reels are tiered hot -> warm -> cold -> dead; dead reels are RETIRED.
    """

    base_interval_s: float = 900.0      # 15 min baseline
    min_interval_s: float = 120.0       # never poll a single reel faster than 2 min
    max_interval_s: float = 86_400.0    # cold reels at most daily
    age_halflife_h: float = 12.0        # interval doubles every 12h of age
    accel_weight: float = 4.0           # acceleration shortens the interval
    # Tier thresholds on age.
    hot_max_age_h: float = 6.0
    warm_max_age_h: float = 48.0
    cold_max_age_h: float = 240.0       # 10 days
    # Retirement: no engagement growth over this window => retire (stop polling).
    retire_after_h: float = 168.0       # 7 days
    retire_min_rel_growth: float = 0.01  # <1% plays growth across window => dead
    # Budget-aware degradation: when remaining budget fraction is below this,
    # only sample HOT reels.
    budget_low_fraction: float = 0.15


class FeatureConfig(BaseModel):
    device: Device = "auto"
    batch_size: int = 16
    # VRAM ceiling; extractors keep at most one model resident and check headroom.
    vram_cap_mb: int = 3200            # leave headroom on a 4 GB card
    embedding_backend: EmbeddingBackend = "deterministic"
    # Real-model identifiers (used when embedding_backend == "real").
    clip_model: str = "clip-ViT-B-32"
    caption_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 512           # fixed wire dim; real backends are projected/padded
    # Cheap gate thresholds (tiered compute): only survivors get expensive models.
    gate_min_plays: int = 0            # relevance/quality floor
    phash_hamming_dup: int = 4         # <= this Hamming distance => duplicate
    # Color extraction.
    color_k: int = 5
    color_max_side: int = 256          # downscale before k-means
    # OCR runs only on a sampled keyframe at this probability (cost control).
    enable_ocr: bool = True
    ocr_sample_rate: float = 0.25
    # Audio.
    enable_audio_fingerprint: bool = True


class TrendConfig(BaseModel):
    # Joint-embedding fusion weights (image / audio / caption).
    weight_image: float = 0.45
    weight_audio: float = 0.30
    weight_caption: float = 0.25
    # UMAP (optional) -> HDBSCAN.
    umap_n_components: int = 10
    umap_n_neighbors: int = 15
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    # "eom" (excess of mass) can emit one huge blob on diverse real data; "leaf"
    # yields finer, more homogeneous trends. Configurable per dataset.
    hdbscan_cluster_selection_method: Literal["eom", "leaf"] = "eom"
    # Time bucket for adoption curve N(t).
    bucket_s: int = 3600               # 1 hour adoption buckets
    # Audio-fingerprint cluster Hamming threshold.
    audio_fp_hamming: int = 6


class ModelConfig(BaseModel):
    # Survival: "death" = adoption rate drops > drop_pct from peak for K buckets.
    death_drop_pct: float = 0.6
    death_consecutive_buckets: int = 3
    # Persistence horizon N (days) for "persists >= N more days".
    persistence_horizon_days: float = 3.0
    # Peak forecasts beyond this many days ahead are reported as unknown (None)
    # rather than as a meaningless extrapolated date.
    max_forecast_days: float = 60.0
    # Calibration method against history.
    calibration: Literal["isotonic", "platt", "none"] = "isotonic"
    # Hawkes power-law kernel prior (theta) and cutoff (seconds).
    hawkes_theta: float = 0.2
    hawkes_kernel_c_s: float = 60.0
    # Mean secondary adoptions per event prior (n*) when followers unknown.
    hawkes_n_star: float = 8.0
    bootstrap_iters: int = 200         # CI via bootstrap


class EmergingConfig(BaseModel):
    # Burst / change-point sensitivity.
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.5
    bocpd_hazard: float = 1.0 / 168.0  # expected run length ~ 1 week of buckets
    kleinberg_gamma: float = 1.0
    # Early classifier window (hours of life used for the early features).
    early_window_h: float = 24.0
    # P(crosses virality threshold) — threshold on peak plays.
    virality_plays_threshold: int = 500_000
    # Opportunity / whitespace.
    opportunity_min_global_accel: float = 0.0  # must be accelerating globally
    opportunity_niche_saturation_cap: float = 0.25  # under-adopted in niche


class ReportConfig(BaseModel):
    default_ranking: RankingKey = "health"
    # Metric that selects the single most-famous exemplar (top_reel).
    top_reel_metric: Literal["peak_plays", "engagement_rate"] = "peak_plays"
    export_dir: str = "data/exports"
    # Best-effort external link resolution for songs (cached, may be skipped).
    resolve_external_links: bool = True


VLMProvider = Literal["anthropic", "openai", "local"]


class AnalyserConfig(BaseModel):
    """Per-reel video analyser (V2). Every threshold here is config-driven.

    The VLM is chosen at runtime via `vlm_provider` (or a per-call override), so
    you can toggle between a local model and either hosted API without code
    changes. No provider is imported by engine code — only by its adapter.
    """

    # --- provider toggle ---------------------------------------------------
    vlm_provider: VLMProvider = "anthropic"

    # Anthropic (Claude). Pricing $/1M tokens for cost estimation.
    anthropic_model: str = "claude-opus-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    anthropic_max_tokens: int = 2048
    anthropic_price_in_per_mtok: float = 5.00
    anthropic_price_out_per_mtok: float = 25.00

    # OpenAI (ChatGPT). Model must be vision-capable.
    openai_model: str = "gpt-4o"
    openai_max_tokens: int = 2048
    openai_price_in_per_mtok: float = 2.50
    openai_price_out_per_mtok: float = 10.00

    # Local VLM. A 4 GB card realistically fits only a small quantised model;
    # see README for the quality tradeoff.
    local_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    local_device: Device = "auto"
    local_load_4bit: bool = True
    local_max_new_tokens: int = 1024

    # --- keyframe montage --------------------------------------------------
    max_frames: int = 12
    frame_grid_seconds: float = 1.5
    montage_max_width: int = 1280
    montage_cols: int = 4

    # --- queue + cost controls --------------------------------------------
    vlm_retries: int = 1              # one retry on parse failure, then fail
    daily_analysis_cap: int = 500
    min_engagement_to_analyse: int = 0
    request_timeout_s: float = 120.0


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RTI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    collection: CollectionConfig = Field(default_factory=CollectionConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    trends: TrendConfig = Field(default_factory=TrendConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    emerging: EmergingConfig = Field(default_factory=EmergingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    analyser: AnalyserConfig = Field(default_factory=AnalyserConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached Settings (used by tests that mutate the environment)."""
    get_settings.cache_clear()
