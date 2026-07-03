"""Synthetic, fully-offline dataset so the whole pipeline runs with zero network.

Design goals (so the statistics are meaningful, not noise):
  * a handful of distinct TRENDS, each with its own palette, audio/song, caption
    style, scene, format, and an ADOPTION PROFILE (emerging / peaking / decaying
    / steady) that governs WHEN its member reels are posted;
  * per-reel engagement that is HEAVY-TAILED (lognormal peak) and follows a
    saturating growth curve from its post time -> honest, irreducible variance;
  * trend-correlated images so colour k-means recovers each trend's palette and
    clustering has real signal to find.

The dataset exposes ground-truth engagement at any simulated time `t`, so the
adaptive resampler builds genuinely sparse-over-time series during the sim.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
from PIL import Image

from reels_trend_intel.storage.rows import Audio, EngagementSample, Reel

SIM_START = datetime(2026, 6, 20, 0, 0, tzinfo=UTC)
HORIZON_H = 264.0  # 11 days -> ends ~2026-07-01

RGB = tuple[int, int, int]


def _hex(rgb: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


@dataclass
class TrendSpec:
    key: str
    ttype: str
    label: str
    palette: list[RGB]
    palette_props: list[float]
    audio_id: str
    song: str
    artist: str
    is_original: bool
    scene_label: str
    niche: bool
    caption_lead: str
    hashtags: list[str]
    duration_s: float
    cut_rate: float
    has_face: bool
    birth_h: float
    adoption: str          # emerging|peaking|decaying|steady
    n_members: int
    pop_mu: float          # lognormal mean (log peak plays)
    pop_sigma: float
    growth_k: float        # per-reel saturation speed (1/hour)
    save_mult: float = 1.0


def _trend_specs() -> list[TrendSpec]:
    return [
        TrendSpec(
            key="smash_burger", ttype="visual", label="Smash-burger close-up / warm low-key",
            palette=[(38, 22, 14), (152, 73, 38), (214, 142, 64), (90, 52, 30), (232, 210, 170)],
            palette_props=[0.40, 0.25, 0.15, 0.12, 0.08],
            audio_id="aud_sizzle01", song="Sizzle Theme", artist="KitchenCore", is_original=True,
            scene_label="food-closeup", niche=True,
            caption_lead="The smash burger that broke the internet",
            hashtags=["#smashburger", "#foodreels", "#burger", "#reels"],
            duration_s=11.0, cut_rate=1.4, has_face=False,
            birth_h=6.0, adoption="emerging", n_members=34, pop_mu=12.4, pop_sigma=1.1,
            growth_k=0.18, save_mult=1.6,
        ),
        TrendSpec(
            key="matcha_pour", ttype="audio", label="Matcha slow-pour / high-key green",
            palette=[(225, 232, 210), (140, 178, 96), (96, 140, 70), (238, 240, 232), (60, 96, 52)],
            palette_props=[0.38, 0.24, 0.16, 0.14, 0.08],
            audio_id="aud_lofi77", song="Soft Morning", artist="Velour", is_original=False,
            scene_label="table", niche=True,
            caption_lead="POV: your calm matcha morning",
            hashtags=["#matcha", "#aesthetic", "#morningroutine", "#cafe"],
            duration_s=9.0, cut_rate=0.6, has_face=False,
            birth_h=2.0, adoption="peaking", n_members=40, pop_mu=12.9, pop_sigma=1.2,
            growth_k=0.26, save_mult=1.2,
        ),
        TrendSpec(
            key="get_ready", ttype="format", label="Get-ready-with-me jump-cuts / studio",
            palette=[(232, 222, 220), (200, 150, 150), (150, 110, 120), (250, 245, 244), (90, 70, 80)],
            palette_props=[0.42, 0.22, 0.16, 0.12, 0.08],
            audio_id="aud_pop2099", song="Glow Up", artist="NEONA", is_original=False,
            scene_label="studio", niche=False,
            caption_lead="GRWM but make it 9 seconds",
            hashtags=["#grwm", "#getreadywithme", "#fyp", "#routine"],
            duration_s=14.0, cut_rate=2.6, has_face=True,
            birth_h=1.0, adoption="decaying", n_members=28, pop_mu=12.1, pop_sigma=1.0,
            growth_k=0.33, save_mult=0.8,
        ),
        TrendSpec(
            key="street_taco", ttype="visual", label="Street-taco night / neon outdoor",
            palette=[(20, 18, 30), (210, 70, 60), (240, 190, 70), (40, 120, 110), (235, 225, 200)],
            palette_props=[0.36, 0.24, 0.18, 0.14, 0.08],
            audio_id="aud_cumbia12", song="Noche Cumbia", artist="Los Faroles", is_original=False,
            scene_label="outdoor", niche=True,
            caption_lead="Street tacos hit different at midnight",
            hashtags=["#tacos", "#streetfood", "#foodie", "#latenight"],
            duration_s=12.0, cut_rate=1.8, has_face=False,
            birth_h=10.0, adoption="emerging", n_members=26, pop_mu=12.0, pop_sigma=1.3,
            growth_k=0.16, save_mult=1.4,
        ),
        TrendSpec(
            key="text_recipe", ttype="caption", label="Text-on-solid recipe hack",
            palette=[(245, 240, 230), (30, 30, 30), (220, 90, 60), (200, 200, 190), (120, 120, 110)],
            palette_props=[0.50, 0.20, 0.14, 0.10, 0.06],
            audio_id="aud_typebeat3", song="Type Beat 3", artist="prod.mix", is_original=True,
            scene_label="text-on-solid", niche=True,
            caption_lead="Save this 3-ingredient dinner",
            hashtags=["#recipe", "#easyrecipe", "#cooking", "#mealprep"],
            duration_s=18.0, cut_rate=0.9, has_face=False,
            birth_h=20.0, adoption="steady", n_members=22, pop_mu=11.6, pop_sigma=0.9,
            growth_k=0.20, save_mult=2.0,
        ),
        TrendSpec(
            key="cliff_dive", ttype="audio", label="Cliff-dive travel / cool outdoor",
            palette=[(30, 80, 120), (120, 180, 210), (220, 230, 235), (20, 50, 80), (180, 200, 170)],
            palette_props=[0.34, 0.26, 0.18, 0.14, 0.08],
            audio_id="aud_epic555", song="Skyfall Drop", artist="Atlas Rise", is_original=False,
            scene_label="outdoor", niche=False,
            caption_lead="Would you make this jump?",
            hashtags=["#travel", "#adventure", "#summer", "#fyp"],
            duration_s=15.0, cut_rate=1.2, has_face=True,
            birth_h=30.0, adoption="emerging", n_members=20, pop_mu=12.6, pop_sigma=1.4,
            growth_k=0.15, save_mult=0.9,
        ),
    ]


@dataclass
class _ReelPlan:
    reel: Reel
    trend_key: str
    posted_h: float
    peak_plays: float
    growth_k: float
    save_mult: float
    rng_seed: int


@dataclass
class SyntheticDataset:
    seed: int = 1729
    sim_start: datetime = SIM_START
    horizon_h: float = HORIZON_H
    specs: list[TrendSpec] = field(default_factory=_trend_specs)
    plans: dict[str, _ReelPlan] = field(default_factory=dict)
    audios: dict[str, Audio] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        for spec in self.specs:
            self.audios[spec.audio_id] = Audio.from_id(
                spec.audio_id, song_title=spec.song, artist=spec.artist,
                is_original_audio=spec.is_original, usage_count=0,
            )
            times = self._sample_post_times(spec, rng)
            self.audios[spec.audio_id].usage_count = len(times)
            for i, ph in enumerate(times):
                shortcode = f"{spec.key[:3].upper()}{i:03d}{int(ph)}"
                posted_at = self.sim_start + timedelta(hours=float(ph))
                reel = Reel.from_shortcode(
                    shortcode,
                    author_handle=f"{spec.key}_{i % 9}",
                    author_url=f"https://www.instagram.com/{spec.key}_{i % 9}/",
                    posted_at=posted_at, audio_id=spec.audio_id,
                    caption=self._caption(spec, i, rng), source="sample",
                    media_url=f"sample://{shortcode}",
                )
                peak = float(np.exp(rng.normal(spec.pop_mu, spec.pop_sigma)))
                self.plans[shortcode] = _ReelPlan(
                    reel=reel, trend_key=spec.key, posted_h=float(ph), peak_plays=peak,
                    growth_k=spec.growth_k, save_mult=spec.save_mult,
                    rng_seed=int(rng.integers(0, 2**31)),
                )

    # --- adoption profile (when reels are posted) -------------------------
    def _adoption_rate(self, spec: TrendSpec, t_h: float) -> float:
        if t_h < spec.birth_h:
            return 0.0
        x = t_h - spec.birth_h
        if spec.adoption == "emerging":
            return float(np.exp(0.018 * x)) * np.exp(-x / 220.0)
        if spec.adoption == "decaying":
            return float(np.exp(-0.03 * x))
        if spec.adoption == "peaking":
            mode = 60.0
            return float((x / mode) * np.exp(1.0 - x / mode))
        return 1.0  # steady

    def _sample_post_times(self, spec: TrendSpec, rng: np.random.Generator) -> list[float]:
        grid = np.linspace(spec.birth_h, self.horizon_h, 400)
        rates = np.array([self._adoption_rate(spec, float(t)) for t in grid])
        rates = rates / (rates.sum() + 1e-9)
        picks = rng.choice(grid, size=spec.n_members, replace=True, p=rates)
        picks = np.clip(picks + rng.normal(0, 1.0, size=spec.n_members), spec.birth_h,
                        self.horizon_h)
        return sorted(float(p) for p in picks)

    def _caption(self, spec: TrendSpec, i: int, rng: np.random.Generator) -> str:
        emojis = ["🔥", "😍", "🤤", "✨", "🙌", "👀"]
        em = "".join(rng.choice(emojis, size=int(rng.integers(1, 4)), replace=True))
        cta = " save this & follow for more" if rng.random() < 0.5 else ""
        tags = " ".join(spec.hashtags)
        return f"{em} {spec.caption_lead}{cta} {tags}"

    # --- ground-truth engagement at simulated time ------------------------
    def engagement_at(self, reel_id: str, t: datetime) -> EngagementSample | None:
        plan = self.plans.get(reel_id)
        if plan is None:
            return None
        tau_h = (t - plan.reel.posted_at).total_seconds() / 3600.0
        if tau_h < 0:
            return None
        frac = 1.0 - float(np.exp(-plan.growth_k * tau_h))  # saturating growth
        plays = plan.peak_plays * frac
        r = np.random.default_rng(plan.rng_seed)
        likes = plays * (0.035 + 0.01 * r.random())
        comments = plays * (0.004 + 0.003 * r.random())
        shares = plays * (0.006 + 0.004 * r.random())
        saves = plays * (0.010 + 0.006 * r.random()) * plan.save_mult
        return EngagementSample(
            reel_id=reel_id, sampled_at=t, plays=int(plays), likes=int(likes),
            comments=int(comments), shares=int(shares), saves=int(saves),
        )

    # --- synthetic media (trend-correlated palette, per-reel-unique layout) -
    def render_image(self, reel_id: str, size: int = 96) -> bytes | None:
        plan = self.plans.get(reel_id)
        if plan is None:
            return None
        spec = next(s for s in self.specs if s.key == plan.trend_key)
        r = np.random.default_rng(plan.rng_seed ^ 0x5151)
        img = np.zeros((size, size, 3), dtype=np.float32)
        # Per-reel random MOSAIC: same palette + ~same colour proportions as the
        # trend, but a unique spatial layout so perceptual hashes differ across
        # reels (no false dedupe) while colour features stay trend-consistent.
        palette = np.array(spec.palette, dtype=np.float32)
        props = np.array(spec.palette_props, dtype=np.float64)
        props = props / props.sum()
        n_rows = int(r.integers(3, 7))
        n_cols = int(r.integers(3, 7))
        ys = np.linspace(0, size, n_rows + 1).astype(int)
        xs = np.linspace(0, size, n_cols + 1).astype(int)
        for i in range(n_rows):
            for j in range(n_cols):
                idx = int(r.choice(len(palette), p=props))
                jitter = r.normal(0, 9, size=3)
                color = np.clip(palette[idx] + jitter, 0, 255)
                img[ys[i]:ys[i + 1], xs[j]:xs[j + 1], :] = color
        img += r.normal(0, 6, size=img.shape)
        arr = np.clip(img, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")
        return buf.getvalue()

    # --- helpers for the sample adapter -----------------------------------
    def reels_sorted(self) -> list[Reel]:
        return [p.reel for p in sorted(self.plans.values(), key=lambda p: p.posted_h)]

    def audio_for(self, reel_id: str) -> Audio | None:
        plan = self.plans.get(reel_id)
        return self.audios.get(plan.reel.audio_id) if plan and plan.reel.audio_id else None

    def ground_truth_trend(self, reel_id: str) -> str | None:
        plan = self.plans.get(reel_id)
        return plan.trend_key if plan else None

    def format_hints(self, reel_id: str) -> dict[str, object]:
        plan = self.plans.get(reel_id)
        if plan is None:
            return {}
        spec = next(s for s in self.specs if s.key == plan.trend_key)
        return {
            "duration_s": spec.duration_s,
            "cut_rate": spec.cut_rate,
            "has_face": spec.has_face,
        }
