"""
digest_eval_harness.py
======================
Evaluation harness for Daily AI Digest.

Why this exists
---------------
A digest that ships 4 news items and 4 "PM Insights" every day looks great in
isolation. But over 30 days, the only honest question is: are the items actually
true, recent, relevant, and actionable? This harness answers that.

Five metrics, scored 0-1 per item, aggregated daily and weekly:
  1. factual_accuracy      LLM-as-judge against fetched source URLs
  2. recency               item published within the last 24h?
  3. relevance_to_user     match against the user's stated focus profile
  4. actionability         is the "PM Insight" specific or generic?
  5. coverage              did the digest miss high-signal news that day?

Output
------
- Per-item JSON score with rationale
- Daily summary with weak-spot diagnosis
- Rolling 7-day trend so you can see if the digest is improving or drifting

How to use in production
------------------------
1. Pipe each day's digest (HTML or markdown) into `evaluate_digest()`
2. Pipe the result into your dashboard or a Slack message
3. Use `compare_to_baseline()` weekly to detect regressions

Notes
-----
- Cost: ~$0.05 per digest evaluated using Claude Sonnet
- Latency: ~30s end-to-end including web fetches
- Run as a Make.com / cron job 1 hour after the digest is generated
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Any

import anthropic  # pip install anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EVAL_MODEL = "claude-sonnet-4-5"  # judge model — keep cheaper than the generator
USER_PROFILE = {
    "role_target": "Staff Product Manager",
    "companies": ["Anthropic", "OpenAI", "Scale AI", "Cohere", "Hugging Face"],
    "focus_areas": [
        "AI agent platforms",
        "LLM evaluation",
        "agent observability",
        "RAG and orchestration",
        "AI platform strategy",
    ],
    "deprioritised": [
        "consumer AI features",
        "image generation hype",
        "generic ChatGPT news",
    ],
}

# Weight by what actually matters for Kalpana's interview prep.
# Recency and accuracy are non-negotiable — wrong or stale = useless.
# Actionability is what differentiates "news aggregator" from "PM coaching tool".
METRIC_WEIGHTS = {
    "factual_accuracy": 0.30,
    "recency": 0.20,
    "relevance_to_user": 0.20,
    "actionability": 0.20,
    "coverage": 0.10,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DigestItem:
    """One news bullet from the digest."""
    headline: str
    summary: str
    why_it_matters: str
    pm_insight: str
    source_url: str | None = None  # ideally provided; if missing, accuracy capped


@dataclass
class ItemScore:
    headline: str
    factual_accuracy: float
    recency: float
    relevance_to_user: float
    actionability: float
    rationale: dict[str, str]
    weighted_total: float = 0.0


@dataclass
class DigestEvaluation:
    date: str
    item_scores: list[ItemScore]
    coverage_score: float
    coverage_misses: list[str]
    daily_score: float
    weak_spots: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Core evaluators — each is small and individually testable
# ---------------------------------------------------------------------------

def _judge_with_claude(client: anthropic.Anthropic, prompt: str) -> dict[str, Any]:
    """Wrap a JSON-only judge call. Defensive parse so one bad response doesn't crash the run."""
    response = client.messages.create(
        model=EVAL_MODEL,
        max_tokens=600,
        system=(
            "You are an evaluator for an AI news digest. "
            "Respond ONLY with valid JSON matching the schema in the user message. "
            "No preamble, no markdown fences, no explanation outside the JSON."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # Strip accidental fences if the judge ignores instructions
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": 0.0, "rationale": f"Judge returned invalid JSON: {text[:200]}"}


def score_factual_accuracy(client: anthropic.Anthropic, item: DigestItem, source_text: str | None) -> tuple[float, str]:
    """
    LLM-as-judge against the actual source. If no source, cap at 0.5 — we cannot
    verify, and an unverifiable claim should not get full marks.
    """
    if not source_text:
        return 0.5, "No source URL provided — accuracy unverifiable, capped at 0.5"

    prompt = f"""Compare the digest claim to the source text. Score factual accuracy 0.0-1.0.

DIGEST CLAIM:
Headline: {item.headline}
Summary: {item.summary}

SOURCE TEXT (truncated):
{source_text[:3000]}

Score 1.0 if the digest claim is fully supported by the source.
Score 0.5 if partially supported or somewhat embellished.
Score 0.0 if the claim is unsupported or contradicted.

Respond as JSON: {{"score": float, "rationale": "one sentence"}}"""

    result = _judge_with_claude(client, prompt)
    return float(result.get("score", 0.0)), result.get("rationale", "")


def score_recency(item: DigestItem, source_published_at: datetime | None, digest_date: datetime) -> tuple[float, str]:
    """Pure-Python — no LLM needed. 1.0 if within 24h, 0.5 if within 72h, else 0.0."""
    if not source_published_at:
        return 0.5, "Source publish time unknown"

    age_hours = (digest_date - source_published_at).total_seconds() / 3600
    if age_hours <= 24:
        return 1.0, f"Item is {round(age_hours, 1)}h old — within freshness window"
    if age_hours <= 72:
        return 0.5, f"Item is {round(age_hours, 1)}h old — stale but acceptable"
    return 0.0, f"Item is {round(age_hours, 1)}h old — too stale for a daily digest"


def score_relevance_to_user(client: anthropic.Anthropic, item: DigestItem, profile: dict) -> tuple[float, str]:
    """How well does this item match Kalpana's stated focus?"""
    prompt = f"""Score how relevant this news item is to a user with this profile:

PROFILE:
- Targeting: {profile['role_target']} at {', '.join(profile['companies'])}
- Focus areas: {', '.join(profile['focus_areas'])}
- Deprioritised: {', '.join(profile['deprioritised'])}

NEWS ITEM:
Headline: {item.headline}
Summary: {item.summary}

Score 0.0-1.0 where:
1.0 = directly relevant to focus areas, useful for interview prep
0.5 = adjacent — interesting but not directly applicable
0.0 = in the deprioritised list or unrelated

Respond as JSON: {{"score": float, "rationale": "one sentence"}}"""

    result = _judge_with_claude(client, prompt)
    return float(result.get("score", 0.0)), result.get("rationale", "")


def score_actionability(client: anthropic.Anthropic, item: DigestItem) -> tuple[float, str]:
    """Is the PM Insight specific and useful, or generic filler?"""
    prompt = f"""Score how actionable this PM Insight is, 0.0-1.0.

PM INSIGHT: {item.pm_insight}

CRITERIA:
- 1.0: Specific, concrete, would change a Staff PM's behaviour or framing
- 0.5: Useful direction but vague (e.g. "consider strategy implications")
- 0.0: Generic platitude (e.g. "this is important for the future of AI")

Respond as JSON: {{"score": float, "rationale": "one sentence"}}"""

    result = _judge_with_claude(client, prompt)
    return float(result.get("score", 0.0)), result.get("rationale", "")


def score_coverage(client: anthropic.Anthropic, items: list[DigestItem], profile: dict) -> tuple[float, list[str]]:
    """
    Did the digest miss any significant AI news the user would have wanted to see?
    This is the hardest metric — it requires the judge to reason about what was MISSING.
    """
    headlines = [it.headline for it in items]
    prompt = f"""You are evaluating an AI news digest for completeness.

USER PROFILE: {profile['role_target']} targeting {', '.join(profile['companies'])}
USER FOCUS: {', '.join(profile['focus_areas'])}

DIGEST INCLUDED THESE ITEMS:
{chr(10).join('- ' + h for h in headlines)}

Based on what you'd reasonably expect a comprehensive AI digest to cover for this user
on a typical day, list any HIGH-SIGNAL topics that seem to be MISSING. Be conservative —
only flag a miss if it's something a Staff PM at Anthropic/OpenAI would clearly want to know.

If nothing significant is missing, return an empty list.

Respond as JSON: {{"score": float (0.0-1.0, where 1.0 = comprehensive), "misses": [list of missed topics]}}"""

    result = _judge_with_claude(client, prompt)
    return float(result.get("score", 0.7)), result.get("misses", [])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_digest(
    items: list[DigestItem],
    digest_date: datetime,
    fetch_source_text: callable | None = None,
    fetch_source_publish_time: callable | None = None,
    profile: dict = USER_PROFILE,
    api_key: str | None = None,
) -> DigestEvaluation:
    """
    Score one daily digest. Returns a structured evaluation.

    fetch_source_text and fetch_source_publish_time are pluggable so this works
    with both web_fetch and a local cache during testing.
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
    item_scores: list[ItemScore] = []

    for item in items:
        source_text = fetch_source_text(item.source_url) if (fetch_source_text and item.source_url) else None
        publish_time = fetch_source_publish_time(item.source_url) if (fetch_source_publish_time and item.source_url) else None

        acc_score, acc_reason = score_factual_accuracy(client, item, source_text)
        rec_score, rec_reason = score_recency(item, publish_time, digest_date)
        rel_score, rel_reason = score_relevance_to_user(client, item, profile)
        act_score, act_reason = score_actionability(client, item)

        weighted = (
            acc_score * METRIC_WEIGHTS["factual_accuracy"] +
            rec_score * METRIC_WEIGHTS["recency"] +
            rel_score * METRIC_WEIGHTS["relevance_to_user"] +
            act_score * METRIC_WEIGHTS["actionability"]
        ) / sum(v for k, v in METRIC_WEIGHTS.items() if k != "coverage")

        item_scores.append(ItemScore(
            headline=item.headline,
            factual_accuracy=acc_score,
            recency=rec_score,
            relevance_to_user=rel_score,
            actionability=act_score,
            rationale={
                "factual_accuracy": acc_reason,
                "recency": rec_reason,
                "relevance_to_user": rel_reason,
                "actionability": act_reason,
            },
            weighted_total=round(weighted, 3),
        ))

    coverage_score, misses = score_coverage(client, items, profile)

    # Daily score = weighted average of item scores, then blended with coverage
    avg_item = sum(s.weighted_total for s in item_scores) / max(len(item_scores), 1)
    daily_score = round(
        avg_item * (1 - METRIC_WEIGHTS["coverage"]) + coverage_score * METRIC_WEIGHTS["coverage"],
        3,
    )

    weak_spots = _diagnose_weak_spots(item_scores, coverage_score)

    return DigestEvaluation(
        date=digest_date.strftime("%Y-%m-%d"),
        item_scores=item_scores,
        coverage_score=round(coverage_score, 3),
        coverage_misses=misses,
        daily_score=daily_score,
        weak_spots=weak_spots,
    )


def _diagnose_weak_spots(item_scores: list[ItemScore], coverage_score: float) -> list[str]:
    """Surface the single most actionable thing to fix tomorrow."""
    avg = lambda key: sum(getattr(s, key) for s in item_scores) / max(len(item_scores), 1)

    weak = []
    if avg("factual_accuracy") < 0.7:
        weak.append("Factual accuracy is the priority fix — pipe source URLs into the digest generator and have it cite verbatim.")
    if avg("actionability") < 0.6:
        weak.append("PM Insights are too generic — prompt the digest generator to require ONE concrete behaviour change per insight.")
    if avg("relevance_to_user") < 0.6:
        weak.append("Relevance is drifting — refresh the user profile with the 5 companies and focus areas at the top of the prompt.")
    if avg("recency") < 0.6:
        weak.append("Stale items detected — tighten the source filter to last-24h only.")
    if coverage_score < 0.6:
        weak.append("Coverage gaps — add a second source pass before generation (e.g. 2 different aggregators).")
    if not weak:
        weak.append("No major weaknesses today — sustain current generation prompt.")
    return weak


def to_dashboard_json(evaluation: DigestEvaluation) -> str:
    """Serialise for the dashboard widget or a Slack post."""
    return json.dumps(asdict(evaluation), indent=2, default=str)


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Demo using the digest from the screenshot.
    In production, parse from the actual Gmail / Make.com payload.
    """
    digest_items = [
        DigestItem(
            headline="Anthropic Ships Claude 3.5 Haiku with Vision + Computer Use",
            summary="Claude's fastest model now includes multimodal capabilities and screen interaction APIs",
            why_it_matters="Direct competitor analysis — Anthropic is bundling capabilities most platforms sell separately",
            pm_insight="Consider unbundling vs. bundling strategy for your AI platform features",
            source_url="https://www.anthropic.com/news/claude-3-5-haiku",
        ),
        DigestItem(
            headline="Microsoft Announces AutoGen 0.4 with Native Multi-Agent Memory",
            summary="Persistent memory across agent conversations, state management, built-in orchestration patterns",
            why_it_matters="Agent orchestration framework gaining enterprise traction",
            pm_insight="Memory persistence is becoming table stakes for agent platforms",
            source_url="https://github.com/microsoft/autogen/releases",
        ),
        # ...etc
    ]

    digest_date = datetime(2026, 5, 4, tzinfo=timezone.utc)
    evaluation = evaluate_digest(items=digest_items, digest_date=digest_date)
    print(to_dashboard_json(evaluation))


if __name__ == "__main__":
    _demo()
