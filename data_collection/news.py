"""Tushare news downloads, shared raw pools, company matching and daily selection."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from common import (
    CODE_DIR, Company, DEFAULT_ALIAS_CSV, RequestRunner, add_common_args, add_pipeline_args,
    clean_content, date_windows, digest, event_days, make_matchers, normalize_text, now,
    parse_optional_datetime, positive_int, read_json, run_cli, safe_error, safe_name,
    spec_from_args, tushare_client, validate_args, write_csv, write_json,
)
from collections import defaultdict
from typing import Any, Iterable
from urllib.parse import urlparse, urlunparse


# Csi300 alias matcher

COMPANY_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "集团股份有限公司",
    "集团有限公司",
    "控股股份有限公司",
    "控股有限公司",
    "集团",
    "股份",
    "公司",
)

AMBIGUOUS_SHORT_ALIASES = {
    "中国",
    "中信",
    "中金",
    "中航",
    "中车",
    "中铁",
    "中电",
    "国电",
    "国投",
    "招商",
    "华能",
    "华电",
    "华润",
    "华夏",
    "东方",
    "光大",
    "兴业",
    "广发",
    "长城",
    "长江",
    "太平洋",
}


@dataclass(frozen=True)
class AliasRecord:
    ticker: str
    alias: str
    alias_type: str
    confidence: float
    valid_from: str = ""
    valid_to: str = ""

    @property
    def compact_alias(self) -> str:
        return compact_alias(self.alias)


@dataclass(frozen=True)
class AliasHit:
    ticker: str
    alias: str
    alias_type: str
    location: str
    score: float
    ambiguous: bool


@dataclass(frozen=True)
class AliasMatcher:
    records: tuple[AliasRecord, ...]
    alias_to_records: dict[str, tuple[AliasRecord, ...]]
    pattern: re.Pattern[str] | None


def compact_alias(value: str) -> str:
    return re.sub(r"[-\s·・.。．,，、()（）\\/]+", "", normalize_text(value)).lower()


def strip_company_suffix(name: str) -> str:
    stripped = normalize_text(name)
    changed = True
    while changed:
        changed = False
        for suffix in COMPANY_SUFFIXES:
            if stripped.endswith(suffix) and len(stripped) > len(suffix) + 1:
                stripped = stripped[: -len(suffix)].strip()
                changed = True
    return stripped


def parse_alias_values(raw: str) -> list[str]:
    values: list[str] = []
    for item in re.split(r"[|;；,，、/]+", raw or ""):
        item = normalize_text(item)
        if item:
            values.append(item)
    return values


def infer_alias_type(alias: str, *, company_name: str, ticker: str, raw_code: str) -> str:
    normalized = normalize_text(alias)
    if normalized == ticker or normalized.lower() == raw_code.lower():
        return "ticker"
    if normalized == normalize_text(company_name):
        return "legal_or_stock_name"
    if re.search(r"[A-Za-z]", normalized):
        return "english_or_code_alias"
    return "short_name"


def alias_confidence(alias: str, alias_type: str) -> float:
    compact = compact_alias(alias)
    if alias_type == "ticker":
        return 0.95
    if alias_type == "legal_or_stock_name":
        return 0.95
    if alias_type == "manual_alias":
        return 0.90
    if len(compact) >= 4:
        return 0.82
    if len(compact) == 3:
        return 0.70
    return 0.45


def is_usable_alias(alias: str) -> bool:
    compact = compact_alias(alias)
    if compact.isdecimal() or len(compact) < 2:
        return False
    if compact in {compact_alias(item) for item in AMBIGUOUS_SHORT_ALIASES}:
        return False
    return True


def derive_company_aliases(company: Company) -> list[AliasRecord]:
    raw_values: list[tuple[str, str]] = []
    raw_values.append((company.company_name, "legal_or_stock_name"))
    raw_values.append((strip_company_suffix(company.company_name), "short_name"))
    if company.query_name:
        raw_values.append((company.query_name, "query_name"))
    for alias in parse_alias_values(company.aliases):
        raw_values.append((alias, "manual_alias"))
    raw_values.append((company.ticker, "ticker"))
    raw_values.append((company.raw_code, "ticker"))

    compact_name = compact_alias(company.company_name)
    if compact_name.endswith("a") and len(compact_name) >= 3:
        raw_values.append((normalize_text(company.company_name)[:-1], "short_name"))

    records: list[AliasRecord] = []
    seen: set[str] = set()
    for alias, alias_type in raw_values:
        alias = normalize_text(alias)
        if not alias or not is_usable_alias(alias):
            continue
        key = compact_alias(alias)
        if key in seen:
            continue
        seen.add(key)
        inferred_type = alias_type or infer_alias_type(
            alias,
            company_name=company.company_name,
            ticker=company.ticker,
            raw_code=company.raw_code,
        )
        records.append(
            AliasRecord(
                ticker=company.ticker,
                alias=alias,
                alias_type=inferred_type,
                confidence=alias_confidence(alias, inferred_type),
                valid_from="",
                valid_to="",
            )
        )
    return records


def load_alias_records(path: Path) -> list[AliasRecord]:
    if not path.exists():
        return []
    records: list[AliasRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ticker = (row.get("ticker") or "").strip()
            alias = normalize_text(row.get("alias"))
            if not ticker or not alias or not is_usable_alias(alias):
                continue
            alias_type = (row.get("alias_type") or "manual_alias").strip()
            try:
                confidence = float(row.get("confidence") or alias_confidence(alias, alias_type))
            except ValueError:
                confidence = alias_confidence(alias, alias_type)
            records.append(
                AliasRecord(
                    ticker=ticker,
                    alias=alias,
                    alias_type=alias_type,
                    confidence=confidence,
                    valid_from=(row.get("valid_from") or "").strip(),
                    valid_to=(row.get("valid_to") or "").strip(),
                )
            )
    return records


def build_alias_matcher(records: list[AliasRecord]) -> AliasMatcher:
    alias_map: dict[str, list[AliasRecord]] = {}
    for record in records:
        key = record.compact_alias
        if not key:
            continue
        alias_map.setdefault(key, []).append(record)
    ordered_aliases = sorted(alias_map, key=lambda item: (-len(item), item))
    pattern = re.compile("|".join(re.escape(item) for item in ordered_aliases)) if ordered_aliases else None
    return AliasMatcher(
        records=tuple(records),
        alias_to_records={key: tuple(value) for key, value in alias_map.items()},
        pattern=pattern,
    )


def build_matcher_for_companies(companies: list[Company], alias_csv: Path | None = None) -> AliasMatcher:
    records: list[AliasRecord] = []
    for company in companies:
        records.extend(derive_company_aliases(company))
    if alias_csv and alias_csv.exists():
        records.extend(load_alias_records(alias_csv))
    deduped: dict[tuple[str, str], AliasRecord] = {}
    for record in records:
        key = (record.ticker, record.compact_alias)
        existing = deduped.get(key)
        if existing is None or record.confidence > existing.confidence:
            deduped[key] = record
    return build_alias_matcher(list(deduped.values()))


def alias_valid_on(record: AliasRecord, match_date: date | None) -> bool:
    if match_date is None:
        return True
    if record.valid_from:
        parsed = parse_optional_datetime(record.valid_from)
        if parsed is not None and match_date < parsed.date():
            return False
    if record.valid_to:
        parsed = parse_optional_datetime(record.valid_to)
        if parsed is not None and match_date > parsed.date():
            return False
    return True


def match_aliases(
    *,
    title: str,
    content: str,
    matcher: AliasMatcher,
    match_date: date | None = None,
    allowed_ticker: str | None = None,
) -> list[AliasHit]:
    if matcher.pattern is None:
        return []
    hits: list[AliasHit] = []
    seen: set[tuple[str, str, str]] = set()
    haystacks = [
        ("title", compact_alias(title)),
        ("content", compact_alias(content)),
    ]
    for location, haystack in haystacks:
        if not haystack:
            continue
        for match in matcher.pattern.finditer(haystack):
            key = match.group(0)
            records = matcher.alias_to_records.get(key, ())
            ambiguous = len({record.ticker for record in records if alias_valid_on(record, match_date)}) > 1
            for record in records:
                if allowed_ticker and record.ticker != allowed_ticker:
                    continue
                # Latin names/codes require word boundaries; numeric-only aliases are excluded.
                if re.fullmatch(r"[A-Za-z0-9 .&_-]+", record.alias):
                    raw_text = title if location == "title" else content
                    boundary = r"(?<![A-Za-z0-9])" + re.escape(record.alias) + r"(?![A-Za-z0-9])"
                    if re.search(boundary, raw_text, re.IGNORECASE) is None:
                        continue
                if not alias_valid_on(record, match_date):
                    continue
                hit_key = (record.ticker, record.compact_alias, location)
                if hit_key in seen:
                    continue
                seen.add(hit_key)
                base = 42.0 if location == "title" else 20.0
                score = base * record.confidence
                if record.alias_type == "ticker":
                    score += 4.0
                if ambiguous:
                    score *= 0.55 if location == "title" else 0.35
                hits.append(
                    AliasHit(
                        ticker=record.ticker,
                        alias=record.alias,
                        alias_type=record.alias_type,
                        location=location,
                        score=round(score, 3),
                        ambiguous=ambiguous,
                    )
                )
    hits.sort(key=lambda item: (item.score, len(compact_alias(item.alias))), reverse=True)
    return hits


def match_tickers(
    *,
    title: str,
    content: str,
    matcher: AliasMatcher,
    match_date: date | None = None,
    min_score: float = 18.0,
) -> list[str]:
    best: dict[str, float] = {}
    for hit in match_aliases(title=title, content=content, matcher=matcher, match_date=match_date):
        best[hit.ticker] = max(best.get(hit.ticker, 0.0), hit.score)
    return sorted(ticker for ticker, score in best.items() if score >= min_score)


# News processing

NEWS_PROCESSING_HELP = """Daily news-pool scoring, cross-source deduplication and representative selection.
Reuses the original project export_daily_news_csv.py rules; no dataset is embedded.
"""

DIRECT_SOURCE_TYPES = {
    "sina_finance_company_news",
    "10jqka_public_stockpage_news",
    "aastocks_hk_stock_news",
    "hkexnews_announcement",
    "tdnet_jpx_disclosure",
}


SOURCE_PRIORITY = {
    "hkexnews_announcement": 36.0,
    "aastocks_hk_stock_news": 34.0,
    "sina_finance_company_news": 32.0,
    "10jqka_public_stockpage_news": 28.0,
    "eastmoney_stock_news_search": 18.0,
    "google_news_rss": 18.0,
    "google_news_rss_company_query": 18.0,
    "google_news_rss_event_query": 14.0,
    "gdelt_doc_api_company_query": 16.0,
    "etnet_hk_stock_news": 16.0,
    "tdnet_jpx_disclosure": 36.0,
    "minkabu_stock_news": 18.0,
    "yahoo_finance_jp_news": 18.0,
}


SOURCE_NAME_BOOST = {
    "路透": 6.0,
    "reuters": 6.0,
    "bloomberg": 6.0,
    "彭博": 6.0,
    "财联社": 5.0,
    "证券时报": 5.0,
    "澎湃": 4.0,
    "界面": 4.0,
    "每日经济新闻": 4.0,
    "yahoo finance": 3.0,
    "新浪财经": 4.0,
    "东方财富": 3.0,
}


STOPWORDS = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "集团股份有限公司",
    "集团有限公司",
    "集团",
    "公司",
)


@dataclass
class ArticleCandidate:
    raw_path: Path
    payload: dict[str, Any]
    market: str
    event_id: str
    ticker: str
    company_name: str
    date: str
    time: str
    source: str
    source_type: str
    title: str
    content: str
    url: str
    article_id: str
    alignment_score: float
    alignment_reasons: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    source_priority: float = 0.0
    representative_score: float = 0.0
    duplicate_count: int = 1
    duplicate_sources: list[str] = field(default_factory=list)
    raw_candidate_count: int = 1

    @property
    def article_text_length(self) -> int:
        return len(self.content)

    def csv_row(self) -> dict[str, str]:
        return {
            "time": self.time,
            "source": self.source,
            "title": self.title,
            "content": self.content,
        }

    def aggregate_row(self) -> dict[str, Any]:
        return {
            "market": self.market,
            "event_id": self.event_id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "date": self.date,
            "time": self.time,
            "source": self.source,
            "source_type": self.source_type,
            "title": self.title,
            "content": self.content,
            "detail_url": self.url,
            "article_id": self.article_id,
            "representative_score": round(self.representative_score, 2),
            "alignment_score": round(self.alignment_score, 2),
            "quality_score": round(self.quality_score, 2),
            "source_priority": round(self.source_priority, 2),
            "duplicate_count": self.duplicate_count,
            "raw_candidate_count": self.raw_candidate_count,
        }


def cjk_char_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def looks_like_utf8_mojibake(text: str) -> bool:
    if not text:
        return False
    latin1_high = sum(1 for char in text if 0x80 <= ord(char) <= 0xFF)
    if latin1_high < 4:
        return False
    if cjk_char_count(text) > 0:
        return False
    return latin1_high / max(len(text), 1) >= 0.15


def repair_utf8_mojibake(text: str) -> str:
    repaired = text
    for _ in range(2):
        if not looks_like_utf8_mojibake(repaired):
            break
        raw_bytes = bytes(ord(char) for char in repaired if ord(char) <= 0xFF)
        candidate = raw_bytes.decode("utf-8", errors="ignore").strip()
        if not candidate:
            break
        if cjk_char_count(candidate) <= cjk_char_count(repaired):
            break
        repaired = candidate
    return repaired


def normalize_article_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    text = html.unescape(str(value))
    text = repair_utf8_mojibake(text)
    text = text.replace("\r", "\n").replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", normalize_article_text(value)).lower()


def first_non_empty(payload: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = normalize_article_text(payload.get(key))
        if value:
            return value
    return ""


def infer_market(news_root: Path) -> str:
    try:
        return news_root.parents[1].name
    except IndexError:
        return ""


def infer_event_id(news_root: Path) -> str:
    try:
        return news_root.parent.name
    except IndexError:
        return ""


def candidate_name_variants(company_name: str, query: str, ticker: str) -> list[str]:
    candidates: set[str] = set()
    for value in (company_name,):
        cleaned = normalize_article_text(value)
        if not cleaned:
            continue
        candidates.add(cleaned)
        stripped = cleaned
        for suffix in STOPWORDS:
            stripped = stripped.replace(suffix, "")
        stripped = stripped.strip()
        if len(compact_text(stripped)) >= 2:
            candidates.add(stripped)
    if ticker:
        candidates.add(ticker)
    return sorted(candidate for candidate in candidates if compact_text(candidate))


def normalize_url(url: str) -> str:
    cleaned = normalize_article_text(url)
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        return cleaned
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        query="",
        fragment="",
    )
    return urlunparse(normalized).rstrip("/")


def stable_hash(*parts: str) -> str:
    joined = "||".join(part.strip() for part in parts if part and part.strip())
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def source_priority_score(source_type: str, source_name: str) -> float:
    score = SOURCE_PRIORITY.get(source_type, 10.0)
    lowered = normalize_article_text(source_name).lower()
    for keyword, boost in SOURCE_NAME_BOOST.items():
        if keyword in lowered:
            score += boost
    return score


def compute_alignment(
    *,
    payload: dict[str, Any],
    title: str,
    content: str,
    source_type: str,
    alias_matcher: AliasMatcher | None = None,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    ticker = normalize_article_text(payload.get("ticker"))
    direct_source = source_type in DIRECT_SOURCE_TYPES
    title_compact = compact_text(title)
    content_compact = compact_text(content)

    if direct_source:
        score += 45.0
        reasons.append("direct_company_source")

    if ticker:
        ticker_lower = ticker.lower()
        if ticker_lower in title_compact:
            score += 20.0
            reasons.append("ticker_in_title")
        elif ticker_lower in content_compact:
            score += 10.0
            reasons.append("ticker_in_content")

    if alias_matcher is not None:
        alias_hits = match_aliases(
            title=title,
            content=content,
            matcher=alias_matcher,
            allowed_ticker=ticker or None,
        )
        if alias_hits:
            best_hit = alias_hits[0]
            score += best_hit.score
            reasons.append(
                "alias_{location}:{alias}:{alias_type}:score={score}".format(
                    location=best_hit.location,
                    alias=best_hit.alias,
                    alias_type=best_hit.alias_type,
                    score=best_hit.score,
                )
            )
            if best_hit.ambiguous:
                reasons.append("alias_ambiguous_downweighted")

    variants = candidate_name_variants(
        normalize_article_text(payload.get("company_name")),
        normalize_article_text(payload.get("query")),
        ticker,
    )
    for name in variants:
        compact_name = compact_text(name)
        if not compact_name or compact_name == ticker.lower():
            continue
        if compact_name in title_compact:
            score += 25.0
            reasons.append(f"name_in_title:{name}")
            break
    else:
        for name in variants:
            compact_name = compact_text(name)
            if compact_name and compact_name != ticker.lower() and compact_name in content_compact:
                score += 12.0
                reasons.append(f"name_in_content:{name}")
                break

    if not direct_source and score == 0:
        reasons.append("no_alignment_signal")
    return max(0.0, min(100.0, score)), reasons


def compute_quality(payload: dict[str, Any], title: str, content: str, source: str, url: str, time_text: str) -> float:
    score = 0.0
    status = normalize_article_text(payload.get("article_extraction_status")).lower()
    status_score = {
        "success": 30.0,
        "skip_success": 24.0,
        "summary_only": 16.0,
        "title_only": 10.0,
        "not_requested": 6.0,
        "detail_failed": 4.0,
        "fetch_failed": 0.0,
        "empty": -6.0,
    }
    score += status_score.get(status, 6.0 if content else 0.0)
    score += min(len(content) / 220.0, 24.0)
    if title:
        score += 6.0
    if source:
        score += 3.0
    if url:
        score += 4.0
    if time_text:
        score += 2.0
    duplicate_hint = normalize_article_text(payload.get("raw_article", {}).get("mode")) if isinstance(payload.get("raw_article"), dict) else ""
    if duplicate_hint == "markdown":
        score += 2.0
    return score


def representative_score(candidate: ArticleCandidate) -> float:
    corroboration_bonus = min(max(candidate.duplicate_count - 1, 0), 3) * 3.0
    return candidate.alignment_score + candidate.quality_score + candidate.source_priority + corroboration_bonus


def content_from_payload(payload: dict[str, Any], title: str) -> str:
    article_text = first_non_empty(payload, ["article_text"])
    if article_text:
        return article_text

    summary = first_non_empty(payload, ["summary", "description"])
    if not summary:
        return ""

    compact_title = compact_text(title)
    compact_summary = compact_text(summary)
    if not compact_summary:
        return ""
    if compact_title and (
        compact_summary == compact_title
        or compact_summary in compact_title
        or compact_title in compact_summary
    ):
        return ""
    if len(summary) < 80:
        return ""
    return summary


def build_candidate(
    news_root: Path,
    date_dir: Path,
    json_path: Path,
    bad_json_files: list[dict[str, str]],
    alias_matcher: AliasMatcher | None,
) -> ArticleCandidate | None:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        bad_json_files.append(
            {
                "ticker": date_dir.parent.name,
                "date": date_dir.name,
                "path": str(json_path),
                "error": str(exc),
            }
        )
        return None
    time_text = first_non_empty(
        payload,
        ["article_publish_time_text", "pub_date_raw", "headline_date_text", "publish_date"],
    )
    source = first_non_empty(payload, ["source_name", "source_type"])
    source_type = normalize_article_text(payload.get("source_type"))
    title = first_non_empty(payload, ["article_title", "title", "headline_title"])
    content = content_from_payload(payload, title)
    url = first_non_empty(payload, ["final_url", "resolved_url", "detail_url", "link", "headline_url"])
    article_id = first_non_empty(payload, ["article_id", "guid"])
    if not article_id:
        article_id = stable_hash(str(json_path), title, url, time_text)

    market = first_non_empty(payload, ["market"]) or infer_market(news_root)
    event_id = first_non_empty(payload, ["event_id"]) or infer_event_id(news_root)
    ticker = first_non_empty(payload, ["ticker"]) or date_dir.parent.name
    company_name = first_non_empty(payload, ["company_name"])

    alignment_score, alignment_reasons = compute_alignment(
        payload=payload,
        title=title,
        content=content,
        source_type=source_type,
        alias_matcher=alias_matcher,
    )
    quality_score = compute_quality(payload, title, content, source, url, time_text)
    priority_score = source_priority_score(source_type, source)

    candidate = ArticleCandidate(
        raw_path=json_path,
        payload=payload,
        market=market,
        event_id=event_id,
        ticker=ticker,
        company_name=company_name,
        date=date_dir.name,
        time=time_text,
        source=source,
        source_type=source_type,
        title=title,
        content=content,
        url=url,
        article_id=article_id,
        alignment_score=alignment_score,
        alignment_reasons=alignment_reasons,
        quality_score=quality_score,
        source_priority=priority_score,
    )
    candidate.representative_score = representative_score(candidate)
    return candidate


def dedupe_key(candidate: ArticleCandidate) -> str:
    normalized_url = normalize_url(candidate.url)
    if normalized_url:
        return f"url::{normalized_url}"
    title_key = compact_text(candidate.title)
    if title_key:
        return f"title::{title_key}::{candidate.date}"
    content_key = compact_text(candidate.content[:400])
    if content_key:
        return f"content::{content_key}::{candidate.date}"
    return f"id::{candidate.article_id}"


def select_best_duplicate(candidates: list[ArticleCandidate]) -> ArticleCandidate:
    best = max(
        candidates,
        key=lambda item: (
            item.quality_score,
            item.source_priority,
            item.article_text_length,
            item.time,
            item.raw_path.name,
        ),
    )
    best.duplicate_count = len(candidates)
    best.duplicate_sources = sorted(
        {
            candidate.source
            for candidate in candidates
            if candidate.source
        }
    )
    best.representative_score = representative_score(best)
    return best


def build_daily_pool(candidates: list[ArticleCandidate], min_alignment_score: float) -> tuple[list[ArticleCandidate], list[ArticleCandidate]]:
    raw_count = len(candidates)
    dedupe_groups: dict[str, list[ArticleCandidate]] = defaultdict(list)
    for candidate in candidates:
        dedupe_groups[dedupe_key(candidate)].append(candidate)

    deduped_candidates = [select_best_duplicate(group) for group in dedupe_groups.values()]
    filtered_pool = [
        candidate
        for candidate in deduped_candidates
        if candidate.source_type in DIRECT_SOURCE_TYPES or candidate.alignment_score >= min_alignment_score
    ]
    pool = filtered_pool or deduped_candidates
    for candidate in pool:
        candidate.raw_candidate_count = raw_count
        candidate.representative_score = representative_score(candidate)
    pool.sort(
        key=lambda item: (
            item.representative_score,
            item.alignment_score,
            item.quality_score,
            item.article_text_length,
            item.time,
        ),
        reverse=True,
    )
    return pool, deduped_candidates


def build_representative_payload(
    *,
    representative: ArticleCandidate,
    raw_candidates: list[ArticleCandidate],
    pool_candidates: list[ArticleCandidate],
) -> dict[str, Any]:
    payload = dict(representative.payload)
    payload.update(
        {
            "market": representative.market,
            "event_id": representative.event_id,
            "ticker": representative.ticker,
            "company_name": representative.company_name,
            "query": normalize_article_text(payload.get("query")) or representative.company_name,
            "source_name": representative.source,
            "title": representative.title,
            "article_title": representative.title,
            "article_text": representative.content,
            "summary": normalize_article_text(payload.get("summary")),
            "publish_date": representative.date,
            "representative_time": representative.time,
            "representative_source": representative.source,
            "representative_title": representative.title,
            "representative_content": representative.content,
            "representative_url": representative.url,
            "representative_article_id": representative.article_id,
            "representative_score": round(representative.representative_score, 2),
            "alignment_score": round(representative.alignment_score, 2),
            "alignment_reasons": representative.alignment_reasons,
            "quality_score": round(representative.quality_score, 2),
            "source_priority": round(representative.source_priority, 2),
            "duplicate_count": representative.duplicate_count,
            "duplicate_sources": representative.duplicate_sources,
            "raw_candidate_count": len(raw_candidates),
            "pool_candidate_count": len(pool_candidates),
            "pool_article_ids": [candidate.article_id for candidate in pool_candidates],
            "pool_sources": [candidate.source for candidate in pool_candidates],
            "pool_json_files": [candidate.raw_path.name for candidate in pool_candidates],
        }
    )
    return payload


# News quality

NEWS_QUALITY_HELP = """Content/event relevance and candidate-risk rules from the project analysis view.
These are heuristic quality flags, not human-reviewed ground truth labels.
"""

MARKET_KEYWORDS = (
    "a股",
    "沪深",
    "港股",
    "股市",
    "市场",
    "券商",
    "策略",
    "投资",
    "板块",
    "行业",
    "指数",
    "资金",
    "风险偏好",
    "避险",
    "股价",
    "期货",
    "债券",
    "黄金",
    "原油",
    "油价",
    "汇率",
)


EVENT_KEYWORDS = {
    "covid19_first_wave": (
        "疫情",
        "新冠",
        "冠状病毒",
        "肺炎",
        "武汉",
        "湖北",
        "防疫",
        "抗疫",
        "口罩",
        "火神山",
        "雷神山",
        "复工",
        "复产",
        "隔离",
        "捐赠",
        "医疗物资",
    ),
    "us_israel_iran_conflict_2025": (
        "以色列",
        "伊朗",
        "以伊",
        "中东",
        "地缘",
        "冲突",
        "空袭",
        "战争",
        "霍尔木兹",
        "原油",
        "石油",
        "油价",
        "黄金",
        "避险",
        "军工",
        "大宗商品",
        "风险偏好",
    ),
    "russia_ukraine_conflict_2022": (
        "俄罗斯",
        "乌克兰",
        "俄乌",
        "俄乌冲突",
        "制裁",
        "地缘",
        "冲突",
        "战争",
        "能源",
        "原油",
        "油价",
        "天然气",
        "粮食",
        "避险",
        "黄金",
        "军工",
        "大宗商品",
        "风险偏好",
    ),
    "israel_palestine_conflict_2023": (
        "以色列",
        "巴勒斯坦",
        "巴以",
        "哈马斯",
        "加沙",
        "中东",
        "地缘",
        "冲突",
        "战争",
        "能源",
        "原油",
        "油价",
        "黄金",
        "避险",
        "军工",
        "大宗商品",
        "风险偏好",
    ),
}


def text_has_any(compact_text: str, keywords: Iterable[str]) -> bool:
    return any(compact_alias(keyword) in compact_text for keyword in keywords if keyword)


def classify_content(title: str, body: str, summary: str) -> str:
    content_len = len(normalize_text(body or summary))
    if not title and content_len == 0:
        return "missing_content"
    if content_len >= 200:
        return "full_text"
    if content_len >= 20:
        return "short_text"
    if title:
        return "title_only"
    return "missing_content"


def classify_event_relevance(event_id: str, source_group: str, title: str, body: str, content_scan_chars: int) -> str:
    compact_text = compact_alias(f"{title} {body[:content_scan_chars]}")
    if text_has_any(compact_text, EVENT_KEYWORDS.get(event_id, ())):
        return "event_direct"
    if text_has_any(compact_text, MARKET_KEYWORDS):
        return "market_impact_context"
    if source_group == "supplement_announcements":
        return "non_event_corporate_disclosure"
    return "event_unrelated"


def classify_risk(
    *,
    source_group: str,
    content_availability: str,
    company_match_strength: str,
    company_match_reason: str,
    event_relevance: str,
) -> str:
    if company_match_strength == "weak_or_ambiguous":
        return "exclude_candidate"
    if company_match_reason == "source_name_collision":
        return "exclude_candidate"
    if content_availability in {"missing_content", "title_only"}:
        return "needs_attention"
    if company_match_strength == "sector_or_market_context":
        return "needs_attention"
    if event_relevance in {"event_unrelated", "non_event_corporate_disclosure"}:
        return "needs_attention"
    if source_group != "main_tushare" and content_availability != "full_text":
        return "needs_attention"
    return "ok"


# Download tushare news

DOWNLOAD_TUSHARE_NEWS_HELP = """Fetch Tushare news flashes or long-form news without company lists or alias tables.

Dependencies: python -m pip install requests pandas
Token: TUSHARE_NEWS_TOKEN or TUSHARE_TOKEN; news access requires separate permission.
Example: python news.py download --api news --sources sina cls --start-date 2020-01-01 --end-date 2020-01-02 --resume
For long-form news, use --api major_news and the provider's Chinese source names.
Write one CSV per source/day with time/source/title/content columns. Preserve API
time strings without inferring time zones or assigning companies and events.
Recursively split windows at the default limits of 1,500 flashes or 400 long-form
articles. If a single second still reaches the limit, fail rather than mark a
potentially truncated result complete.
Documentation: https://tushare.pro/document/2?doc_id=143
               https://tushare.pro/document/2?doc_id=195
"""

LIMITS = {"news": 1500, "major_news": 400}
NEWS_CSV_FIELDS = ["time", "source", "title", "content"]


def fetch_complete(pro, runner, api, source, start, end, limit):
    def fetch():
        kwargs = {"src": source, "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
                  "end_date": end.strftime("%Y-%m-%d %H:%M:%S")}
        if api == "major_news":
            kwargs["fields"] = "title,content,pub_time,src"
        return pro.query(api, **kwargs)

    frame = runner(fetch)
    if len(frame) >= limit:
        seconds = int((end - start).total_seconds())
        if seconds < 1:
            raise RuntimeError("单秒内结果仍达到行数上限，无法确认完整性")
        midpoint = start + timedelta(seconds=seconds // 2)
        return (fetch_complete(pro, runner, api, source, start, midpoint, limit)
                + fetch_complete(pro, runner, api, source, midpoint + timedelta(seconds=1), end, limit))
    rows = []
    for item in frame.fillna("").to_dict(orient="records"):
        rows.append({"time": str(item.get("datetime" if api == "news" else "pub_time", "")),
                     "source": str(item.get("src") or source),
                     "title": str(item.get("title", "")), "content": str(item.get("content", ""))})
    return rows


def download_news_main():
    parser = argparse.ArgumentParser(description=DOWNLOAD_TUSHARE_NEWS_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "tushare_news")
    parser.set_defaults(request_gap_seconds=10.0)
    parser.add_argument("--api", choices=LIMITS, default="news")
    parser.add_argument("--sources", nargs="+", required=True, help="news 用 sina/cls；major_news 用中文来源名")
    parser.add_argument("--row-limit", type=positive_int, help="API 限量变化时调整拆分阈值")
    args = parser.parse_args()
    validate_args(parser, args)
    limit = args.row_limit or LIMITS[args.api]
    pro = tushare_client(args.timeout_seconds, news=True)
    runner = RequestRunner(args.request_gap_seconds, args.retries)
    failures = 0
    try:
        for source in dict.fromkeys(args.sources):
            for day, _ in date_windows(args.start_date, args.end_date, 1):
                path = args.output_dir / args.api / f"limit_{limit}" / safe_name(source) / f"{day}.csv"
                if args.resume and path.is_file():
                    print(f"跳过 {source} {day}")
                    continue
                try:
                    start = datetime.combine(day, time.min)
                    end = datetime.combine(day, time(23, 59, 59))
                    rows = fetch_complete(pro, runner, args.api, source, start, end, limit)
                    unique = {tuple(row[field] for field in NEWS_CSV_FIELDS): row for row in rows}
                    rows = sorted(unique.values(), key=lambda row: (row["time"], row["title"], row["content"]))
                    write_csv(path, NEWS_CSV_FIELDS, rows)
                    print(f"{source} {day}: {len(rows)} 行" + ("（空结果，请核对权限和历史覆盖）" if not rows else ""))
                except Exception as exc:
                    failures += 1
                    print(f"{source} {day}: {safe_error(exc)}", file=sys.stderr)
    finally:
        pro.close()
    return 1 if failures else 0


# Build tushare raw pool

BUILD_TUSHARE_RAW_POOL_HELP = """Build a shared Tushare raw news pool with original fields and resumable checkpoints.

Cache layout: work-dir/tushare/{news,major_news}/{source}/{start}_{end}.json.gz.
Fetch each source once per day and recursively split windows that reach the API
row limit. Record empty responses as successful queries with an empty status;
they do not establish that the entire market had no news.
"""

def pool_tasks(settings, events):
    first, last = settings["raw_pool_start"], settings["raw_pool_end"]
    if bool(first) != bool(last):
        raise ValueError("raw_pool_start 和 raw_pool_end 必须同时填写")
    if first:
        day, final = date.fromisoformat(first), date.fromisoformat(last)
        if day > final:
            raise ValueError("原始池日期顺序错误")
        days = []
        while day <= final:
            days.append(day)
            day += timedelta(days=1)
    else:
        days = event_days(events)
    for api, sources in (("news", settings["news_sources"]), ("major_news", settings["major_sources"])):
        for source in dict.fromkeys(sources):
            for day in days:
                yield {"endpoint": api, "source": source,
                       "start_dt": str(day) + " 00:00:00", "end_dt": str(day) + " 23:59:59",
                       "row_limit": settings["news_row_limits"][api], "schema_version": 1}


def cache_path(root, task):
    start = datetime.fromisoformat(task["start_dt"]).strftime("%Y%m%d_%H%M%S")
    end = datetime.fromisoformat(task["end_dt"]).strftime("%Y%m%d_%H%M%S")
    return root / "tushare" / task["endpoint"] / safe_name(task["source"]) / f"{start}_{end}.json.gz"


def validate_cache(path, task):
    payload = read_json(path)
    if any(payload.get(key) != value for key, value in task.items()):
        raise ValueError("缓存窗口/来源/接口或版本不匹配")
    records = payload.get("records")
    if not isinstance(records, list) or payload.get("row_count") != len(records):
        raise ValueError("缓存记录数校验失败")
    if digest(records) != payload.get("records_sha256"):
        raise ValueError("缓存内容哈希校验失败")
    return payload


def fetch_records(pro, runner, task, start, end):
    def fetch():
        kwargs = {"src": task["source"], "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
                  "end_date": end.strftime("%Y-%m-%d %H:%M:%S")}
        if task["endpoint"] == "major_news":
            kwargs["fields"] = "title,content,pub_time,src"
        return pro.query(task["endpoint"], **kwargs)
    frame = runner(fetch)
    if len(frame) >= task["row_limit"]:
        seconds = int((end - start).total_seconds())
        if seconds < 1:
            raise RuntimeError("单秒仍达到接口上限，无法确认完整性")
        midpoint = start + timedelta(seconds=seconds // 2)
        return (fetch_records(pro, runner, task, start, midpoint)
                + fetch_records(pro, runner, task, midpoint + timedelta(seconds=1), end))
    records = frame.fillna("").to_dict(orient="records")
    timestamp = "datetime" if task["endpoint"] == "news" else "pub_time"
    for record in records:
        if timestamp not in record or "content" not in record:
            raise ValueError("非空新闻响应缺少时间或正文字段")
        published = datetime.fromisoformat(str(record[timestamp]))
        if published.tzinfo is not None:
            raise ValueError("新闻时间带时区但请求窗口未指定时区，请核对数据源")
        if not start <= published <= end:
            raise ValueError("API 返回窗口外新闻，拒绝标记该片段为完整")
    return records


def acquire_pool(root, settings, events, *, from_cache=False, refresh=False):
    if from_cache and refresh:
        raise ValueError("from-cache 与 refresh 不能同时使用")
    runner = RequestRunner(settings["request_gap_seconds"], settings["retries"])
    results, paths = [], []
    pro = None
    checkpoint = root / "metadata" / "tushare_raw_pool_checkpoint.json"
    try:
        for task in pool_tasks(settings, events):
            path = cache_path(root, task)
            state = dict(task, cache=str(path.relative_to(root)))
            try:
                cached = None
                if path.is_file() and not refresh:
                    try:
                        cached = validate_cache(path, task)
                    except (ValueError, OSError, EOFError):
                        if from_cache:
                            raise
                if cached is None:
                    if from_cache:
                        raise FileNotFoundError(f"原始池缺少完整缓存: {path.relative_to(root)}")
                    if pro is None:
                        pro = tushare_client(settings["timeout_seconds"], news=True)
                    records = fetch_records(pro, runner, task,
                        datetime.fromisoformat(task["start_dt"]), datetime.fromisoformat(task["end_dt"]))
                    cached = dict(task, row_count=len(records), records=records,
                                  records_sha256=digest(records), retrieved_at=now())
                    write_json(path, cached, compressed=True)
                    origin = "downloaded"
                else:
                    origin = "cache"
                paths.append(path)
                state.update(status="empty" if not cached["records"] else "complete",
                             row_count=cached["row_count"], records_sha256=cached["records_sha256"], origin=origin)
                print(f"{task['endpoint']} {task['source']} {task['start_dt'][:10]}: {state['row_count']} ({origin})", flush=True)
            except Exception as exc:
                state.update(status="error", error=safe_error(exc))
                print(state["error"], file=sys.stderr, flush=True)
            results.append(state)
            write_json(checkpoint, {"updated_at": now(), "tasks": results,
                "complete_tasks": sum(item["status"] != "error" for item in results),
                "failed_tasks": sum(item["status"] == "error" for item in results)})
        if any(item["status"] == "error" for item in results):
            raise RuntimeError("原始新闻池存在失败/缺失片段，见检查点；使用相同配置重跑可继续")
        if not paths:
            raise ValueError("未配置任何新闻来源")
    finally:
        if pro is not None:
            pro.close()
    return paths


def build_pool_main():
    parser = argparse.ArgumentParser(description=BUILD_TUSHARE_RAW_POOL_HELP)
    add_pipeline_args(parser)
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root, settings, events, _ = spec_from_args(args, require_companies=False)
    acquire_pool(root, settings, events, from_cache=args.from_cache, refresh=args.refresh)
    return 0


# Build company news library

BUILD_COMPANY_NEWS_LIBRARY_HELP = """Raw news pool -> company news library -> deduplicated daily pools -> daily representatives.

Reuse the project's alias confidence, daily-pool scoring, source priority, and
representative selection rules. Exclude numeric-only aliases, apply word
boundaries to Latin codes, and do not assign articles on ambiguous aliases alone.
Store each build under a configuration, code, and raw-pool fingerprint to prevent
old records from entering a build after aliases change.
"""

def build_library(root, settings, events, companies, paths):
    if not companies:
        raise ValueError("构建公司库需要公司名单")
    wanted_days = set(event_days(events))
    available_days = {parse_optional_datetime(read_json(path)["start_dt"]).date() for path in paths}
    if not wanted_days <= available_days:
        raise ValueError("原始池日期范围未覆盖全部事件观察窗口")
    config = {"companies": [asdict(company) for company in companies], "events": events, "settings": settings}
    spec_hash = digest(config)
    code_hash = digest({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in CODE_DIR.glob("*.py")})
    pool_hash = digest([(str(path.relative_to(root)), read_json(path)["records_sha256"]) for path in paths])
    build_id = digest([spec_hash, code_hash, pool_hash])[:24]
    build = root / "builds" / build_id
    if (build / "manifest.json").is_file():
        write_json(root / "metadata" / "current_company_build.json", {"build_id": build_id})
        print(f"复用已完成公司库 {build_id}")
        return build
    by_market = defaultdict(list)
    for company in companies:
        by_market[company.market].append(company)
    matchers = make_matchers(companies, settings)
    lookups = {market: {company.ticker: company for company in group} for market, group in by_market.items()}
    aliases = []
    for market, matcher in matchers.items():
        for record in matcher.records:
            aliases.append({"market": market, **asdict(record)})
    write_csv(build / "metadata" / "company_aliases.csv",
        ["market", "ticker", "alias", "alias_type", "confidence", "valid_from", "valid_to"], aliases)
    write_csv(build / "metadata" / "companies.csv", list(asdict(companies[0])), [asdict(c) for c in companies])
    date_dirs = set()
    raw_count = rejected = 0
    for path in paths:
        payload = read_json(path)
        for item in payload["records"]:
            raw_count += 1
            api, source = payload["endpoint"], payload["source"]
            title = clean_content(item.get("title"))
            content = clean_content(item.get("content"))
            timestamp = normalize_text(item.get("datetime" if api == "news" else "pub_time"))
            published = parse_optional_datetime(timestamp)
            if published is None:
                raise ValueError("原始池记录没有发布时间")
            day = published.date().isoformat()
            article_id = digest([day, title, content])[:24]
            for market, matcher in matchers.items():
                hits = match_aliases(title=title, content=content, matcher=matcher, match_date=published.date())
                accepted = {}
                for hit in hits:
                    if hit.ambiguous or hit.score < settings["min_match_score"]:
                        rejected += 1
                        continue
                    company = lookups[market][hit.ticker]
                    # Avoid mistaking a news provider's name for a company name.
                    if hit.location == "content" and normalize_text(hit.alias).casefold() == source.casefold():
                        rejected += 1
                        continue
                    if hit.ticker not in accepted or hit.score > accepted[hit.ticker].score:
                        accepted[hit.ticker] = hit
                for ticker, hit in accepted.items():
                    company = lookups[market][ticker]
                    target = build / "raw" / market / "company_news" / "news" / ticker / day
                    date_dirs.add((market, ticker, day))
                    document = {"market": market, "ticker": ticker, "company_name": company.company_name,
                        "article_id": article_id, "title": title, "article_title": title,
                        "article_text": content, "summary": "", "source_name": str(item.get("src") or source),
                        "source_type": "tushare_news" if api == "news" else "tushare_major_news",
                        "publish_date": day, "article_publish_time_text": timestamp, "pub_date_raw": timestamp,
                        "article_extraction_status": "success" if content else "title_only",
                        "query": company.query_name or company.company_name,
                        "match_alias": hit.alias, "match_location": hit.location, "match_score": hit.score,
                        "match_method": "company_alias", "raw_cache": str(path.relative_to(root))}
                    # Hash the full body to preserve revisions sharing the same source and title.
                    filename = digest([api, source, timestamp, title, content])[:24] + ".json"
                    write_json(target / filename, document)
    all_stats, quality_rows = [], []
    for market, ticker, day in sorted(date_dirs):
        news_root = build / "raw" / market / "company_news" / "news"
        directory = news_root / ticker / day
        bad_json = []
        candidates = [build_candidate(news_root, directory, path, bad_json, matchers[market]) for path in sorted(directory.glob("*.json")) if path.name != "representative_news.json"]
        if bad_json:
            raise ValueError(f"公司原始新闻 JSON 损坏: {bad_json[0]['path']}")
        candidates = [candidate for candidate in candidates if candidate is not None]
        pool, deduped = build_daily_pool(candidates, settings["min_alignment_score"])
        # Enforce the configured threshold explicitly and audit candidates instead of retaining all low-score fallbacks.
        pool = [candidate for candidate in pool if candidate.alignment_score >= settings["min_alignment_score"]]
        write_csv(directory / "news.csv", ["time", "source", "title", "content"], [candidate.csv_row() for candidate in pool])
        if pool:
            representative = pool[0]
            write_json(directory / "representative_news.json", build_representative_payload(
                representative=representative, raw_candidates=candidates, pool_candidates=pool))
            write_csv(directory / "representative_news.csv", ["time", "source", "title", "content"], [representative.csv_row()])
        for candidate in pool:
            quality_rows.append({"market": market, "ticker": ticker, "date": day, "article_id": candidate.article_id,
                "alignment_score": candidate.alignment_score, "quality_score": candidate.quality_score,
                "representative_score": candidate.representative_score, "duplicate_count": candidate.duplicate_count})
        all_stats.append({"market": market, "ticker": ticker, "date": day, "raw_candidate_count": len(candidates),
            "deduped_count": len(deduped), "pool_count": len(pool)})
    write_csv(build / "metadata" / "daily_pool_stats.csv",
        ["market", "ticker", "date", "raw_candidate_count", "deduped_count", "pool_count"], all_stats)
    write_csv(build / "metadata" / "news_quality_scores.csv",
        ["market", "ticker", "date", "article_id", "alignment_score", "quality_score", "representative_score", "duplicate_count"], quality_rows)
    write_json(build / "manifest.json", {"build_id": build_id, "spec_hash": spec_hash, "code_hash": code_hash,
        "pool_hash": pool_hash, "raw_caches": [{"path": str(path.relative_to(root)), "records_sha256": read_json(path)["records_sha256"]} for path in paths], "generated_at": now(), "raw_news_rows": raw_count,
        "company_day_rows": len(all_stats), "pool_news_rows": sum(row["pool_count"] for row in all_stats),
        "rejected_alias_hits": rejected, "companies": config["companies"], "events": events})
    write_json(root / "metadata" / "current_company_build.json", {"build_id": build_id})
    print(f"公司新闻库 {build_id}: {len(all_stats)} 个公司日")
    return build


def build_company_main():
    parser = argparse.ArgumentParser(description=BUILD_COMPANY_NEWS_LIBRARY_HELP)
    add_pipeline_args(parser)
    args = parser.parse_args()
    root, settings, events, companies = spec_from_args(args)
    paths = acquire_pool(root, settings, events, from_cache=True)
    build_library(root, settings, events, companies, paths)
    return 0


def main():
    """Dispatch direct news downloads, shared raw pooling or company processing."""
    commands = {"download": download_news_main, "pool": build_pool_main,
                "company": build_company_main}
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        command = sys.argv.pop(1)
        return commands[command]()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands, help="Select the news workflow step")
    parser.parse_args()
    return 0


if __name__ == "__main__":
    run_cli(main)
