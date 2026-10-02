"""HTTP retries, provider clients, configuration and cache utilities."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import ast
import csv
import gzip
import hashlib
import html
import json
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from dataclasses import asdict, dataclass


# Common

COMMON_HELP = """Shared download utilities. Do not read the original project's data, .env, or credential files."""

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = CODE_DIR.parent / "downloaded_data"


def iso_date(value):
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("日期格式应为 YYYY-MM-DD") from exc


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("必须为非负有限数值")
    return number


def add_common_args(parser, subdir):
    parser.add_argument("--start-date", type=iso_date, required=True)
    parser.add_argument("--end-date", type=iso_date, required=True, help="包含这一天")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / subdir)
    parser.add_argument("--resume", action="store_true", help="跳过已完整写入的相同窗口")
    parser.add_argument("--request-gap-seconds", type=nonnegative, default=1.0)
    parser.add_argument("--retries", type=positive_int, default=3, help="最多尝试次数")
    parser.add_argument("--timeout-seconds", type=positive_int, default=30)


def validate_args(parser, args):
    if args.end_date < args.start_date:
        parser.error("end-date 不能早于 start-date")
    output = args.output_dir.resolve()
    if output == CODE_DIR or CODE_DIR in output.parents:
        parser.error("输出目录必须位于代码发布目录之外")
    args.output_dir = output


def add_symbol_args(parser):
    parser.add_argument("--symbols", nargs="+", help="按数据源要求填写代码")
    parser.add_argument("--symbols-file", type=Path, help="使用者自己的代码列表，每行一个")


def load_symbols(parser, args):
    symbols = list(args.symbols or [])
    if args.symbols_file:
        symbols.extend(
            line.strip() for line in args.symbols_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    symbols = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
    if not symbols:
        parser.error("需要 --symbols 或 --symbols-file")
    if any(not re.fullmatch(r"[A-Z0-9.^=_-]+", symbol) for symbol in symbols):
        parser.error("股票代码包含不支持的字符")
    return symbols


def date_windows(start, end, days):
    while start <= end:
        stop = min(start + timedelta(days=days - 1), end)
        yield start, stop
        if stop == end:
            break
        start = stop + timedelta(days=1)


def safe_name(value):
    return "item_" + re.sub(r"[^\w.-]", "_", value)


def safe_error(exc):
    message = str(exc)
    for key in ("TUSHARE_TOKEN", "TUSHARE_NEWS_TOKEN"):
        token = os.environ.get(key, "").strip()
        if token:
            message = message.replace(token, "[REDACTED]")
    return message


class RequestRunner:
    """Rate-limit requests and retry failures without printing tokens or request bodies."""

    def __init__(self, gap, attempts):
        self.gap = gap
        self.attempts = attempts
        self.last_call = None

    def __call__(self, function):
        for attempt in range(self.attempts):
            if self.last_call is not None:
                time.sleep(max(0, self.gap - (time.monotonic() - self.last_call)))
            self.last_call = time.monotonic()
            try:
                return function()
            except Exception as exc:
                message = safe_error(exc)
                if attempt == self.attempts - 1 or any(
                    word in message.lower()
                    for word in ("权限", "token", "积分", "unauthorized", "forbidden")
                ):
                    raise
                print(f"请求失败，准备重试 ({attempt + 1}/{self.attempts}): {message}", file=sys.stderr)
                time.sleep(min(2 ** (attempt + 1), 30))


class TushareClient:
    """Use the official Pro HTTP protocol and check HTTP status and API error codes.

    Protocol: https://tushare.pro/document/1?doc_id=130
    Check HTTP errors explicitly instead of treating the SDK's empty DataFrame
    fallback as a successful query with no news.
    """

    def __init__(self, token, timeout):
        import requests
        self._token = token
        self._timeout = timeout
        self._session = requests.Session()

    def query(self, api_name, fields="", **params):
        import pandas as pd
        response = self._session.post(
            "https://api.tushare.pro",
            json={"api_name": api_name, "token": self._token, "params": params, "fields": fields},
            timeout=self._timeout,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(f"Tushare: {result.get('msg', 'API 调用失败')}")
        data = result.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("fields"), list) or not isinstance(data.get("items"), list):
            raise RuntimeError("Tushare 返回无效响应")
        return pd.DataFrame(data["items"], columns=data["fields"])

    def close(self):
        self._session.close()


def tushare_client(timeout, *, news=False):
    token = (os.environ.get("TUSHARE_NEWS_TOKEN", "") if news else "").strip()
    token = token or os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("请先设置环境变量 TUSHARE_TOKEN（新闻也可用 TUSHARE_NEWS_TOKEN）")
    try:
        import requests
        import pandas
    except ImportError as exc:
        raise RuntimeError("请运行 python -m pip install requests pandas") from exc
    return TushareClient(token, timeout)


def write_csv(path, fields, rows):
    """Write atomically so a failed write cannot leave a final file mistaken for a completed download."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_cli(main):
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已中止，可用 --resume 继续。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        raise SystemExit(1)


# Pipeline common

PIPELINE_COMMON_HELP = """Pipeline data types, configuration validation, and atomic cache utilities."""

MARKETS = {"CSI300": "CNY", "SP500": "USD", "HSTECH": "HKD", "Nikkei225": "JPY"}
DEFAULT_ALIAS_CSV = DEFAULT_OUTPUT / "pipeline" / "metadata" / "aliases.csv"
DEFAULT_CONFIG = CODE_DIR / "config.py"
DATE_FIELDS = ("pre_start_date", "event_start_date", "event_end_date", "post_end_date")


@dataclass(frozen=True)
class Company:
    ticker: str
    company_name: str
    exchange: str = ""
    raw_code: str = ""
    query_name: str = ""
    aliases: str = ""
    membership_start: str = ""
    membership_end: str = ""
    market: str = ""
    price_source: str = ""
    price_symbol: str = ""
    currency: str = ""


def normalize_text(value):
    return re.sub(r"\s+", " ", html.unescape(str(value or "")).replace("\xa0", " ")).strip()


def clean_content(value):
    text = re.sub(r"<script\b[^>]*>.*?</script>", "", str(value or ""), flags=re.I | re.S)
    return normalize_text(re.sub(r"<[^>]+>", " ", text))


def parse_optional_datetime(value):
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        if re.fullmatch(r"\d{8}", text):
            return datetime.strptime(text, "%Y%m%d")
        raise ValueError(f"无法解析日期时间: {text}")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def output_root(value):
    path = Path(value).resolve()
    if path == CODE_DIR or CODE_DIR in path.parents:
        raise ValueError("数据输出目录必须在代码发布目录之外")
    if CODE_DIR in path.parents or path in CODE_DIR.parents:
        raise ValueError("不能将代码目录或其祖先作为流程输出目录")
    return path


def write_json(path, payload, *, compressed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if compressed:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
    else:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_companies_csv(path):
    return [make_company(row) for row in read_csv(path)]


def yahoo_symbol(market, ticker, exchange=""):
    if market == "SP500":
        return ticker.replace(".", "-")
    if market == "HSTECH":
        return f"{int(ticker.split('.')[0]):04d}.HK"
    if market == "Nikkei225":
        return ticker.removesuffix(".T") + ".T"
    if ticker.endswith((".SS", ".SZ", ".BJ")):
        return ticker
    if "." in ticker:
        code, suffix = ticker.split(".", 1)
    else:
        code, suffix = ticker, exchange.upper()
    if suffix not in ("SH", "SZ", "BJ"):
        raise ValueError("A股需指定 exchange=SH/SZ/BJ 或 price_symbol")
    return code + "." + ("SS" if suffix == "SH" else suffix)


def make_company(row):
    market = row.get("market", "")
    ticker = str(row.get("ticker", "")).strip().upper()
    name = normalize_text(row.get("company_name") or row.get("name"))
    if market not in MARKETS or not ticker or not name:
        raise ValueError("公司需要有效 market、ticker、company_name")
    if not re.fullmatch(r"[A-Z0-9._-]+", ticker):
        raise ValueError("公司 ticker 只能含字母、数字、点、下划线和连字符")
    source = row.get("price_source") or ("tushare" if market == "CSI300" else "yahoo")
    if source not in ("tushare", "yahoo") or (market == "Nikkei225" and source != "yahoo"):
        raise ValueError("无效 price_source；日股使用 yahoo")
    exchange = str(row.get("exchange", "")).upper()
    symbol = str(row.get("price_symbol") or "").upper()
    if not symbol:
        if source == "yahoo":
            symbol = yahoo_symbol(market, ticker, exchange)
        elif market == "HSTECH":
            symbol = f"{int(ticker.split('.')[0]):05d}.HK"
        elif market == "CSI300":
            if ticker.endswith((".SH", ".SZ", ".BJ")):
                symbol = ticker
            elif exchange in ("SH", "SZ", "BJ"):
                symbol = ticker + "." + exchange
            else:
                raise ValueError("A股 Tushare 需提供 price_symbol 或 exchange")
        else:
            symbol = ticker
    if not re.fullmatch(r"[A-Z0-9.^=_-]+", symbol):
        raise ValueError("无效 price_symbol")
    currency = row.get("currency") or MARKETS[market]
    aliases = row.get("aliases", "")
    if isinstance(aliases, (tuple, list)):
        aliases = "|".join(str(item) for item in aliases)
    values = {field: str(row.get(field, "") or "") for field in ("raw_code", "query_name", "membership_start", "membership_end")}
    for field in ("membership_start", "membership_end"):
        if values[field]:
            date.fromisoformat(values[field])
    if values["membership_start"] and values["membership_end"] and values["membership_start"] > values["membership_end"]:
        raise ValueError("membership_start 必须不晚于 membership_end")
    return Company(ticker=ticker, company_name=name, exchange=exchange, market=market,
                   price_source=source, price_symbol=symbol, currency=currency,
                   aliases=str(aliases), **values)


def read_config(path):
    # Read literal values only; do not execute the configuration as arbitrary Python code.
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("SETTINGS", "EVENTS", "COMPANIES"):
                    values[target.id] = ast.literal_eval(node.value)
    return values


def load_spec(path=DEFAULT_CONFIG, companies_csv=None, event_ids=None, require_companies=True):
    defaults = read_config(DEFAULT_CONFIG)
    custom = read_config(Path(path))
    settings = defaults["SETTINGS"] | custom.get("SETTINGS", {})
    settings["news_row_limits"] = defaults["SETTINGS"]["news_row_limits"] | custom.get("SETTINGS", {}).get("news_row_limits", {})
    for key in ("news_sources", "major_sources"):
        if not isinstance(settings[key], list) or any(not isinstance(value, str) or not value.strip() for value in settings[key]):
            raise ValueError(f"{key} 需要非空来源名称列表")
    for key in ("timeout_seconds", "retries", "content_scan_chars"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} 必须为正整数")
    for key in ("request_gap_seconds", "price_request_gap_seconds", "min_match_score", "min_alignment_score"):
        if not isinstance(settings[key], (int, float)) or not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"{key} 需要非负有限数值")
    if any(not isinstance(value, int) or value < 1 for value in settings["news_row_limits"].values()):
        raise ValueError("news_row_limits 需要正整数")
    events = custom.get("EVENTS", defaults["EVENTS"])
    if event_ids:
        unknown = set(event_ids) - {event["event_id"] for event in events}
        if unknown:
            raise ValueError(f"未知 event_id: {sorted(unknown)}")
        events = [event for event in events if event["event_id"] in event_ids]
    if not events:
        raise ValueError("至少需要一个事件")
    seen = set()
    for event in events:
        event_id = event["event_id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", event_id) or event_id in seen:
            raise ValueError("event_id 需唯一且只能包含字母、数字、下划线、连字符")
        seen.add(event_id)
        dates = [date.fromisoformat(event[field]) for field in DATE_FIELDS]
        if dates != sorted(dates):
            raise ValueError(f"{event_id}: 事件和观察窗口日期顺序错误")
    companies = load_companies_csv(Path(companies_csv)) if companies_csv else [make_company(row) for row in custom.get("COMPANIES", [])]
    if require_companies and not companies:
        raise ValueError("COMPANIES 为空，请先生成/填写公司配置或传 --companies-csv")
    keys = [(company.market, company.ticker) for company in companies]
    if len(keys) != len(set(keys)):
        raise ValueError("market/ticker 必须唯一")
    return settings, events, companies


def event_days(events):
    days = set()
    for event in events:
        start = date.fromisoformat(event["pre_start_date"])
        end = date.fromisoformat(event["post_end_date"])
        while start <= end:
            days.add(start)
            start += timedelta(days=1)
    return sorted(days)


def period(event, day):
    if day < event["event_start_date"]:
        return "pre_event"
    if day <= event["event_end_date"]:
        return "during_event"
    return "post_event"


def add_pipeline_args(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--companies-csv", type=Path)
    parser.add_argument("--aliases-csv", type=Path, help="自己的别名表：market,ticker,alias,alias_type,confidence,valid_from,valid_to")
    parser.add_argument("--event-ids", nargs="+")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_OUTPUT / "pipeline")


def spec_from_args(args, *, require_companies=True):
    settings, events, companies = load_spec(args.config, args.companies_csv, args.event_ids, require_companies)
    if args.aliases_csv:
        settings["alias_records"] = settings.get("alias_records", []) + read_csv(args.aliases_csv)
    return output_root(args.work_dir), settings, events, companies


def load_build(root):
    pointer = read_json(root / "metadata" / "current_company_build.json")
    build = root / "builds" / pointer["build_id"]
    if not build.is_dir():
        raise ValueError("公司新闻库构建不存在，请运行 company 阶段")
    return build, read_json(build / "manifest.json")


def matching_build(root, settings, events, companies):
    build, manifest = load_build(root)
    expected = digest({"companies": [asdict(c) for c in companies], "events": events, "settings": settings})
    if manifest["spec_hash"] != expected:
        raise ValueError("公司库与当前配置不一致，请重新运行 company 阶段")
    current_code = digest({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in CODE_DIR.glob("*.py")})
    if manifest["code_hash"] != current_code:
        raise ValueError("处理代码已变更，请重新运行 company 阶段")
    for cache in manifest["raw_caches"]:
        payload = read_json(root / cache["path"])
        if digest(payload["records"]) != cache["records_sha256"]:
            raise ValueError("原始池已更新或损坏，请重新运行 company 阶段")
    return build, manifest


def make_matchers(companies, settings):
    """Explicit alias validity is independent of index membership dates."""
    from news import AliasRecord, build_alias_matcher, compact_alias, derive_company_aliases
    from collections import defaultdict
    by_market = defaultdict(list)
    known = {(c.market, c.ticker) for c in companies}
    explicit = []
    overridden = set()
    for row in settings.get("alias_records", []):
        identity = row["market"], str(row["ticker"])
        if identity not in known:
            raise ValueError("Explicit alias has unknown market/ticker")
        alias = normalize_text(row["alias"])
        confidence = float(row.get("confidence") or 0.9)
        if not 0 <= confidence <= 1:
            raise ValueError("Alias confidence must be in [0,1]")
        start, end = row.get("valid_from", ""), row.get("valid_to", "")
        if start:
            date.fromisoformat(start)
        if end:
            date.fromisoformat(end)
        if start and end and start > end:
            raise ValueError("Invalid alias validity interval")
        from news import is_usable_alias
        if not is_usable_alias(alias):
            raise ValueError("Alias is numeric-only, too short or ambiguous")
        explicit.append((identity[0], AliasRecord(identity[1], alias, row.get("alias_type") or "manual_alias", confidence, start, end)))
        overridden.add((identity[0], identity[1], compact_alias(alias)))
    for company in companies:
        for record in derive_company_aliases(company):
            if (company.market, company.ticker, record.compact_alias) not in overridden:
                by_market[company.market].append(record)
        by_market.setdefault(company.market, [])
    for market, record in explicit:
        by_market[market].append(record)
    return {market: build_alias_matcher(records) for market, records in by_market.items()}

