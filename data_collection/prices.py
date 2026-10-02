"""Yahoo Finance and Tushare price downloads and normalized price libraries."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from datetime import date, datetime, time, timedelta, timezone
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from common import (
    RequestRunner, add_common_args, add_pipeline_args, add_symbol_args, date_windows, digest,
    event_days, load_symbols, normalize_text, now, positive_int, read_json, run_cli,
    safe_error, safe_name, spec_from_args, tushare_client, validate_args, write_csv,
    write_json,
)


# Download yahoo prices

DOWNLOAD_YAHOO_PRICES_HELP = """Fetch Yahoo Finance daily prices using the project's chart-download approach.

Dependencies: python -m pip install requests tzdata
Example: python prices.py yahoo --symbols AAPL MSFT 0700.HK 7203.T --start-date 2020-01-01 --end-date 2020-03-31
Use Yahoo symbols directly: BRK-B for US, 0700.HK for Hong Kong, and 7203.T for Japan.
OHLC is unadjusted; adj_close is Yahoo's adjusted closing price; volume is in shares.
Use the exchange's time zone for trading dates. Yahoo chart is a website endpoint
whose availability can change.
"""

YAHOO_FIELDS = ["date", "ticker", "symbol", "open", "high", "low", "close", "adj_close", "volume"]


def epoch(day):
    return int(datetime.combine(day, time.min, tzinfo=timezone.utc).timestamp())


def chart_rows(payload, symbol, start, end):
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"Yahoo: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo 没有返回有效结果")
    result = results[0]
    tz_name = (result.get("meta") or {}).get("exchangeTimezoneName")
    try:
        if not tz_name:
            raise ZoneInfoNotFoundError("missing exchange timezone")
        local_tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError("缺少交易所时区，请运行 python -m pip install tzdata") from exc
    indicators = result.get("indicators") or {}
    prices = (indicators.get("quote") or [{}])[0]
    adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose") or []

    def at(values, index):
        return values[index] if index < len(values) and values[index] is not None else ""

    rows = {}
    for index, timestamp in enumerate(result.get("timestamp") or []):
        day = datetime.fromtimestamp(timestamp, local_tz).date()
        if not start <= day <= end:
            continue
        row = {"date": day.isoformat(), "ticker": symbol, "symbol": symbol}
        row.update({key: at(prices.get(key) or [], index)
                    for key in ("open", "high", "low", "close", "volume")})
        row["adj_close"] = at(adjusted, index)
        if row["close"] != "":
            rows[day] = row
    return [rows[day] for day in sorted(rows)]


def download_yahoo_main():
    parser = argparse.ArgumentParser(description=DOWNLOAD_YAHOO_PRICES_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "yahoo_prices")
    add_symbol_args(parser)
    args = parser.parse_args()
    validate_args(parser, args)
    symbols = load_symbols(parser, args)
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("请运行 python -m pip install requests tzdata") from exc
    runner = RequestRunner(args.request_gap_seconds, args.retries)
    failures = 0
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        # Use the user's HTTPS_PROXY / HTTP_PROXY settings without a hard-coded proxy.
        for symbol in symbols:
            path = args.output_dir / f"{safe_name(symbol)}_{args.start_date}_{args.end_date}.csv"
            if args.resume and path.is_file():
                print(f"跳过 {symbol}")
                continue
            try:
                def fetch():
                    response = session.get(
                        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}",
                        params={"period1": epoch(args.start_date - timedelta(days=1)),
                                "period2": epoch(args.end_date + timedelta(days=2)),
                                "interval": "1d", "events": "history", "includeAdjustedClose": "true"},
                        timeout=args.timeout_seconds,
                    )
                    response.raise_for_status()
                    return chart_rows(response.json(), symbol, args.start_date, args.end_date)

                rows = runner(fetch)
                if not rows:
                    failures += 1
                    print(f"{symbol}: 日期范围内无行情，请检查代码、上市时间和交易日", file=sys.stderr)
                    continue
                write_csv(path, YAHOO_FIELDS, rows)
                print(f"{symbol}: {len(rows)} 行")
            except Exception as exc:
                failures += 1
                print(f"{symbol}: {safe_error(exc)}", file=sys.stderr)
    return 1 if failures else 0


# Download tushare prices

DOWNLOAD_TUSHARE_PRICES_HELP = """Fetch unadjusted daily A-share, Hong Kong, or US prices from Tushare.

Dependencies: python -m pip install requests pandas
Read TUSHARE_TOKEN from the environment, without .env files or command-line tokens.
Example: python prices.py tushare --market cn --symbols 600519.SH 000001.SZ --start-date 2020-01-01 --end-date 2020-03-31
Hong Kong: --market hk --symbols 00700.HK; US: --market us --symbols AAPL.
Preserve API fields and units, sort by trade_date, and do not synthesize adj_close.
A-share vol is measured in lots and amount in thousands of CNY; these are not
directly interchangeable with Yahoo volume.
Documentation: https://tushare.pro/document/2?doc_id=27
Hong Kong/US access may require separate permission. Fetch one symbol at a time
in windows of at most 365 days to avoid truncation.
"""

ENDPOINTS = {"cn": "daily", "hk": "hk_daily", "us": "us_daily"}


def download_tushare_main():
    parser = argparse.ArgumentParser(description=DOWNLOAD_TUSHARE_PRICES_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, "tushare_prices")
    add_symbol_args(parser)
    parser.add_argument("--market", choices=ENDPOINTS, required=True)
    parser.add_argument("--slice-days", type=positive_int, default=365)
    args = parser.parse_args()
    validate_args(parser, args)
    if args.slice_days > 365:
        parser.error("slice-days 不能大于365，以避免 API 行数上限截断")
    symbols = load_symbols(parser, args)
    pro = tushare_client(args.timeout_seconds)
    endpoint = ENDPOINTS[args.market]
    runner = RequestRunner(args.request_gap_seconds, args.retries)
    failures = 0
    try:
        for symbol in symbols:
            path = args.output_dir / args.market / f"{safe_name(symbol)}_{args.start_date}_{args.end_date}.csv"
            if args.resume and path.is_file():
                print(f"跳过 {symbol}")
                continue
            try:
                frames = []
                for start, end in date_windows(args.start_date, args.end_date, args.slice_days):
                    frame = runner(lambda: pro.query(endpoint, ts_code=symbol,
                        start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d")))
                    if len(frame) >= 6000:
                        raise RuntimeError("结果达到 API 上限，请缩小 slice-days 后重试")
                    if not frame.empty:
                        frames.append(frame)
                if not frames:
                    failures += 1
                    print(f"{symbol}: 日期范围内无行情，请检查代码、权限及上市日期", file=sys.stderr)
                    continue
                import pandas as pd
                data = pd.concat(frames, ignore_index=True)
                data = data.drop_duplicates(["ts_code", "trade_date"]).sort_values("trade_date")
                write_csv(path, list(data.columns), data.to_dict(orient="records"))
                print(f"{symbol}: {len(data)} 行")
            except Exception as exc:
                failures += 1
                print(f"{symbol}: {safe_error(exc)}", file=sys.stderr)
    finally:
        pro.close()
    return 1 if failures else 0


# Build price library

BUILD_PRICE_LIBRARY_HELP = """Fetch and normalize prices for four markets, retaining raw responses and provenance.

Use unadjusted OHLC. Keep Yahoo adj_close in caches and audit outputs separately
from close. Convert Tushare A-share vol from lots to shares by multiplying by 100;
Hong Kong/US vol and Yahoo volume are already measured in shares.
Leave suspended sessions, weekends, and missing prices unfilled. Do not replace
delisted companies with their successors.
"""

PRICE_FIELDS = ["market", "ticker", "date", "open", "high", "low", "close", "volume", "currency"]


def price_tasks(events, companies):
    days = event_days(events)
    ranges = []
    for day in days:
        if ranges and ranges[-1][1] + timedelta(days=1) == day:
            ranges[-1] = (ranges[-1][0], day)
        else:
            ranges.append((day, day))
    for company in companies:
        for start, end in ranges:
            for first, last in date_windows(start, end, 365):
                yield company, {"market": company.market, "ticker": company.ticker,
                    "source": company.price_source, "symbol": company.price_symbol,
                    "currency": company.currency, "start_date": str(first), "end_date": str(last),
                    "schema_version": 1}


def cache_path(root, task):
    return root / "prices" / "raw_cache" / task["market"] / task["source"] / (digest(task)[:24] + ".json.gz")


def validate_price_cache(path, task):
    payload = read_json(path)
    if any(payload.get(key) != value for key, value in task.items()):
        raise ValueError("股价缓存配置不匹配")
    if not isinstance(payload.get("records"), list) or digest(payload["records"]) != payload.get("records_sha256"):
        raise ValueError("股价缓存内容校验失败")
    for row in payload["records"]:
        if row["market"] != task["market"] or row["ticker"] != task["ticker"] or row["currency"] != task["currency"]:
            raise ValueError("股价缓存身份/币种错误")
        if not task["start_date"] <= row["date"] <= task["end_date"]:
            raise ValueError("股价缓存包含窗口外数据")
    return payload


def fetch_prices(company, task, runner, yahoo_session, pro, timeout):
    start, end = date.fromisoformat(task["start_date"]), date.fromisoformat(task["end_date"])
    if task["source"] == "yahoo":
        def fetch():
            response = yahoo_session.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/" + quote(task["symbol"], safe=""),
                params={"period1": epoch(start - timedelta(days=1)), "period2": epoch(end + timedelta(days=2)),
                        "interval": "1d", "events": "history", "includeAdjustedClose": "true"},
                timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            rows = chart_rows(payload, task["symbol"], start, end)
            metadata = payload["chart"]["result"][0].get("meta", {})
            actual_currency = metadata.get("currency")
            if actual_currency and actual_currency != task["currency"]:
                raise ValueError(f"Yahoo 币种 {actual_currency} 与公司配置 {task['currency']} 不符")
            return rows, payload
        records, raw = runner(fetch)
        rows = [{**{field: row[field] for field in ("date", "open", "high", "low", "close", "volume")},
                 "market": company.market, "ticker": company.ticker, "currency": company.currency,
                 "adj_close": row["adj_close"]} for row in records]
        return rows, raw, 1
    endpoints = {"CSI300": "daily", "HSTECH": "hk_daily", "SP500": "us_daily"}
    def fetch():
        frame = pro.query(endpoints[company.market], ts_code=task["symbol"],
            start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        cap = 5000 if company.market == "HSTECH" else 6000
        if len(frame) >= cap:
            raise ValueError("单股365天结果仍达到接口上限，拒绝截断")
        records = frame.fillna("").to_dict(orient="records")
        for row in records:
            if str(row.get("ts_code", "")).upper() != task["symbol"]:
                raise ValueError("股价接口返回了另一只股票")
        return records
    records = runner(fetch)
    factor = 100 if company.market == "CSI300" else 1
    rows = []
    for row in records:
        day = datetime.strptime(str(row["trade_date"]), "%Y%m%d").date().isoformat()
        if not task["start_date"] <= day <= task["end_date"]:
            raise ValueError("股价接口返回窗口外数据")
        values = {field: row.get(field, "") for field in ("open", "high", "low", "close")}
        volume = row.get("vol", "")
        values["volume"] = float(volume) * factor if volume != "" else ""
        rows.append({"market": company.market, "ticker": company.ticker, "date": day,
                     "currency": company.currency, "adj_close": "", **values})
    return rows, records, factor


def acquire_prices(root, settings, events, companies, *, from_cache=False, refresh=False):
    if from_cache and refresh:
        raise ValueError("from-cache 与 refresh 不能同时使用")
    import requests
    runner = RequestRunner(settings.get("price_request_gap_seconds", 1.0), settings["retries"])
    pro = None
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    audit, rows, snapshots = [], {}, []
    checkpoint = root / "metadata" / "price_checkpoint.json"
    try:
        for company, task in price_tasks(events, companies):
            path = cache_path(root, task)
            state = dict(task, cache=str(path.relative_to(root)))
            try:
                cached = None
                if path.is_file() and not refresh:
                    try:
                        cached = validate_price_cache(path, task)
                    except (ValueError, OSError, EOFError):
                        if from_cache:
                            raise
                if cached is None:
                    if from_cache:
                        raise FileNotFoundError(f"缺少股价缓存: {task['market']} {task['ticker']} {task['start_date']}")
                    if task["source"] == "tushare" and pro is None:
                        pro = tushare_client(settings["timeout_seconds"])
                    records, raw, factor = fetch_prices(company, task, runner, session, pro, settings["timeout_seconds"])
                    cached = dict(task, records=records, raw_response=raw, row_count=len(records),
                        records_sha256=digest(records), volume_multiplier=factor, adjustment="unadjusted", retrieved_at=now())
                    write_json(path, cached, compressed=True)
                    origin = "downloaded"
                else:
                    origin = "cache"
                for row in cached["records"]:
                    key = row["market"], row["ticker"], row["date"]
                    if key in rows and rows[key] != row:
                        raise ValueError("重叠窗口出现不同股价记录")
                    rows[key] = row
                snapshots.append({"task": task, "records_sha256": cached["records_sha256"]})
                state.update(status="empty" if not cached["records"] else "complete", row_count=len(cached["records"]),
                    volume_multiplier=cached["volume_multiplier"], adjustment="unadjusted", origin=origin)
                print(f"股价 {company.market} {company.ticker} {task['start_date']}: {len(cached['records'])} ({origin})", flush=True)
            except Exception as exc:
                state.update(status="error", error=safe_error(exc))
                print(state["error"], file=sys.stderr, flush=True)
            audit.append(state)
            write_json(checkpoint, {"updated_at": now(), "tasks": audit, "failed_tasks": sum(item["status"] == "error" for item in audit)})
    finally:
        session.close()
        if pro is not None:
            pro.close()
    if any(item["status"] == "error" for item in audit):
        raise RuntimeError("股价获取存在失败/缺失片段，见 price_checkpoint.json")
    fingerprint = digest(snapshots)[:24]
    output = root / "prices" / "normalized" / fingerprint
    ordered = [rows[key] for key in sorted(rows)]
    write_csv(output / "price_ohlcv.csv", PRICE_FIELDS, ordered)
    write_csv(output / "price_provenance.csv",
        PRICE_FIELDS + ["adj_close"], ordered)
    write_csv(output / "price_source_audit.csv",
        ["market", "ticker", "source", "symbol", "currency", "start_date", "end_date", "status", "row_count", "volume_multiplier", "adjustment", "cache", "origin"], audit)
    for market in sorted({company.market for company in companies}):
        write_csv(output / "by_market" / f"{market}_prices.csv", PRICE_FIELDS,
            [row for row in ordered if row["market"] == market])
    write_json(output / "manifest.json", {"fingerprint": fingerprint, "generated_at": now(),
        "row_count": len(ordered), "tasks": snapshots, "events": events, "volume_unit": "shares",
        "adjustment": "unadjusted", "missing_price_policy": "no_imputation"})
    write_json(root / "metadata" / "current_price_build.json", {"fingerprint": fingerprint})
    return output


def load_price_build(root, events, companies):
    fingerprint = read_json(root / "metadata" / "current_price_build.json")["fingerprint"]
    output = root / "prices" / "normalized" / fingerprint
    manifest = read_json(output / "manifest.json")
    expected = [task for _, task in price_tasks(events, companies)]
    if [item["task"] for item in manifest["tasks"]] != expected:
        raise ValueError("股价库与当前事件/公司配置不一致，请重新运行 prices 阶段")
    return output


def build_price_main():
    parser = argparse.ArgumentParser(description=BUILD_PRICE_LIBRARY_HELP)
    add_pipeline_args(parser)
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root, settings, events, companies = spec_from_args(args)
    acquire_prices(root, settings, events, companies, from_cache=args.from_cache, refresh=args.refresh)
    return 0


def main():
    """Dispatch standalone downloads or a normalized price-library build."""
    commands = {"yahoo": download_yahoo_main, "tushare": download_tushare_main,
                "build": build_price_main}
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        command = sys.argv.pop(1)
        return commands[command]()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands, help="Select a provider or build the price library")
    parser.parse_args()
    return 0


if __name__ == "__main__":
    run_cli(main)
