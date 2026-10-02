"""Editable pipeline configuration with an empty company universe.

Fill COMPANIES, generate your own configuration with run_pipeline.py create-config,
or pass --companies-csv to run_pipeline.py. No existing company or alias data is
bundled. Default dates follow the newer frozen Data Field Specification and may
be edited for your study.
"""
SETTINGS = {
    "news_sources": ["sina", "10jqka", "eastmoney", "wallstreetcn", "cls", "yicai", "fenghuang", "jinrongjie", "yuncaijing"],
    "major_sources": ["新浪财经", "同花顺", "华尔街见闻", "财联社", "第一财经", "凤凰财经"],
    "news_row_limits": {"news": 1500, "major_news": 400},
    "request_gap_seconds": 10.0,
    "price_request_gap_seconds": 1.0,
    "alias_records": [],
    "timeout_seconds": 30,
    "retries": 3,
    "min_match_score": 18.0,
    "min_alignment_score": 20.0,
    "content_scan_chars": 2400,
    "raw_pool_start": "",
    "raw_pool_end": "",
}
EVENTS = [
    {"event_id": "covid19_first_wave", "pre_start_date": "2019-11-01", "event_start_date": "2020-01-30",
     "event_end_date": "2020-03-11", "post_end_date": "2020-06-29"},
    {"event_id": "russia_ukraine_conflict_2022", "pre_start_date": "2021-11-23", "event_start_date": "2022-02-24",
     "event_end_date": "2022-03-31", "post_end_date": "2022-06-29"},
    {"event_id": "israel_palestine_conflict_2023", "pre_start_date": "2023-07-09", "event_start_date": "2023-10-07",
     "event_end_date": "2023-11-06", "post_end_date": "2024-02-13"},
    {"event_id": "us_israel_iran_conflict_2025", "pre_start_date": "2025-03-15", "event_start_date": "2025-06-13",
     "event_end_date": "2025-06-24", "post_end_date": "2025-09-22"},
]
COMPANIES = []
# Fields: market, ticker, company_name, aliases (separated by |),
# price_source (yahoo/tushare), price_symbol, currency.
# Optional: exchange, raw_code, query_name, membership_start, membership_end.
# membership_start/end bound membership validity; empty values indicate an observed-company universe.
# Users must supply Chinese/Japanese aliases and former names; the API does not translate aliases.
