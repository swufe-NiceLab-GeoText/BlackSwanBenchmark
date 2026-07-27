# BlackSwanNewsBench Data Dictionary

## Common Fields

| Field | Meaning | Public level |
|---|---|---|
| `event_id` | Stable event identifier. | public |
| `market` | Market universe name: CSI300, HSTECH, Nikkei225, or SP500. | public |
| `ticker` | Source ticker or symbol file stem. | public |
| `date` | Record date used for daily aggregation. | public |
| `source_role` | Source role in inventory: `main_local_news`, `tushare_supplement`, or `gdelt_metadata`. | public |
| `source_name` | Publisher/source name from the upstream record. | public |
| `source_type` | Upstream source type label. | public |

## Inventory Fields

| Field | Meaning | Public level |
|---|---|---|
| `data_version` | Generated P0 freeze version. | public |
| `source_dir` | Local frozen source directory under `data/`. | public |
| `source_type` | Human-readable source type. | public |
| `content_policy` | Public artifact policy for the source. | public |
| `rights_class_default` | Default rights class for rows from the source. | public |
| `gdelt_metadata_only` | Whether the source is GDELT metadata-only. | public |
| `p0_status` | Whether the source is included in the P0 freeze. | public |
| `ticker_jsonl_files_reported` | Ticker file count from the source summary. | public |
| `ticker_jsonl_files_on_disk` | Ticker file count observed under the frozen event directory. | public |
| `article_rows` | Raw article/mention row count in the source summary. | public |
| `candidate_article_rows` | Rows that passed upstream candidate filtering before final acceptance. | public |
| `ok_article_rows` | Rows marked usable by upstream source construction. | public |
| `bad_input_rows` | Rows rejected by upstream input validation. | public |
| `first_date` | Earliest observed record date for a source-event. | public |
| `last_date` | Latest observed record date for a source-event. | public |
| `metadata_sha256` | SHA-256 digest over source metadata input files. | public |
| `notes` | Caveats such as metadata-only status or coverage gaps. | public |

## Event Fields

| Field | Meaning | Public level |
|---|---|---|
| `event_name` | Human-readable event name. | public |
| `event_type` | Coarse crisis type. | public |
| `pre_start_date` | Earliest frozen data date for the event. | public |
| `event_start_date` | External event anchor date. | public |
| `event_end_date` | Benchmark initial shock-window endpoint. | public |
| `post_end_date` | Latest frozen data date for the event. | public |
| `description` | Short event-window description. | public |
| `source_note` | Source note and event-anchor links. | public |

## Company Fields

| Field | Meaning | Public level |
|---|---|---|
| `company_id` | Stable benchmark company key formatted as `{market}:{ticker}`. | public |
| `company_name` | Most common sampled source company name. | public |
| `local_name` | Local-language or sampled source name when available. | public |
| `exchange` | Coarse exchange or trading venue group. | public |
| `currency` | Expected price currency for later label construction. | public |
| `universe_policy` | Universe construction policy for the row. Current primary value: `observed_company_universe`. | public |
| `membership_start` | Point-in-time membership start when known; blank because the primary denominator is observed-company scope. | public |
| `membership_end` | Point-in-time membership end when known; blank because the primary denominator is observed-company scope. | public |
| `event_in_universe` | Semicolon-separated events where the company has at least one frozen source file or article/mention record. | public |
| `source_covered` | Source-file availability category. | public |
| `price_available` | Price-data availability status; currently `unknown`. | public |
| `alias_source` | Source directories where aliases/names were observed. | public |
| `aliases` | Semicolon-separated sampled aliases; public metadata, not article text. | public |
| `audit_status` | Current universe/matching audit status. | public |

## Source News Text Fields

| Field | Meaning | Public level |
|---|---|---|
| `source_news_id` | Stable source-news row key. | public |
| `company_name` | Company name for the matched ticker. | public |
| `source_name` | News source name. | public |
| `source_language` | Source-provider language code added during Hugging Face sharding: `zh`, `en`, or `ja`. | derived |
| `url_hash` | URL hash used for deduplication. | public |
| `title` | News title. | public |
| `content` | News content used by the public source-news text table. | public |

## Missingness Semantics

| Value | Meaning |
|---|---|
| empty date/name field | Unknown or not yet audited. |
| `source_covered=gdelt_only` | A ticker has GDELT metadata files but no main/local file in the frozen source set. |
| `source_covered=supplement_only` | A ticker appears only in a supplement layer. |
| `source_covered=main_and_gdelt` | A ticker has at least one main/local and one GDELT source file. |
| `price_available=unknown` | Price availability is not encoded in `companies.csv`; use `price_ohlcv.parquet` for published price rows. |

Source unavailable must remain distinct from true zero-news observations in all
feature builders.

The primary denominator is the observed-company universe. It should not be
described as a complete point-in-time constituent universe.

## Price OHLCV Fields

Official file: `data/price/*.parquet`.

Primary key: `market, ticker, date`.

| Field | Meaning | Public level |
|---|---|---|
| `open` | Daily open price. | public |
| `high` | Daily high price. | public |
| `low` | Daily low price. | public |
| `close` | Daily close price. | public |
| `volume` | Daily trading volume. | public |
| `currency` | Trading currency for the price row. | public |

## GDELT Firm-Day Feature Fields

Official file: `data/gdelt_firm/*.parquet`.

Primary key: `event_id, market, ticker, date`.

Rows are pre-event only: `pre_start_date <= date < event_start_date`.

| Field | Meaning | Public level |
|---|---|---|
| `company_name` | Company name carried from `companies.csv` for readability. | public |
| `gdelt_news_count` | Unique GDELT article count per firm-day after high-confidence matching; deduplicated by URL hash with `article_id` fallback. | public |
| `gdelt_source_count` | Distinct GDELT source names for the firm-day. | public |
| `gdelt_source_entropy` | Source-name entropy over firm-day GDELT rows. | public |
| `avg_gdelt_tone` | Mean GDELT tone over firm-day GDELT rows. | public |
| `min_gdelt_tone` | Minimum GDELT tone over firm-day GDELT rows. | public |
| `max_gdelt_tone` | Maximum GDELT tone over firm-day GDELT rows. | public |
| `positive_news_count` | Count of firm-day GDELT rows with positive tone. | public |
| `negative_news_count` | Count of firm-day GDELT rows with negative tone. | public |
| `neutral_news_count` | Count of firm-day GDELT rows with exactly zero tone. | public |
| `tone_missing_count` | Count of firm-day GDELT rows where `tone_avg` is missing. | public |
| `positive_news_share` | `positive_news_count / gdelt_news_count`, or 0 when no GDELT news is observed. | public |
| `negative_news_share` | `negative_news_count / gdelt_news_count`, or 0 when no GDELT news is observed. | public |
| `gdelt_theme_count` | Number of distinct GDELT theme tokens on the firm-day. | public |
| `gdelt_entity_count` | Number of distinct GDELT organization/entity tokens on the firm-day. | public |
| `top_gdelt_themes` | Semicolon-separated top theme tokens with counts, e.g. `TAX_FNCACT:42`. | public |
| `coverage_missing_flag` | True when the company-event lacks GDELT source-file coverage; not true zero news. | public |

## GDELT Market-Day Context Fields

Official file: `data/gdelt_market/*.parquet`.

Primary key: `event_id, market, date`.

Rows are pre-event only: `pre_start_date <= date < event_start_date`.

| Field | Meaning | Public level |
|---|---|---|
| `eligible_firm_count` | Observed-company count for the market-event denominator. | public |
| `covered_firm_count` | Firms with at least one high-confidence GDELT row on that market-day. | public |
| `covered_firm_share` | `covered_firm_count / eligible_firm_count`. | public |
| `market_gdelt_firm_mention_count` | Firm-article match count; the same article may count once for each matched company. | public |
| `market_gdelt_unique_article_count` | Unique GDELT article count after URL-hash deduplication within the market-day. | public |
| `avg_market_gdelt_tone_equal_weighted` | Mean of covered firm-day tone means; each covered firm has equal weight. | public |
| `avg_market_gdelt_tone_news_weighted` | Mean tone over firm-article mentions; firms with more GDELT mentions have more weight. | public |
| `min_market_firm_tone` | Lowest covered firm-day average tone on the market-day. | public |
| `negative_firm_count` | Covered firms whose firm-day average tone is negative. | public |
| `negative_firm_share` | `negative_firm_count / covered_firm_count`, or 0 when no firms are covered. | public |
| `market_negative_news_count` | Firm-article mentions with negative tone on the market-day. | public |
| `market_negative_news_share` | Share of firm-article mentions with negative tone. | public |
| `market_source_count` | Distinct GDELT source names in the market-day context. | public |
| `market_source_entropy` | Source-name entropy in the market-day context. | public |
| `market_theme_count` | Number of distinct GDELT theme tokens in unique market-day articles. | public |
| `market_entity_count` | Number of distinct GDELT organization/entity tokens in unique market-day articles. | public |
| `top_market_gdelt_themes` | Semicolon-separated top market GDELT theme tokens with counts. | public |
| `coverage_missing_flag` | True when market-event GDELT source coverage is unavailable for the date. | public |

## Ready-to-Run Daily Features

Official files: `data/daily_features/*.parquet`.

Primary key: `event_id, market, ticker, date`. These rows contain strictly
pre-event structured price, source-news coverage, firm-level GDELT, and
market-level GDELT inputs. They contain no event-window label columns.

## Source-News Text Embeddings

Official files: `data/text_embeddings/*.parquet`.

Primary key: `event_id, market, ticker, date`. `text_doc_count` records the
number of source-news associations represented by the daily embedding;
`embedding_model` identifies BAAI/bge-m3; `embedding_dim` is 1024; and
`emb_000` through `emb_1023` store the vector values.

## Event Outcomes

Official files: `data/outcomes/*.parquet`.

Primary key: `event_id, market, ticker`.

| Field | Meaning | Public level |
|---|---|---|
| `event_return` | Company close-to-close return over the event window. | derived |
| `label_available_flag` | Whether the event-window outcome can be constructed. | derived |

The files intentionally omit `high_impact_label`. Split-aware abnormal returns
and training-only Q90 labels are generated by the benchmark runtime.
