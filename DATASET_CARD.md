---
pretty_name: BlackSwanNewsBench
language:
- zh
- en
- ja
license: other
task_categories:
- time-series-forecasting
- tabular-classification
- tabular-regression
tags:
- finance
- multimodal
- benchmark
- event-driven
- out-of-distribution
- gdelt
size_categories:
- 1M<n<10M
configs:
- config_name: outcomes
  default: true
  data_files:
  - split: full
    path: data/outcomes/*.parquet
- config_name: price
  data_files:
  - split: full
    path: data/price/*.parquet
- config_name: news
  data_files:
  - split: full
    path: data/news/*.parquet
- config_name: gdelt_firm
  data_files:
  - split: full
    path: data/gdelt_firm/*.parquet
- config_name: gdelt_market
  data_files:
  - split: full
    path: data/gdelt_market/*.parquet
- config_name: daily_features
  data_files:
  - split: full
    path: data/daily_features/*.parquet
- config_name: text_embeddings
  data_files:
  - split: full
    path: data/text_embeddings/*.parquet
---

# BlackSwanNewsBench

BlackSwanNewsBench is a leakage-safe, multimodal, multi-market benchmark for
company-level impact prediction around rare shock events. Each benchmark sample
is an `(event_id, market, ticker)` tuple. Model inputs are restricted to dates
strictly before the event anchor; event-window returns are labels only.

## Scope

- Four anchored shock windows and four equity markets: CSI300, HSTECH,
  Nikkei225, and SP500.
- 1,344 observed firms, 5,046 candidate event-company samples, and 4,837
  samples with valid event-window outcomes.
- Chinese, English, and Japanese source-news coverage.
- Price, source-linked company news, firm-level GDELT, market-level GDELT, and
  BGE-M3 news representations.

Exact market and event-market counts are provided in
`market_data_description.csv` and `event_market_data_description.csv`.

## Data sources

- Daily equity prices are primarily collected through Tushare and Yahoo
  Finance, with missing U.S. price records supplemented through AkShare/Sina.
- Chinese company news comes from the frozen collection of major financial
  sources recorded in `metadata/source_inventory.csv`; English company news is
  sourced from Nasdaq.
- Japanese corporate disclosures are collected through JPX/TDnet interfaces,
  the EDINET API v2, and the Yanoshin TDnet historical archive.
- Global media-attention metadata is derived from GDELT 2.1 GKG via BigQuery.

Provider names describe provenance and do not transfer upstream ownership or
licensing rights to this benchmark.

## Repository layout

```text
BlackSwanNewsBench/
|-- README.md
|-- market_data_description.csv
|-- event_market_data_description.csv
|-- metadata/
|   |-- events.csv
|   |-- companies.csv
|   |-- source_inventory.csv
|   `-- release_manifest.json
|-- data/
|   |-- price/
|   |-- news/
|   |-- gdelt_firm/
|   |-- gdelt_market/
|   |-- daily_features/
|   |-- text_embeddings/
|   `-- outcomes/
|-- splits/
`-- docs/
```

The large tables are stored as typed, compressed Parquet shards. This retains
the modality-oriented organization used by financial multimodal datasets while
supporting Hugging Face Dataset Viewer, streaming, and selective loading.

## Load from Hugging Face

```python
from datasets import load_dataset

repo_id = "YOUR_ORG/BlackSwanNewsBench"
outcomes = load_dataset(repo_id, "outcomes", split="full")
prices = load_dataset(repo_id, "price", split="full", streaming=True)
news = load_dataset(repo_id, "news", split="full", streaming=True)
```

`full` is a storage split used by the Hub. It is not a benchmark train/test
split. Official RandomFirm, Leave-One-Event-Out (LOEO), and
Leave-One-Market-Out (LOMO) definitions are frozen under `splits/` and are
applied by the benchmark code.

## Standardized tasks

- **T1:** high-impact classification. The high-impact threshold is the Q90 of
  training-split absolute abnormal returns and is never fit on validation or
  test labels.
- **T2:** signed abnormal-return regression over the event window.
- **T3:** event-market impact ranking, using `abs(T2_prediction)` and grouping
  metrics by `event_id + market`.

Base outcome files intentionally do not contain a materialized
`high_impact_label`.

## Data configurations

| Config | Unit | Description |
|---|---|---|
| `outcomes` | event-market-firm | Base event returns and label availability |
| `price` | market-firm-trading-day | Daily OHLCV observations |
| `news` | event-market-firm-news association | Source-linked title/content records; repeated associations are not unique article counts |
| `gdelt_firm` | event-market-firm-day | Firm-level metadata-derived GDELT signals |
| `gdelt_market` | event-market-day | Market-level GDELT context |
| `daily_features` | event-market-firm-day | Strictly pre-event merged structured inputs |
| `text_embeddings` | event-market-firm-day | BGE-M3 1024-dimensional daily news representations |

## Rights and licensing

This repository uses `license: other` because upstream price, source-news, and
metadata assets have source-specific terms. The project owner has recorded a
publication-rights assertion for the frozen source-news artifact, but this is
not an external legal opinion. GDELT tables contain metadata-derived aggregate
features, not third-party article bodies. See `docs/rights_policy.md` and
`metadata/source_inventory.csv` before mirroring or redistributing the data.

## Documentation

- `docs/data_dictionary.md`: fields, keys, and missingness semantics.
- `docs/benchmark_protocol.md`: target construction and evaluation protocol.
- `docs/benchmark_card.md`: intended use and limitations.
- `docs/rights_policy.md`: release boundary and source-specific caveats.
- `docs/huggingface_layout.md`: Hub configs, storage split, and file sharding.

