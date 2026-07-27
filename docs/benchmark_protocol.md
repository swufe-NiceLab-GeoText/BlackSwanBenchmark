# BlackSwanNewsBench Benchmark Protocol

## Prediction unit

Each sample is an `(event_id, market, ticker)` event-company tuple. Event
metadata is stored in `metadata/events.csv` and company metadata in
`metadata/companies.csv`.

## Temporal boundary

```text
prediction_time = event_start_date
feature_end_time = event_start_date - 1 calendar day
label_start_time = first available trading day on or after event_start_date
label_end_time = last available trading day on or before event_end_date
```

`data/daily_features/*.parquet`, `data/text_embeddings/*.parquet`,
`data/gdelt_firm/*.parquet`, and `data/gdelt_market/*.parquet` contain only
pre-event model inputs. Event-window outcomes are isolated under
`data/outcomes/*.parquet`.

## Targets

The public outcome table stores `event_return` and `label_available_flag`.
Abnormal returns are constructed per split/fold by the benchmark runtime. The
T1 high-impact threshold is the training split's Q90 of absolute abnormal
returns; no validation or test label is used to fit this threshold.

- T1: high-impact classification.
- T2: signed abnormal-return regression.
- T3: impact ranking with `T3_prediction = abs(T2_prediction)`, evaluated
  within each `event_id + market` group.

## Evaluation definitions

- `random_firm_split_seed42.json`: firm-disjoint in-domain evaluation.
- `leave_one_event_out.json`: Leave-One-Event-Out (LOEO).
- `leave_one_market_out.json`: Leave-One-Market-Out (LOMO).
- `default_chronological.json`: chronological deployment-style definition.

The Hugging Face `full` split is only a storage split. It must not replace these
benchmark definitions.

## Missingness

Source unavailable, source available with zero records, and source available
with observed records are distinct states. Preprocessing parameters and all
target-derived quantities must be fit on the training partition only.
