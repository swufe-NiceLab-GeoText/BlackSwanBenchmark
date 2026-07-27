# Rights Policy

## Release Principle

BlackSwanNewsBench public artifacts must be reproducible without redistributing
unauthorized full-text news. As of 2026-05-14, the project/data owner has
confirmed that the frozen local/main and Tushare source-news title, summary,
and body fields may be published for this benchmark. This assertion is recorded
as `project_confirmed_publishable_2026-05-14`; it is not an external legal
opinion.

## Allowed in Public Artifact

- Schemas and data dictionaries.
- Source inventory and dataset manifest.
- Event and company metadata.
- Daily OHLCV price rows in `data/price/*.parquet`.
- Local/main and Tushare source-news title and content fields in
  `data/news/*.parquet`, under the project
  publication-rights assertion above.
- Aggregated GDELT firm-day and market-day features.
- Split JSON files.
- Benchmark code and model code.
- Event-market visualization views.

## Restricted or Excluded

- Any news text outside the frozen source-news artifact or without a recorded
  publication-rights assertion.
- Private API keys, access tokens, credentials, local absolute paths, and private
  credentials.
- GDELT `Article` generated stubs as if they were article bodies, real article
  titles, or third-party full text.

## GDELT

GDELT is used as public metadata: URL, source, date, language, tone, themes,
entities, counts, and derived aggregate signals. Local GDELT fields named
`Article`, `Article_title`, and `summary` are generated matching stubs, not
third-party article bodies. They may be released only under renamed fields such
as `gdelt_generated_stub`, `gdelt_generated_match_title`, and
`gdelt_generated_match_summary`, with GDELT citation. Text experiments using
these fields must be described as generated-stub baselines, not full-text news
models.

The official aggregate GDELT features are split into firm-day company exposure
and market-day context tables. These tables use metadata-derived counts, tone,
source, theme, and entity features only; they do not redistribute third-party
article bodies.

## Local News Sources

Local/main and Tushare source-news text is publishable in this benchmark under
the project-confirmed rights assertion. The published table is
`data/news/*.parquet` and contains only
`source_news_id, event_id, market, ticker, company_name, date, source_name,
url_hash, title, content`.

## Future Text Baselines

Public P1 text baselines may use `source_news_text.parquet`. GDELT generated
stubs are not part of the current public release tables and must not be
described as third-party full-text news.
