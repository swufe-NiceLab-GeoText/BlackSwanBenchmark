# BlackSwanNewsBench Benchmark Card

## Purpose

BlackSwanNewsBench is a multimodal, multi-market benchmark for pre-event
corporate impact prediction under major black-swan shocks. The benchmark is
designed for KDD Datasets & Benchmarks style evaluation: fixed data boundaries,
fixed splits, fixed metrics, public audit artifacts, and explicit source-rights
constraints.

## Prediction Unit

Each sample is an `(event_id, market, ticker)` event-company pair. Inputs are
restricted to rows before `event_start_date`; event-window returns are labels
only.

## Tasks And Metrics

The benchmark reports only the existing six primary metrics:

- T1 impact classification: AUPRC, AUROC.
- T2 abnormal return regression: MAE, Spearman.
- T3 event-market impact ranking: NDCG@10%, Spearman.

T3 predictions are always `abs(T2_prediction)` and T3 metrics are grouped by
`event_id + market`.

## Main Evaluation Views

- `RandomFirm`: in-domain firm-disjoint generalization.
- `leave_one_event_out`: cross-event OOD generalization.
- `leave_one_market_out`: cross-market OOD generalization.
- `default_chronological`: deployment-like split definition. It is not reported
  as completed unless a coverage-gated result table exists.

## Labels

The main label protocol is `split_part_market_return_v2`, which computes
abnormal returns relative to the event-market mean inside each split part.
Runtime high-impact labels are generated from train-split-only Q90 thresholds;
base label files must not materialize `high_impact_label`.

The optional robustness protocol `leave_one_out_market_return_v1` is available
for label sensitivity audits. It is not mixed into completed main leaderboards.

## Baseline Reporting Policy

Official-source models must preserve the upstream body/backbone. If a benchmark
head or supervised objective is adapted for BlackSwanNewsBench labels, the model
is reported as an `architecture-preserved adapted baseline`. Blocked official
models are excluded from completed main leaderboards.

SwanHead is reported as a local proposed reference model. Its current final
protocol keeps one architecture but uses task-specific validation search for
T1/T2/T3 hyperparameters.

## Public Release Boundary

The release can include schemas, split files, metric scripts, source inventory,
derived structured features, GDELT metadata features, BGE-M3 embeddings, hashes,
and rebuild instructions. Restricted raw news text is not redistributed unless
the source-level rights audit permits it.

## Known Limitations

- Current v1 data has four events and four markets; event-family expansion is a
  future benchmark update, not a completed result.
- Source availability differs by market, event, language, and provider.
- The benchmark is for reproducible research, not investment advice or live
  trading.
