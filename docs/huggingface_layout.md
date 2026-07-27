# Hugging Face Release Layout

The repository follows a modality-first layout similar to existing financial
multimodal datasets, but stores typed Parquet shards instead of multi-gigabyte
CSV or opaque ZIP-only modality archives.

Each dataset config in the root `README.md` maps to one schema and one
Hugging Face storage split named `full`. Files are partitioned by market, and
the news config is additionally partitioned by event. This supports Dataset
Viewer schema inference, streaming, and selective downloads.

The `full` storage split is deliberately separate from benchmark evaluation
definitions. RandomFirm, LOEO, LOMO, and chronological protocols remain JSON
artifacts under `splits/` because LOEO and LOMO contain multiple held-out folds
rather than a single static train/test partition.
