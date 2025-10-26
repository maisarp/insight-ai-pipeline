# Insight AI Pipeline

Insight AI Pipeline — local ML and clustering toolkit for secure, casefile-based analytics.

This repository provides a minimal, production-minded skeleton for an AI engineering project that:
- accepts one or many "fichas" (casefiles) as CSV/JSON/XLSX or folders,
- performs validation, anonymization and feature engineering,
- trains models locally (sklearn) and persists them,
- runs inference/analysis to produce per-person insights,
- exposes a CLI entrypoint that can be packaged as an executable.

Highlights
- Package name: insight_ai_pipeline
- CLI script: insight-ai (configured in pyproject.toml)
- Main modes: train, analyze, predict
- Privacy-first: anonymization helpers included; do not version sensitive data.
