# IdeaRanker: Pure-Python AIML Project (No External Dependencies)

IdeaRanker is a complete, dependency-free AI/ML project implemented using only the Python standard library. It demonstrates an end-to-end product development workflow aligned with practical business impact:

- Problem framing: Rank product feature ideas by likelihood of success
- Data: Lightweight synthetic dataset with realistic signals
- Modeling: From-scratch Logistic Regression with feature scaling
- Evaluation: Train/validation split and K-Fold cross-validation
- Inference: CLI scoring for new ideas
- Pricing: Simple price optimization and business case modeling

This project is designed to showcase product thinking, ML fundamentals, business modeling, and strong engineering craft without relying on third-party packages.

## Quickstart

Requirements: Python 3.10+

```bash
# From the repository root
python -m aiml_idearanker.cli_train --data aiml_idearanker/sample_data.csv --model artifacts/model.json

# Predict on new ideas (CSV with the same feature columns, minus label)
python -m aiml_idearanker.cli_predict --model artifacts/model.json --input aiml_idearanker/sample_inference.csv --output artifacts/predictions.csv

# Price optimization example
python -m aiml_idearanker.cli_pricing --input artifacts/predictions.csv --output artifacts/pricing_report.csv
```

Artifacts will be written to the `artifacts/` directory.

## Data Schema

Training CSV (`sample_data.csv`) columns:
- novelty_score: float in [0, 1]
- feasibility_score: float in [0, 1]
- projected_users: integer (potential monthly active users)
- est_dev_weeks: integer (development weeks)
- prior_similar_success_rate: float in [0, 1]
- label: 0/1 (success ground truth)

Inference CSV (`sample_inference.csv`) columns (no label):
- novelty_score, feasibility_score, projected_users, est_dev_weeks, prior_similar_success_rate

## What this demonstrates
- End-to-end product lifecycle: data ingestion → modeling → evaluation → inference → monetization
- Clean code, documentation, and CLIs suitable for productionization
- No external dependencies, portable and easy to run anywhere

## Implementation Details
- Logistic Regression trained with batch gradient descent and L2 regularization
- Standardization per feature (mean/variance from training only; persisted with the model)
- K-Fold cross-validation with shuffled folds and reproducible seed
- Metrics: accuracy, precision, recall, F1
- Pricing: simple price-demand curve derived from model confidence and user scale; solves for price that maximizes revenue (or profit if cost is provided)

## Project Structure
```
aiml_idearanker/
  __init__.py
  data.py
  model.py
  metrics.py
  cv.py
  pricing.py
  utils.py
  cli_train.py
  cli_predict.py
  cli_pricing.py
  sample_data.csv
  sample_inference.csv
artifacts/  (created at runtime)
```

## Reproducibility
- Fixed random seed used for data splits and weight initialization
- Model, scaler params, and metadata persisted as JSON

## Notes
- The dataset is synthetic and intended for demonstration. Replace with your own data as needed.
- For larger datasets, you can stream rows from disk; the current implementation loads into memory for simplicity.

## License
MIT
