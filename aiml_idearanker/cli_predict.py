import argparse
import csv
from typing import Dict, List

from .data import build_features, load_csv
from .model import IdeaRankerModel
from .utils import load_json


def run_predict(args: argparse.Namespace) -> None:
	payload = load_json(args.model)
	model = IdeaRankerModel.from_dict(payload)
	rows = load_csv(args.input)
	X, _ = build_features(rows)
	probs = model.predict_proba(X)
	with open(args.output, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["prob_success"]) 
		for p in probs:
			writer.writerow([f"{p:.6f}"])
	print(f"Wrote predictions to {args.output}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict success probability for ideas")
	parser.add_argument("--model", required=True, help="Path to model JSON")
	parser.add_argument("--input", required=True, help="Path to inference CSV")
	parser.add_argument("--output", required=True, help="Path to output predictions CSV")
	args = parser.parse_args()
	run_predict(args)


if __name__ == "__main__":
	main()
