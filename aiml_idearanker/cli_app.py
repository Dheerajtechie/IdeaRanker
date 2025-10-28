import argparse
import os

from .cli_train import run_train
from .cli_predict import run_predict
from .cli_merge import run_merge
from .cli_pricing import run_pricing
from .cli_report import run_report


def main() -> None:
	p = argparse.ArgumentParser(description="End-to-end IdeaRanker app runner")
	p.add_argument("--data", default="aiml_idearanker/sample_data.csv")
	p.add_argument("--inference", default="aiml_idearanker/sample_inference.csv")
	p.add_argument("--artifacts", default="artifacts")
	p.add_argument("--unit_cost", type=float, default=1.0)
	p.add_argument("--lr", type=float, default=0.1)
	p.add_argument("--epochs", type=int, default=300)
	p.add_argument("--l2", type=float, default=0.0)
	args = p.parse_args()

	os.makedirs(args.artifacts, exist_ok=True)
	model_path = os.path.join(args.artifacts, "model.json")
	pred_path = os.path.join(args.artifacts, "predictions.csv")
	merged_path = os.path.join(args.artifacts, "predictions_with_users.csv")
	pricing_path = os.path.join(args.artifacts, "pricing_report.csv")
	report_path = os.path.join(args.artifacts, "product_brief.txt")

	# Train
	_run_train = argparse.Namespace(data=args.data, model=model_path, lr=args.lr, epochs=args.epochs, l2=args.l2, threshold=0.5)
	run_train(_run_train)

	# Predict
	_run_predict = argparse.Namespace(model=model_path, input=args.inference, output=pred_path)
	run_predict(_run_predict)

	# Merge for pricing
	_run_merge = argparse.Namespace(predictions=pred_path, inference=args.inference, output=merged_path)
	run_merge(_run_merge)

	# Pricing
	_run_pricing = argparse.Namespace(input=merged_path, output=pricing_path, unit_cost=args.unit_cost)
	run_pricing(_run_pricing)

	# Report
	_run_report = argparse.Namespace(input=merged_path, output=report_path, unit_cost=args.unit_cost)
	run_report(_run_report)

	print("End-to-end run completed.")
	print(f"Model: {model_path}")
	print(f"Predictions: {pred_path}")
	print(f"Pricing: {pricing_path}")
	print(f"Report: {report_path}")


if __name__ == "__main__":
	main()
