import argparse
from typing import Dict, List

from .data import build_features, load_csv
from .metrics import accuracy, precision_recall_f1, threshold_predictions
from .model import IdeaRankerModel
from .utils import save_json


def run_train(args: argparse.Namespace) -> None:
	rows = load_csv(args.data)
	X, y = build_features(rows)
	model = IdeaRankerModel()
	model.metadata = {
		"learning_rate": str(args.lr),
		"epochs": str(args.epochs),
		"l2": str(args.l2),
		"data": args.data,
	}
	model.fit(X, y, lr=args.lr, epochs=args.epochs, l2=args.l2)
	probs = model.predict_proba(X)
	y_pred = threshold_predictions(probs, args.threshold)
	acc = accuracy(y, y_pred)
	p, r, f1 = precision_recall_f1(y, y_pred)
	print(f"Train metrics | acc={acc:.3f} p={p:.3f} r={r:.3f} f1={f1:.3f}")
	payload: Dict[str, object] = model.to_dict()
	save_json(args.model, payload)
	print(f"Saved model to {args.model}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Train IdeaRanker model (pure Python)")
	parser.add_argument("--data", required=True, help="Path to training CSV")
	parser.add_argument("--model", required=True, help="Output path for model JSON")
	parser.add_argument("--lr", type=float, default=0.1)
	parser.add_argument("--epochs", type=int, default=300)
	parser.add_argument("--l2", type=float, default=0.0)
	parser.add_argument("--threshold", type=float, default=0.5)
	args = parser.parse_args()
	run_train(args)


if __name__ == "__main__":
	main()
