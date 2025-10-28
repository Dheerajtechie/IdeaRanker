import argparse
import csv


def run_merge(args: argparse.Namespace) -> None:
	# inputs: predictions.csv (prob_success), inference.csv (projected_users)
	preds = []
	with open(args.predictions, "r", encoding="utf-8", newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			preds.append(row["prob_success"])  # str keeping decimal
	users = []
	with open(args.inference, "r", encoding="utf-8", newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			users.append(row["projected_users"])  # keep as string
	rows = min(len(preds), len(users))
	with open(args.output, "w", encoding="utf-8", newline="") as f:
		w = csv.writer(f)
		w.writerow(["prob_success", "projected_users"]) 
		for i in range(rows):
			w.writerow([preds[i], users[i]])
	print(f"Wrote merged pricing input to {args.output}")


def main() -> None:
	p = argparse.ArgumentParser(description="Merge predictions with projected_users for pricing")
	p.add_argument("--predictions", required=True, help="CSV with prob_success column")
	p.add_argument("--inference", required=True, help="CSV with projected_users column")
	p.add_argument("--output", required=True, help="Output CSV path")
	args = p.parse_args()
	run_merge(args)


if __name__ == "__main__":
	main()
