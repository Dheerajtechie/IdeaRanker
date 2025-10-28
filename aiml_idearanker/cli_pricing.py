import argparse
import csv
from typing import List

from .pricing import optimize_revenue


def run_pricing(args: argparse.Namespace) -> None:
	# Expects predictions CSV with column prob_success and an auxiliary input with projected_users
	# For simplicity, we read a CSV that has both columns: prob_success, projected_users
	rows: List[dict] = []
	with open(args.input, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for r in reader:
			rows.append(r)
	with open(args.output, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["prob_success", "projected_users", "best_price", "expected_revenue", "expected_profit"]) 
		for r in rows:
			p = float(r.get("prob_success", "0") or 0.0)
			u = float(r.get("projected_users", "0") or 0.0)
			res = optimize_revenue(p, u, unit_cost=args.unit_cost)
			writer.writerow([
				f"{p:.6f}",
				int(u),
				f"{res['best_price']:.2f}",
				f"{res['expected_revenue']:.2f}",
				f"{res['expected_profit']:.2f}",
			])
	print(f"Wrote pricing report to {args.output}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Optimize pricing given prob_success and projected_users")
	parser.add_argument("--input", required=True, help="CSV with columns: prob_success, projected_users")
	parser.add_argument("--output", required=True, help="Output CSV path")
	parser.add_argument("--unit_cost", type=float, default=0.0)
	args = parser.parse_args()
	run_pricing(args)


if __name__ == "__main__":
	main()
