import argparse
import csv
from typing import List, Tuple

from .pricing import optimize_revenue


def summarize(probabilities: List[float]) -> Tuple[float, float, float]:
	if not probabilities:
		return 0.0, 0.0, 0.0
	pmin = min(probabilities)
	pmax = max(probabilities)
	pavg = sum(probabilities) / len(probabilities)
	return pmin, pmax, pavg


def run_report(args: argparse.Namespace) -> None:
	rows = []
	with open(args.input, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for r in reader:
			rows.append(r)
	probs: List[float] = []
	best_prices: List[float] = []
	revenues: List[float] = []
	profits: List[float] = []
	for r in rows:
		p = float(r.get("prob_success", 0.0) or 0.0)
		u = float(r.get("projected_users", 0.0) or 0.0)
		probs.append(p)
		res = optimize_revenue(p, u, unit_cost=args.unit_cost)
		best_prices.append(res["best_price"])
		revenues.append(res["expected_revenue"])
		profits.append(res["expected_profit"])
	pmin, pmax, pavg = summarize(probs)
	rev_total = sum(revenues)
	profit_total = sum(profits)
	with open(args.output, "w", encoding="utf-8") as f:
		f.write("IdeaRanker Product Brief\n")
		f.write("=======================\n\n")
		f.write(f"Ideas scored: {len(probs)}\n")
		f.write(f"Success probability: min={pmin:.3f} avg={pavg:.3f} max={pmax:.3f}\n")
		f.write(f"Total expected revenue (naive curve): ${rev_total:,.2f}\n")
		f.write(f"Total expected profit (unit_cost={args.unit_cost}): ${profit_total:,.2f}\n\n")
		# Recommendation heuristics
		f.write("Recommendations\n")
		f.write("- Prioritize ideas with prob_success >= 0.7\n")
		f.write("- Deprioritize ideas with development time > 16 weeks unless prob_success >= 0.85\n")
		f.write("- Use best_price as a starting point; validate with A/B tests and user research\n")
	print(f"Wrote report to {args.output}")


def main() -> None:
	p = argparse.ArgumentParser(description="Generate product brief with pricing insights")
	p.add_argument("--input", required=True, help="CSV with prob_success, projected_users")
	p.add_argument("--output", required=True, help="Output .txt report path")
	p.add_argument("--unit_cost", type=float, default=0.0)
	args = p.parse_args()
	run_report(args)


if __name__ == "__main__":
	main()
