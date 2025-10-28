import argparse
import csv
import html
import os


def run_dashboard(args: argparse.Namespace) -> None:
	items = []
	with open(args.pricing, "r", encoding="utf-8", newline="") as f:
		r = csv.DictReader(f)
		for row in r:
			items.append(row)
	html_body = [
		"<html><head><meta charset='utf-8'><title>IdeaRanker Dashboard</title>",
		"<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px} table{border-collapse:collapse;width:100%} th,td{border:1px solid #ddd;padding:8px} th{background:#f2f2f2;text-align:left} .kpi{display:flex;gap:24px;margin-bottom:16px} .kpi div{background:#fafafa;padding:12px;border:1px solid #eee;border-radius:8px}</style>",
		"</head><body>",
		"<h1>IdeaRanker Dashboard</h1>",
	]

	def safe(x: str) -> str:
		return html.escape(x)

	# KPIs
	try:
		probs = [float(i.get("prob_success", 0.0) or 0.0) for i in items]
		revs = [float(i.get("expected_revenue", 0.0) or 0.0) for i in items]
		pmin, pmax = (min(probs), max(probs)) if probs else (0.0, 0.0)
		pavg = (sum(probs) / len(probs)) if probs else 0.0
		rev_total = sum(revs)
		html_body.append(f"<div class='kpi'><div><b>Ideas</b><br>{len(items)}</div><div><b>Prob min/avg/max</b><br>{pmin:.3f}/{pavg:.3f}/{pmax:.3f}</div><div><b>Total revenue</b><br>${rev_total:,.2f}</div></div>")
	except Exception:
		pass

	html_body.append("<table><thead><tr><th>#</th><th>Prob Success</th><th>Projected Users</th><th>Best Price</th><th>Expected Revenue</th><th>Expected Profit</th></tr></thead><tbody>")
	for idx, row in enumerate(items, start=1):
		html_body.append("<tr>" + "".join([
			f"<td>{idx}</td>",
			f"<td>{safe(row.get('prob_success',''))}</td>",
			f"<td>{safe(row.get('projected_users',''))}</td>",
			f"<td>${safe(row.get('best_price',''))}</td>",
			f"<td>${safe(row.get('expected_revenue',''))}</td>",
			f"<td>${safe(row.get('expected_profit',''))}</td>",
		]) + "</tr>")
	html_body.append("</tbody></table>")
	html_body.append("</body></html>")
	os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
	with open(args.output, "w", encoding="utf-8") as f:
		f.write("".join(html_body))
	print(f"Wrote dashboard to {args.output}")


def main() -> None:
	p = argparse.ArgumentParser(description="Generate HTML dashboard from pricing report")
	p.add_argument("--pricing", required=True, help="pricing_report.csv path")
	p.add_argument("--output", required=True, help="dashboard HTML output path")
	args = p.parse_args()
	run_dashboard(args)


if __name__ == "__main__":
	main()
