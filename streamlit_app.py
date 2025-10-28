import io
import json
import csv
from typing import List, Dict

import streamlit as st
import pandas as pd
import altair as alt

from aiml_idearanker.data import build_features, FEATURE_COLUMNS
from aiml_idearanker.model import IdeaRankerModel
from aiml_idearanker.pricing import optimize_revenue
from aiml_idearanker.utils import load_json


st.set_page_config(page_title="IdeaRanker", page_icon="💡", layout="wide")
st.title("IdeaRanker – Streamlit")
st.caption("Upload real ideas, score success probability, price, and plan an optimal portfolio.")

# Sidebar controls
st.sidebar.header("Configuration")
unit_cost = st.sidebar.number_input("Unit cost", min_value=0.0, value=1.0, step=0.1)
max_weeks = st.sidebar.number_input("Max dev weeks (portfolio)", min_value=1, value=24, step=1)

# Model loader
st.subheader("1) Load Model")
model_file = st.file_uploader("Upload model.json (optional)", type=["json"], key="model")
model: IdeaRankerModel
if model_file is not None:
	try:
		payload = json.loads(model_file.getvalue().decode("utf-8"))
		model = IdeaRankerModel.from_dict(payload)
		st.success("Model loaded from upload.")
	except Exception as e:
		st.error(f"Failed to load model: {e}")
else:
	# Try artifacts/model.json as a default
	try:
		payload = load_json("artifacts/model.json")
		model = IdeaRankerModel.from_dict(payload)
		st.info("Using default artifacts/model.json")
	except Exception:
		model = IdeaRankerModel()  # not trained; will error on predict if used
		st.warning("No model provided. Upload model.json or train with CLI first.")

# CSV ideas loader
st.subheader("2) Load Ideas CSV")
left, right = st.columns(2)
with left:
	st.caption("Required columns (or map your columns): \n" + ", ".join(FEATURE_COLUMNS))
ideas_file = st.file_uploader("Upload ideas CSV", type=["csv"], key="ideas")

rows: List[Dict[str, str]] = []
df_raw: pd.DataFrame | None = None
if ideas_file is not None:
	try:
		content = ideas_file.getvalue().decode("utf-8")
		df_raw = pd.read_csv(io.StringIO(content))
		st.success(f"Loaded {len(df_raw)} rows.")
	except Exception as e:
		st.error(f"Failed to parse CSV: {e}")

# Schema mapping
mapping = {}
if df_raw is not None:
	st.subheader("3) Map Columns (if needed)")
	cols = list(df_raw.columns)
	m1, m2, m3 = st.columns(3)
	with m1:
		mapping["novelty_score"] = st.selectbox("novelty_score", cols, index=cols.index("novelty_score") if "novelty_score" in cols else 0)
		mapping["projected_users"] = st.selectbox("projected_users", cols, index=cols.index("projected_users") if "projected_users" in cols else 0)
	with m2:
		mapping["feasibility_score"] = st.selectbox("feasibility_score", cols, index=cols.index("feasibility_score") if "feasibility_score" in cols else 0)
		mapping["est_dev_weeks"] = st.selectbox("est_dev_weeks", cols, index=cols.index("est_dev_weeks") if "est_dev_weeks" in cols else 0)
	with m3:
		mapping["prior_similar_success_rate"] = st.selectbox("prior_similar_success_rate", cols, index=cols.index("prior_similar_success_rate") if "prior_similar_success_rate" in cols else 0)

	# Apply mapping
	rows = []
	for _, r in df_raw.iterrows():
		rows.append({
			"novelty_score": r.get(mapping["novelty_score"], 0),
			"feasibility_score": r.get(mapping["feasibility_score"], 0),
			"projected_users": r.get(mapping["projected_users"], 0),
			"est_dev_weeks": r.get(mapping["est_dev_weeks"], 0),
			"prior_similar_success_rate": r.get(mapping["prior_similar_success_rate"], 0),
		})

# Filters
st.subheader("4) Filters & Run")
f1, f2, f3, f4 = st.columns(4)
with f1:
	min_users = st.number_input("Min projected users", min_value=0, value=0, step=100)
with f2:
	min_prob = st.number_input("Min probability threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
with f3:
	max_weeks_filter = st.number_input("Max dev weeks per idea", min_value=0, value=0, step=1)
with f4:
	_sort_key = st.selectbox("Sort by", ["expected_revenue","prob_success","expected_profit"]) 

run = st.button("Predict + Price", type="primary")

if run:
	if not rows:
		st.error("Upload and map a CSV first.")
	elif not getattr(model, "weights", None):
		st.error("Load a trained model.json first.")
	else:
		X, _ = build_features(rows)
		probs = model.predict_proba(X)
		result_rows: List[Dict[str, object]] = []
		for i, (p, r) in enumerate(zip(probs, rows)):
			projected_users = float(r.get("projected_users", 0) or 0)
			est_weeks = float(r.get("est_dev_weeks", 0) or 0)
			opt = optimize_revenue(p, projected_users, unit_cost=unit_cost)
			result_rows.append({
				"#": i + 1,
				"prob_success": round(p, 6),
				"projected_users": int(projected_users),
				"dev_weeks": int(est_weeks),
				"best_price": round(opt["best_price"], 2),
				"expected_revenue": round(opt["expected_revenue"], 2),
				"expected_profit": round(opt["expected_profit"], 2),
			})
		# Apply filters
		if min_users > 0:
			result_rows = [r for r in result_rows if r["projected_users"] >= min_users]
		if min_prob > 0:
			result_rows = [r for r in result_rows if r["prob_success"] >= min_prob]
		if max_weeks_filter > 0:
			result_rows = [r for r in result_rows if r["dev_weeks"] <= max_weeks_filter]
		# Sort
		result_rows.sort(key=lambda r: r[_sort_key], reverse=True)

		# KPIs
		probs_list = [r["prob_success"] for r in result_rows]
		pmin = min(probs_list) if probs_list else 0.0
		pavg = sum(probs_list) / len(probs_list) if probs_list else 0.0
		pmax = max(probs_list) if probs_list else 0.0
		rev_total = sum(r["expected_revenue"] for r in result_rows)
		profit_total = sum(r["expected_profit"] for r in result_rows)

		c1, c2, c3, c4 = st.columns(4)
		c1.metric("Ideas", len(result_rows))
		c2.metric("Prob min/avg/max", f"{pmin:.3f}/{pavg:.3f}/{pmax:.3f}")
		c3.metric("Total revenue", f"${rev_total:,.2f}")
		c4.metric("Total profit", f"${profit_total:,.2f}")

		# Charts
		st.subheader("Distributions")
		df_res = pd.DataFrame(result_rows)
		chart_prob = alt.Chart(df_res).mark_bar().encode(x=alt.X("prob_success:Q", bin=True), y="count()", tooltip=["count()"]).properties(height=180)
		chart_rev = alt.Chart(df_res).mark_bar(color="#0b5fff").encode(x=alt.X("expected_revenue:Q", bin=True), y="count()", tooltip=["count()"]).properties(height=180)
		st.altair_chart(chart_prob | chart_rev, use_container_width=True)

		st.subheader("Results Table")
		st.dataframe(result_rows, use_container_width=True, hide_index=True)

		# Portfolio optimization
		st.subheader("Recommended Portfolio")
		def select_portfolio(items: List[Dict[str, float]], max_weeks: int) -> List[int]:
			W = max(0, int(max_weeks))
			n = len(items)
			dp = [[0.0] * (W + 1) for _ in range(n + 1)]
			for i in range(1, n + 1):
				wt = max(0, int(items[i - 1]["weeks"]))
				val = float(items[i - 1]["value"])
				for w in range(W + 1):
					if wt <= w:
						dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - wt] + val)
					else:
						dp[i][w] = dp[i - 1][w]
			w = W
			chosen: List[int] = []
			for i in range(n, 0, -1):
				if dp[i][w] != dp[i - 1][w]:
					chosen.append(i - 1)
					w -= max(0, int(items[i - 1]["weeks"]))
			chosen.reverse()
			return chosen

		items = [{"weeks": r["dev_weeks"], "value": r["expected_revenue"]} for r in result_rows]
		idxs = select_portfolio(items, int(max_weeks))
		portfolio = [result_rows[i] for i in idxs]

		p_weeks = sum(r["dev_weeks"] for r in portfolio)
		p_rev = sum(r["expected_revenue"] for r in portfolio)
		p_profit = sum(r["expected_profit"] for r in portfolio)

		pc1, pc2, pc3, pc4 = st.columns(4)
		pc1.metric("Selected", len(portfolio))
		pc2.metric("Total weeks", f"{p_weeks}")
		pc3.metric("Total revenue", f"${p_rev:,.2f}")
		pc4.metric("Total profit", f"${p_profit:,.2f}")

		st.dataframe(portfolio, use_container_width=True, hide_index=True)

		# Exports
		out = io.StringIO()
		writer = csv.writer(out)
		writer.writerow(["prob_success","projected_users","dev_weeks","best_price","expected_revenue","expected_profit"]) 
		for r in result_rows:
			writer.writerow([r["prob_success"], r["projected_users"], r["dev_weeks"], f"{r['best_price']:.2f}", f"{r['expected_revenue']:.2f}", f"{r['expected_profit']:.2f}"])
		st.download_button("Download pricing_report.csv", data=out.getvalue(), file_name="pricing_report.csv", mime="text/csv")

		outp = io.StringIO()
		writerp = csv.writer(outp)
		writerp.writerow(["prob_success","projected_users","dev_weeks","best_price","expected_revenue","expected_profit"]) 
		for r in portfolio:
			writerp.writerow([r["prob_success"], r["projected_users"], r["dev_weeks"], f"{r['best_price']:.2f}", f"{r['expected_revenue']:.2f}", f"{r['expected_profit']:.2f}"])
		st.download_button("Download portfolio.csv", data=outp.getvalue(), file_name="portfolio.csv", mime="text/csv")
