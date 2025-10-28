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

# Theme switcher
theme = st.sidebar.selectbox("Theme", ["Light","Dark"], index=0)
if theme == "Light":
	primary, bg, sub, text, muted = ("#0b5fff", "#ffffff", "#f6f8fa", "#111", "#667085")
else:
	primary, bg, sub, text, muted = ("#74b0ff", "#0b1220", "#111827", "#f5f7fa", "#9aa4b2")

# Custom CSS
st.markdown(
	f"""
	<style>
		:root {{ --primary:{primary}; --bg:{bg}; --sub:{sub}; --text:{text}; --muted:{muted}; }}
		html, body, .block-container {{ background: var(--bg); color: var(--text); }}
		header[role="banner"] {{ background: linear-gradient(90deg, var(--primary) 0%, #8bc6ff 100%); color:#fff; }}
		.block-container {{ padding-top: 1.0rem; }}
		.stButton>button {{ background: var(--primary); color:#fff; border:0; padding:0.6rem 1rem; border-radius:10px; transition: transform .06s ease; }}
		.stButton>button:hover {{ filter:brightness(0.95); transform: translateY(-1px); }}
		.kpi {{ background: var(--sub); border:1px solid rgba(255,255,255,.08); padding:14px 16px; border-radius:12px; }}
		.small {{ color: var(--muted); font-size: 0.9rem; }}
		.card {{ background: var(--sub); border:1px solid rgba(255,255,255,.08); padding:16px; border-radius:12px; }}
		footer.fixed {{ position: sticky; bottom: 0; background: var(--sub); border-top:1px solid rgba(255,255,255,.08); padding:8px 12px; border-radius:12px 12px 0 0; }}
		h1, h2, h3 {{ letter-spacing:0.2px }}
	</style>
	""",
	unsafe_allow_html=True,
)

st.title("💡 IdeaRanker")
st.caption("Upload, score, price, and plan your roadmap with portfolio optimization.")

# Sidebar controls
st.sidebar.header("Configuration")
unit_cost = st.sidebar.number_input("Unit cost", min_value=0.0, value=1.0, step=0.1)
max_weeks = st.sidebar.number_input("Max dev weeks (portfolio)", min_value=1, value=24, step=1)

# Tabs for workflow
tab1, tab2, tab3 = st.tabs(["Data & Model", "Results", "Portfolio"])

with tab1:
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
		try:
			payload = load_json("artifacts/model.json")
			model = IdeaRankerModel.from_dict(payload)
			st.info("Using default artifacts/model.json")
		except Exception:
			model = IdeaRankerModel()
			st.warning("No model provided. Upload model.json or train with CLI first.")

	st.subheader("2) Load Ideas CSV")
	st.caption("Required columns (or map your columns): \n" + ", ".join(FEATURE_COLUMNS))
	ideas_file = st.file_uploader("Upload ideas CSV", type=["csv"], key="ideas")

	raws: List[Dict[str, str]] = []
	df_raw: pd.DataFrame | None = None
	if ideas_file is not None:
		try:
			content = ideas_file.getvalue().decode("utf-8")
			df_raw = pd.read_csv(io.StringIO(content))
			st.success(f"Loaded {len(df_raw)} rows.")
		except Exception as e:
			st.error(f"Failed to parse CSV: {e}")

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

		raws = []
		for _, r in df_raw.iterrows():
			raws.append({
				"novelty_score": r.get(mapping["novelty_score"], 0),
				"feasibility_score": r.get(mapping["feasibility_score"], 0),
				"projected_users": r.get(mapping["projected_users"], 0),
				"est_dev_weeks": r.get(mapping["est_dev_weeks"], 0),
				"prior_similar_success_rate": r.get(mapping["prior_similar_success_rate"], 0),
			})

	st.subheader("4) Filters")
	f1, f2, f3 = st.columns(3)
	with f1:
		min_users = st.number_input("Min projected users", min_value=0, value=0, step=100)
	with f2:
		min_prob = st.number_input("Min probability", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
	with f3:
		max_weeks_filter = st.number_input("Max dev weeks per idea", min_value=0, value=0, step=1)

	run = st.button("Predict + Price", type="primary")

with tab2:
	st.subheader("Results & Insights")
	if 'results' not in st.session_state:
		st.info("Run predictions in the 'Data & Model' tab.")
	else:
		result_rows = st.session_state['results']
		probs_list = [r["prob_success"] for r in result_rows]
		pmin = min(probs_list) if probs_list else 0.0
		pavg = sum(probs_list) / len(probs_list) if probs_list else 0.0
		pmax = max(probs_list) if probs_list else 0.0
		rev_total = sum(r["expected_revenue"] for r in result_rows)
		profit_total = sum(r["expected_profit"] for r in result_rows)
		c1, c2, c3, c4 = st.columns(4)
		c1.markdown(f"<div class='kpi'><b>Ideas</b><br>{len(result_rows)}</div>", unsafe_allow_html=True)
		c2.markdown(f"<div class='kpi'><b>Prob min/avg/max</b><br>{pmin:.3f}/{pavg:.3f}/{pmax:.3f}</div>", unsafe_allow_html=True)
		c3.markdown(f"<div class='kpi'><b>Total revenue</b><br>${rev_total:,.2f}</div>", unsafe_allow_html=True)
		c4.markdown(f"<div class='kpi'><b>Total profit</b><br>${profit_total:,.2f}</div>", unsafe_allow_html=True)
		st.divider()
		df_res = pd.DataFrame(result_rows)
		chart_prob = alt.Chart(df_res).mark_bar().encode(x=alt.X("prob_success:Q", bin=True), y="count()").properties(height=180)
		chart_rev = alt.Chart(df_res).mark_bar(color=primary).encode(x=alt.X("expected_revenue:Q", bin=True), y="count()").properties(height=180)
		st.altair_chart(chart_prob | chart_rev, use_container_width=True)
		st.dataframe(result_rows, use_container_width=True, hide_index=True)

with tab3:
	st.subheader("Recommended Portfolio")
	if 'results' not in st.session_state:
		st.info("Run predictions first.")
	else:
		result_rows = st.session_state['results']
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
		c1, c2, c3, c4 = st.columns(4)
		c1.markdown(f"<div class='kpi'><b>Selected</b><br>{len(portfolio)} ideas</div>", unsafe_allow_html=True)
		c2.markdown(f"<div class='kpi'><b>Total weeks</b><br>{int(p_weeks)}</div>", unsafe_allow_html=True)
		c3.markdown(f"<div class='kpi'><b>Total revenue</b><br>${p_rev:,.2f}</div>", unsafe_allow_html=True)
		c4.markdown(f"<div class='kpi'><b>Total profit</b><br>${p_profit:,.2f}</div>", unsafe_allow_html=True)
		st.dataframe(portfolio, use_container_width=True, hide_index=True)

# Run pipeline when clicked in tab1
if 'results' not in st.session_state:
	st.session_state['results'] = None
if 'run' not in st.session_state:
	st.session_state['run'] = False

if 'raws' not in locals():
	raws = []

if run and raws:
	X, _ = build_features(raws)
	probs = model.predict_proba(X)
	result_rows: List[Dict[str, object]] = []
	for i, (p, r) in enumerate(zip(probs, raws)):
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
	# store and filter/sort
	if run:
		res = result_rows
		if min_users > 0:
			res = [r for r in res if r["projected_users"] >= min_users]
		if min_prob > 0:
			res = [r for r in res if r["prob_success"] >= min_prob]
		if max_weeks_filter > 0:
			res = [r for r in res if r["dev_weeks"] <= max_weeks_filter]
		st.session_state['results'] = res

st.markdown("<footer class='fixed small'>IdeaRanker • Built for product leaders to ship the right ideas</footer>", unsafe_allow_html=True)
