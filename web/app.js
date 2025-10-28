(function(){
	"use strict";

	// Embedded default model (export from artifacts/model.json)
	const DEFAULT_MODEL = {
		"weights": [0.973151750073213,1.384131304946152,0.04812224790906328,1.6357567409915157,-1.9250741581106545,0.6332355900464715],
		"scaler_means": [0.5733333333333331,0.6333333333333333,22133.333333333332,10.6,0.5133333333333333],
		"scaler_stds": [0.2610874693788782,0.18093671611393647,19985.23264336755,3.8877095717511763,0.2393940763877081],
		"metadata": {"learning_rate":"0.1","epochs":"300","l2":"0.0"}
	};

	const CSV_TEMPLATE_HEADER = ["novelty_score","feasibility_score","projected_users","est_dev_weeks","prior_similar_success_rate"];
	const DEMO_ROWS = [
		{novelty_score:0.8, feasibility_score:0.8, projected_users:30000, est_dev_weeks:10, prior_similar_success_rate:0.7},
		{novelty_score:0.4, feasibility_score:0.6, projected_users:12000, est_dev_weeks:12, prior_similar_success_rate:0.5},
		{novelty_score:0.9, feasibility_score:0.5, projected_users:50000, est_dev_weeks:8, prior_similar_success_rate:0.4},
	];

	// Utilities
	function sigmoid(x){
		if(x>=0){ const z=Math.exp(-x); return 1/(1+z);} else { const z=Math.exp(x); return z/(1+z);} 
	}
	function standardizeRow(row, means, stds){
		const out=[]; for(let j=0;j<row.length;j++){ const s=stds[j]===0?1:stds[j]; out.push((row[j]-means[j])/s); } return out;
	}
	function predictProbaRow(weights, row){
		let z=weights[0]; for(let j=0;j<row.length;j++){ z+=weights[j+1]*row[j]; } return sigmoid(z);
	}
	function demandCurve(prob, users){
		const base=Math.max(0,Math.min(1,prob))*Math.max(0,users);
		const pts=[]; for(let i=1;i<=40;i++){ const price=i*0.5; const demand=base*Math.max(0,1-price/20); pts.push([price,demand]); }
		return pts;
	}
	function optimizeRevenue(prob, users, unitCost){
		let best={price:0,rev:0,profit:0};
		for(const [price,demand] of demandCurve(prob, users)){
			const rev=price*demand; const profit=Math.max(0,price-unitCost)*demand;
			if(rev>best.rev){ best={price:price,rev:rev,profit:profit}; }
		}
		return best;
	}
	function parseCsv(text){
		const lines=text.split(/\r?\n/).filter(l=>l.trim().length>0);
		if(lines.length===0) return {header:[], rows:[]};
		const header=lines[0].split(",").map(s=>s.trim());
		const rows=[]; for(let i=1;i<lines.length;i++){ const parts=lines[i].split(","); const obj={}; for(let j=0;j<header.length;j++){ obj[header[j]]=(parts[j]||"").trim(); } rows.push(obj);} 
		return {header, rows};
	}
	function toFloat(v){ if(v===undefined||v===null||v==="") return 0; const n=Number(v); return isNaN(n)?0:n; }
	function downloadCsv(filename, header, rows){
		const lines=[header.join(",")];
		for(const r of rows){ lines.push(header.map(h=>String(r[h]??"")).join(",")); }
		const blob=new Blob([lines.join("\n")],{type:"text/csv;charset=utf-8;"});
		const url=URL.createObjectURL(blob); const a=document.createElement("a"); a.href=url; a.download=filename; a.click(); URL.revokeObjectURL(url);
	}

	// 0/1 Knapsack for portfolio selection
	function selectPortfolio(items, maxWeeks){
		// items: [{idx, weeks, value}] maximize value under sum weeks <= maxWeeks
		const n=items.length; const W = Math.max(0, Math.floor(maxWeeks));
		const dp = Array.from({length:n+1}, ()=>Array(W+1).fill(0));
		for(let i=1;i<=n;i++){
			const wt = Math.max(0, Math.floor(items[i-1].weeks));
			const val = items[i-1].value;
			for(let w=0; w<=W; w++){
				if(wt<=w){ dp[i][w] = Math.max(dp[i-1][w], dp[i-1][w-wt] + val); }
				else { dp[i][w] = dp[i-1][w]; }
			}
		}
		// reconstruct
		let w=W; const chosen=[];
		for(let i=n;i>=1;i--){
			if(dp[i][w]!==dp[i-1][w]){ chosen.push(i-1); w -= Math.max(0, Math.floor(items[i-1].weeks)); }
		}
		chosen.reverse();
		return chosen; // indices into items
	}

	// State
	let model=null; // {weights, scaler_means, scaler_stds}
	let ideas=[]; // parsed rows
	let lastResults=[];

	// DOM
	const elModelStatus=document.getElementById("model-status");
	const elCsvStatus=document.getElementById("csv-status");
	const elRunStatus=document.getElementById("run-status");
	const elKpis=document.getElementById("kpis");
	const elTable=document.getElementById("results-table");
	const elBody=document.getElementById("results-body");
	const elPortfolio=document.getElementById("portfolio");
	const elPortfolioBody=document.getElementById("portfolio-body");
	const elPortfolioKpis=document.getElementById("portfolio-kpis");
	
	document.getElementById("btn-load-default").addEventListener("click", ()=>{
		model=DEFAULT_MODEL; elModelStatus.textContent="Embedded model loaded.";
	});
	
	document.getElementById("file-model").addEventListener("change", async (e)=>{
		const f=e.target.files && e.target.files[0]; if(!f){return;}
		try{
			const txt=await f.text(); const json=JSON.parse(txt);
			if(!json.weights||!json.scaler_means||!json.scaler_stds) throw new Error("Invalid model.json");
			model=json; elModelStatus.textContent=`Loaded model: ${f.name}`;
		}catch(err){ elModelStatus.textContent="Failed to load model.json"; console.error(err); }
	});
	
	document.getElementById("file-csv").addEventListener("change", async (e)=>{
		const f=e.target.files && e.target.files[0]; if(!f){return;}
		try{
			const txt=await f.text(); const parsed=parseCsv(txt);
			if(parsed.header.join(",").toLowerCase()!==CSV_TEMPLATE_HEADER.join(",")){
				elCsvStatus.textContent="Invalid columns. Use the template."; return;
			}
			ideas=parsed.rows;
			elCsvStatus.textContent=`Loaded ${ideas.length} rows.`;
		}catch(err){ elCsvStatus.textContent="Failed to parse CSV"; console.error(err); }
	});

	document.getElementById("btn-download-template").addEventListener("click", ()=>{
		downloadCsv("ideas_template.csv", CSV_TEMPLATE_HEADER, []);
	});

	document.getElementById("btn-load-demo").addEventListener("click", ()=>{
		ideas = DEMO_ROWS.map(r=>({
			novelty_score: String(r.novelty_score),
			feasibility_score: String(r.feasibility_score),
			projected_users: String(r.projected_users),
			est_dev_weeks: String(r.est_dev_weeks),
			prior_similar_success_rate: String(r.prior_similar_success_rate)
		}));
		elCsvStatus.textContent = `Loaded demo dataset (${ideas.length} rows).`;
	});
	
	document.getElementById("btn-run").addEventListener("click", ()=>{
		if(!model){ elRunStatus.textContent="Load a model first."; return; }
		if(!ideas || ideas.length===0){ elRunStatus.textContent="Load ideas CSV first (or demo)."; return; }
		const unitCost=toFloat(document.getElementById("unit-cost").value);
		// Build features and compute
		const results=[];
		for(const r of ideas){
			const row=[
				toFloat(r.novelty_score),
				toFloat(r.feasibility_score),
				toFloat(r.projected_users),
				toFloat(r.est_dev_weeks),
				toFloat(r.prior_similar_success_rate)
			];
			const zrow=standardizeRow(row, model.scaler_means, model.scaler_stds);
			const p=predictProbaRow(model.weights, zrow);
			const opt=optimizeRevenue(p, row[2], unitCost);
			results.push({ prob_success: p, projected_users: row[2], dev_weeks: row[3], best_price: opt.price, expected_revenue: opt.rev, expected_profit: opt.profit });
		}
		lastResults = results.slice();
		renderResults(lastResults);
		elRunStatus.textContent=`Completed: ${results.length} rows.`;
		window.__resultsRows = results.map(r=>({
			prob_success: r.prob_success.toFixed(6),
			projected_users: Math.round(r.projected_users),
			dev_weeks: Math.round(r.dev_weeks),
			best_price: r.best_price.toFixed(2),
			expected_revenue: r.expected_revenue.toFixed(2),
			expected_profit: r.expected_profit.toFixed(2)
		}));
		// hide portfolio until recalculated
		elPortfolio.classList.add("hidden");
	});

	function renderResults(results){
		// KPIs
		const probs=results.map(r=>r.prob_success);
		const pmin=probs.length?Math.min(...probs):0;
		const pmax=probs.length?Math.max(...probs):0;
		const pavg=probs.length?probs.reduce((a,b)=>a+b,0)/probs.length:0;
		const revTotal=results.reduce((a,b)=>a+b.expected_revenue,0);
		elKpis.innerHTML = `
			<div class="kpi"><b>Ideas</b><br>${results.length}</div>
			<div class="kpi"><b>Prob min/avg/max</b><br>${pmin.toFixed(3)}/${pavg.toFixed(3)}/${pmax.toFixed(3)}</div>
			<div class="kpi"><b>Total revenue</b><br>$${revTotal.toLocaleString(undefined,{minimumFractionDigits:2, maximumFractionDigits:2})}</div>
		`;
		// Table
		elBody.innerHTML="";
		results.forEach((r,idx)=>{
			const tr=document.createElement("tr");
			function td(v){ const d=document.createElement("td"); d.textContent=v; return d; }
			tr.appendChild(td(String(idx+1)));
			tr.appendChild(td(r.prob_success.toFixed(6)));
			tr.appendChild(td(String(Math.round(r.projected_users))));
			tr.appendChild(td(String(Math.round(r.dev_weeks))));
			tr.appendChild(td(`$${r.best_price.toFixed(2)}`));
			tr.appendChild(td(`$${r.expected_revenue.toFixed(2)}`));
			tr.appendChild(td(`$${r.expected_profit.toFixed(2)}`));
			elBody.appendChild(tr);
		});
		elTable.classList.remove("hidden");
	}

	document.getElementById("btn-apply-sort").addEventListener("click", ()=>{
		if(!lastResults.length){ elRunStatus.textContent="Run first."; return; }
		const key=document.getElementById("sort-key").value;
		const ord=document.getElementById("sort-order").value;
		const sorted = lastResults.slice().sort((a,b)=>{
			const va=a[key], vb=b[key];
			return ord==="asc" ? (va - vb) : (vb - va);
		});
		renderResults(sorted);
	});

	document.getElementById("btn-portfolio").addEventListener("click", ()=>{
		if(!lastResults.length){ elRunStatus.textContent="Run first."; return; }
		const maxWeeks = toFloat(document.getElementById("max-weeks").value);
		const items = lastResults.map((r,idx)=>({ idx, weeks: r.dev_weeks, value: r.expected_revenue }));
		const chosenIdx = selectPortfolio(items, maxWeeks);
		const chosen = chosenIdx.map(i=>lastResults[i]);
		// render portfolio
		const totalWeeks = chosen.reduce((a,b)=>a+b.dev_weeks,0);
		const totalRevenue = chosen.reduce((a,b)=>a+b.expected_revenue,0);
		const totalProfit = chosen.reduce((a,b)=>a+b.expected_profit,0);
		elPortfolioKpis.innerHTML = `
			<div class="kpi"><b>Selected</b><br>${chosen.length} ideas</div>
			<div class="kpi"><b>Total weeks</b><br>${Math.round(totalWeeks)}</div>
			<div class="kpi"><b>Total revenue</b><br>$${totalRevenue.toLocaleString(undefined,{minimumFractionDigits:2, maximumFractionDigits:2})}</div>
			<div class="kpi"><b>Total profit</b><br>$${totalProfit.toLocaleString(undefined,{minimumFractionDigits:2, maximumFractionDigits:2})}</div>
		`;
		elPortfolioBody.innerHTML="";
		chosen.forEach((r,idx)=>{
			const tr=document.createElement("tr");
			function td(v){ const d=document.createElement("td"); d.textContent=v; return d; }
			tr.appendChild(td(String(idx+1)));
			tr.appendChild(td(r.prob_success.toFixed(6)));
			tr.appendChild(td(String(Math.round(r.projected_users))));
			tr.appendChild(td(String(Math.round(r.dev_weeks))));
			tr.appendChild(td(`$${r.best_price.toFixed(2)}`));
			tr.appendChild(td(`$${r.expected_revenue.toFixed(2)}`));
			tr.appendChild(td(`$${r.expected_profit.toFixed(2)}`));
			elPortfolioBody.appendChild(tr);
		});
		elPortfolio.classList.remove("hidden");
	});

	document.getElementById("btn-download").addEventListener("click", ()=>{
		const rows = window.__resultsRows || [];
		if(!rows.length){ elRunStatus.textContent="Nothing to download. Run first."; return; }
		downloadCsv("pricing_report.csv", ["prob_success","projected_users","dev_weeks","best_price","expected_revenue","expected_profit"], rows);
	});

	// Auto boot: load model + demo + run
	document.addEventListener("DOMContentLoaded", ()=>{
		try{
			model = DEFAULT_MODEL; elModelStatus.textContent = "Embedded model loaded.";
			ideas = DEMO_ROWS.map(r=>({
				novelty_score: String(r.novelty_score),
				feasibility_score: String(r.feasibility_score),
				projected_users: String(r.projected_users),
				est_dev_weeks: String(r.est_dev_weeks),
				prior_similar_success_rate: String(r.prior_similar_success_rate)
			}));
			elCsvStatus.textContent = `Loaded demo dataset (${ideas.length} rows).`;
			document.getElementById("btn-run").click();
		}catch(e){ console.error(e); }
	});
})();
