from typing import Dict, List, Tuple


def demand_curve(prob_success: float, projected_users: float) -> List[Tuple[float, float]]:
	# Returns list of (price, expected_demand) pairs
	# Simple decreasing demand as price increases; scale by prob_success and user base
	base = max(0.0, min(1.0, prob_success)) * max(0.0, projected_users)
	points: List[Tuple[float, float]] = []
	for price in [i * 0.5 for i in range(1, 41)]:  # $0.5 to $20
		demand = base * max(0.0, 1.0 - price / 20.0)
		points.append((price, demand))
	return points


def optimize_revenue(prob_success: float, projected_users: float, unit_cost: float = 0.0) -> Dict[str, float]:
	best_price = 0.0
	best_revenue = 0.0
	best_profit = 0.0
	for price, demand in demand_curve(prob_success, projected_users):
		revenue = price * demand
		profit = max(0.0, price - unit_cost) * demand
		if revenue > best_revenue:
			best_revenue = revenue
			best_price = price
			best_profit = profit
	return {
		"best_price": best_price,
		"expected_revenue": best_revenue,
		"expected_profit": best_profit,
	}
