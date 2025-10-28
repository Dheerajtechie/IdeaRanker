import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_SEED = 42


def ensure_dir(path: str) -> None:
	if not path:
		return
	os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
	random.seed(seed)


def sigmoid(x: float) -> float:
	if x >= 0:
		z = math.exp(-x)
		return 1.0 / (1.0 + z)
	else:
		z = math.exp(x)
		return z / (1.0 + z)


def dot(a: List[float], b: List[float]) -> float:
	return sum(x * y for x, y in zip(a, b))


def mean_std(values: List[float]) -> Tuple[float, float]:
	if not values:
		return 0.0, 1.0
	m = sum(values) / len(values)
	var = sum((v - m) * (v - m) for v in values) / max(1, len(values) - 1)
	std = math.sqrt(var) if var > 0 else 1.0
	return m, std


def standardize_column(values: List[float]) -> Tuple[List[float], float, float]:
	m, s = mean_std(values)
	return [((v - m) / s) for v in values], m, s


def apply_standardize(values: List[float], mean: float, std: float) -> List[float]:
	if std == 0:
		return [0.0 for _ in values]
	return [((v - mean) / std) for v in values]


def save_json(path: str, payload: Dict[str, Any]) -> None:
	dirname = os.path.dirname(path)
	if dirname:
		ensure_dir(dirname)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def argmax_index(values: List[float]) -> int:
	best_i = 0
	best_v = values[0]
	for i, v in enumerate(values):
		if v > best_v:
			best_i, best_v = i, v
	return best_i
