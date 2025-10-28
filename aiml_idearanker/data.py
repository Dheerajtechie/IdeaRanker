import csv
from typing import Dict, List, Tuple

from .utils import set_global_seed


FEATURE_COLUMNS = [
	"novelty_score",
	"feasibility_score",
	"projected_users",
	"est_dev_weeks",
	"prior_similar_success_rate",
]
LABEL_COLUMN = "label"


def load_csv(path: str) -> List[Dict[str, str]]:
	rows: List[Dict[str, str]] = []
	with open(path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append(dict(row))
	return rows


def to_float(row: Dict[str, str], key: str) -> float:
	value = row.get(key, "0").strip()
	return float(value) if value else 0.0


def to_int(row: Dict[str, str], key: str) -> int:
	value = row.get(key, "0").strip()
	return int(float(value)) if value else 0


def build_features(rows: List[Dict[str, str]]) -> Tuple[List[List[float]], List[int]]:
	X: List[List[float]] = []
	y: List[int] = []
	for r in rows:
		features = [
			to_float(r, "novelty_score"),
			to_float(r, "feasibility_score"),
			float(to_int(r, "projected_users")),
			float(to_int(r, "est_dev_weeks")),
			to_float(r, "prior_similar_success_rate"),
		]
		X.append(features)
		if LABEL_COLUMN in r:
			y.append(1 if to_int(r, LABEL_COLUMN) > 0 else 0)
	return X, y


def train_val_split(X: List[List[float]], y: List[int], val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[List[float]], List[int], List[List[float]], List[int]]:
	set_global_seed(seed)
	indices = list(range(len(X)))
	# Simple shuffle using random from utils seed
	import random
	random.shuffle(indices)
	cut = int(len(indices) * (1.0 - val_ratio))
	train_idx = indices[:cut]
	val_idx = indices[cut:]
	X_train = [X[i] for i in train_idx]
	y_train = [y[i] for i in train_idx]
	X_val = [X[i] for i in val_idx]
	y_val = [y[i] for i in val_idx]
	return X_train, y_train, X_val, y_val
