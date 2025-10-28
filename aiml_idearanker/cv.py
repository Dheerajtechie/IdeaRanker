from typing import Dict, List, Tuple

from .metrics import accuracy, precision_recall_f1, threshold_predictions
from .utils import set_global_seed


def k_fold_indices(n: int, k: int, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
	set_global_seed(seed)
	import random
	indices = list(range(n))
	random.shuffle(indices)
	fold_size = max(1, n // k)
	folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k - 1)]
	folds.append(indices[(k - 1) * fold_size:])
	pairs: List[Tuple[List[int], List[int]]] = []
	for i in range(k):
		val_idx = folds[i]
		train_idx: List[int] = []
		for j in range(k):
			if j != i:
				train_idx.extend(folds[j])
		pairs.append((train_idx, val_idx))
	return pairs


def evaluate_folds(probs_per_fold: List[List[float]], y_true_per_fold: List[List[int]], threshold: float = 0.5) -> Dict[str, float]:
	accs: List[float] = []
	ps: List[float] = []
	rs: List[float] = []
	f1s: List[float] = []
	for probs, y_true in zip(probs_per_fold, y_true_per_fold):
		y_pred = threshold_predictions(probs, threshold)
		accs.append(accuracy(y_true, y_pred))
		p, r, f1 = precision_recall_f1(y_true, y_pred)
		ps.append(p)
		rs.append(r)
		f1s.append(f1)
	return {
		"accuracy": sum(accs) / max(1, len(accs)),
		"precision": sum(ps) / max(1, len(ps)),
		"recall": sum(rs) / max(1, len(rs)),
		"f1": sum(f1s) / max(1, len(f1s)),
	}
