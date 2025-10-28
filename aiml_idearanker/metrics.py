from typing import List, Tuple


def threshold_predictions(probs: List[float], threshold: float = 0.5) -> List[int]:
	return [1 if p >= threshold else 0 for p in probs]


def confusion(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
	tp = fp = tn = fn = 0
	for t, p in zip(y_true, y_pred):
		if t == 1 and p == 1:
			tp += 1
		elif t == 0 and p == 1:
			fp += 1
		elif t == 0 and p == 0:
			tn += 1
		else:
			fn += 1
	return tp, fp, tn, fn


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
	correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
	return correct / max(1, len(y_true))


def precision_recall_f1(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
	tp, fp, tn, fn = confusion(y_true, y_pred)
	precision = tp / max(1, (tp + fp))
	recall = tp / max(1, (tp + fn))
	if precision + recall == 0:
		f1 = 0.0
	else:
		f1 = 2 * precision * recall / (precision + recall)
	return precision, recall, f1
