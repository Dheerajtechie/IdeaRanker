from typing import Dict, List, Tuple

from .utils import sigmoid, standardize_column, apply_standardize


class StandardScaler:
	def __init__(self) -> None:
		self.means: List[float] = []
		self.stds: List[float] = []

	def fit(self, X: List[List[float]]) -> None:
		if not X:
			self.means, self.stds = [], []
			return
		n_features = len(X[0])
		self.means = []
		self.stds = []
		for j in range(n_features):
			col = [row[j] for row in X]
			_, m, s = self.transform_single_column(col)
			self.means.append(m)
			self.stds.append(s)

	def transform_single_column(self, col: List[float]) -> Tuple[List[float], float, float]:
		col_z, m, s = standardize_column(col)
		return col_z, m, s

	def transform(self, X: List[List[float]]) -> List[List[float]]:
		if not X:
			return []
		if not self.means:
			return X
		Xz: List[List[float]] = []
		for row in X:
			Xz.append([apply_standardize([row[j]], self.means[j], self.stds[j])[0] for j in range(len(row))])
		return Xz


def initialize_weights(n_features: int) -> List[float]:
	# Bias + weights
	return [0.0 for _ in range(n_features + 1)]


def predict_proba_row(weights: List[float], row: List[float]) -> float:
	z = weights[0]  # bias
	for j, x in enumerate(row, start=1):
		z += weights[j] * x
	return sigmoid(z)


def predict_proba(weights: List[float], X: List[List[float]]) -> List[float]:
	return [predict_proba_row(weights, row) for row in X]


def train_logistic_regression(
	X: List[List[float]],
	y: List[int],
	lr: float = 0.1,
	epochs: int = 200,
	l2: float = 0.0,
) -> List[float]:
	if not X:
		return []
	n_features = len(X[0])
	w = initialize_weights(n_features)
	for _ in range(epochs):
		# gradients: bias + weights
		grad = [0.0 for _ in range(n_features + 1)]
		for row, t in zip(X, y):
			p = predict_proba_row(w, row)
			err = p - t
			grad[0] += err
			for j, x in enumerate(row, start=1):
				grad[j] += err * x
		# L2 regularization (excluding bias)
		for j in range(1, n_features + 1):
			grad[j] += l2 * w[j]
		# update
		for j in range(n_features + 1):
			w[j] -= lr * (grad[j] / max(1, len(X)))
	return w


class IdeaRankerModel:
	def __init__(self) -> None:
		self.scaler = StandardScaler()
		self.weights: List[float] = []
		self.metadata: Dict[str, str] = {}

	def fit(self, X: List[List[float]], y: List[int], lr: float = 0.1, epochs: int = 200, l2: float = 0.0) -> None:
		self.scaler.fit(X)
		Xz = self.scaler.transform(X)
		self.weights = train_logistic_regression(Xz, y, lr=lr, epochs=epochs, l2=l2)

	def predict_proba(self, X: List[List[float]]) -> List[float]:
		Xz = self.scaler.transform(X)
		return predict_proba(self.weights, Xz)

	def to_dict(self) -> Dict[str, object]:
		return {
			"weights": self.weights,
			"scaler_means": self.scaler.means,
			"scaler_stds": self.scaler.stds,
			"metadata": self.metadata,
		}

	@classmethod
	def from_dict(cls, payload: Dict[str, object]) -> "IdeaRankerModel":
		m = cls()
		m.weights = list(payload.get("weights", []))  # type: ignore[arg-type]
		m.scaler.means = list(payload.get("scaler_means", []))  # type: ignore[arg-type]
		m.scaler.stds = list(payload.get("scaler_stds", []))  # type: ignore[arg-type]
		m.metadata = dict(payload.get("metadata", {}))  # type: ignore[arg-type]
		return m
