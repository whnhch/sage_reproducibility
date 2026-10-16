"""Trend LLM-proxy: routes each row to the correlation or ratio tree by its 'insight_type'."""
import numpy as np


class TrendProxy:
    def __init__(self, correlation, ratio):
        self.correlation = correlation
        self.ratio = ratio

    def predict(self, X):
        is_ratio = (X['insight_type'] == 'rat').to_numpy()
        if not is_ratio.any():
            return self.correlation.predict(X)
        if is_ratio.all():
            return self.ratio.predict(X)
        y = np.empty(len(X), dtype=object)
        y[~is_ratio] = self.correlation.predict(X[~is_ratio])
        y[is_ratio] = self.ratio.predict(X[is_ratio])
        return y
