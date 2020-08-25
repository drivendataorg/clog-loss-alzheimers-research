from sklearn.metrics import accuracy_score, matthews_corrcoef


class BaseMetric:
    def __init__(self, thresh=None):
        self.thresh = thresh
        self.clean()

    def clean(self):
        self.preds = []
        self.tgts = []

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        if self.thresh is not None:
            preds = preds > self.thresh
        else:
            preds = preds.argmax(-1)

        self.preds.extend(preds)
        self.tgts.extend(targets)


class MatthewsCorrcoef(BaseMetric):
    def evaluate(self):
        return matthews_corrcoef(self.tgts, self.preds)


class Accuracy(BaseMetric):
    def evaluate(self):
        return accuracy_score(self.tgts, self.preds)