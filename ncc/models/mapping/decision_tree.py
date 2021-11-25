# -*- coding: utf-8 -*-


from sklearn.tree import DecisionTreeClassifier

from ncc.models import register_model
from ncc.models.ncc_model import (
    NccEncoder,
    NccEncoderModel,
)


class DecisionTreeEncoder(NccEncoder):
    def __init__(
        self,
        criterion="entropy",
        max_depth=5,
        min_samples_leaf=5,
    ):
        """
        optimal_device: 0 for cpu; 1 for gpu
        """
        super(DecisionTreeEncoder, self).__init__(dictionary=None)
        self._model = DecisionTreeClassifier(
            random_state=204, splitter="best",
            criterion=criterion, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf)

    def fit(self, src_tokens, ground_truth, **kwargs):
        self._model.fit(src_tokens, ground_truth)

    def forward(self, src_tokens, **kwargs):
        return self._model.predict(src_tokens)


@register_model('decision_tree')
class DecisionTree(NccEncoderModel):
    """
    ref: Grewe, D., Wang, Z., & O’Boyle, M. (2013). [Portable Mapping of Data Parallel Programs to OpenCL for Heterogeneous Systems](http://www.ece.neu.edu/groups/nucar/NUCARTALKS/cgo2013-grewe.pdf). In CGO. IEEE.
    """

    def __init__(self, encoder):
        super(DecisionTree, self).__init__(encoder)

    @classmethod
    def build_model(cls):
        encoder = DecisionTreeEncoder()
        return cls(encoder)

    def fit(self, *args, **kwargs):
        self.encoder.fit(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
