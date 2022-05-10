"""Classes for constructing word representations."""
import torch
import torch.nn as nn

class Model(nn.Module):
  '''An abstract class for neural models that
  assign a single vector to each word in a text.
  '''

  def __init__(self, args):
    super(Model, self).__init__()

  def forward(self, *args):
    """Assigns a vector to each word in a batch."""
    raise NotImplementedError("Model is an abstract class; "
        "use one of the implementing classes.")


class DiskModel(Model):
  '''A class for providing pre-computed word representations.
  Assumes the batch is constructed of loaded-from-disk
  embeddings.
  '''

  def __init__(self, args):
    super(DiskModel, self).__init__(args)

  def forward(self, batch):
    """Returns the batch itself.

    Args:
      batch: a batch of pre-computed embeddings loaded from disk.

    Returns:
      The batch, unchanged
    """
    return batch


class PyTorchModel(Model):

  def __init__(self, args, **kwargs):
    super(PyTorchModel, self).__init__(args)

