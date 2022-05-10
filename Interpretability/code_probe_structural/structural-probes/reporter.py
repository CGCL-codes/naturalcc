"""Contains classes for computing and reporting evaluation metrics."""

from collections import defaultdict
import os

from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import numpy as np 
import json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000

class Reporter:
  """Base class for reporting.

  Attributes:
    test_reporting_constraint: Any reporting method
      (identified by a string) not in this list will not
      be reported on for the test set.
  """

  def __init__(self, args):
    raise NotImplementedError("Inherit from this class and override __init__")

  def __call__(self, prediction_batches, dataloader, split_name):
    """
    Performs all reporting methods as specifed in the yaml experiment config dict.
    
    Any reporting method not in test_reporting_constraint will not
      be reported on for the test set.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataloader: A DataLoader for a data split
      split_name the string naming the data split: {train,dev,test}
    """
    for method in self.reporting_methods:
      if method in self.reporting_method_dict:
        if split_name == 'test' and method not in self.test_reporting_constraint:
          tqdm.write("Reporting method {} not in test set reporting "
              "methods (reporter.py); skipping".format(method))
          continue
        tqdm.write("Reporting {} on split {}".format(method, split_name))
        self.reporting_method_dict[method](prediction_batches
            , dataloader, split_name)
      else:
        tqdm.write('[WARNING] Reporting method not known: {}; skipping'.format(method))

  def write_json(self, prediction_batches, dataset, split_name):
    """Writes observations and predictions to disk.
    
    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    json.dump([prediction_batch.tolist() for prediction_batch in prediction_batches]
        , open(os.path.join(self.reporting_root, split_name+'.predictions'), 'w'))
    json.dump([[x[0][:-1] for x in observation_batch] for _,_,_, observation_batch in dataset],
        open(os.path.join(self.reporting_root, split_name+'.observations'), 'w'))

class WordPairReporter(Reporter):
  """Reporting class for wordpair (distance) tasks"""

  def __init__(self, args):
    self.args = args
    self.reporting_methods = args['reporting']['reporting_methods']
    self.reporting_method_dict = {
        'spearmanr': self.report_spearmanr,
        'image_examples':self.report_image_examples,
        # 'uuas':self.report_uuas_and_tikz,
        'write_predictions':self.write_json
        }
    self.reporting_root = args['reporting']['root']
    self.test_reporting_constraint = {'spearmanr', 'uuas', 'root_acc'}

  def report_spearmanr(self, prediction_batches, dataset, split_name):
    """Writes the Spearman correlations between predicted and true distances.

    For each word in each sentence, computes the Spearman correlation between
    all true distances between that word and all other words, and all
    predicted distances between that word and all other words.

    Computes the average such metric between all sentences of the same length.
    Writes these averages to disk.
    Then computes the average Spearman across sentence lengths 5 to 50;
    writes this average to disk.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    lengths_to_spearmanrs = defaultdict(list)
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        words = observation.code_tokens
        length = int(length)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()
        spearmanrs = [spearmanr(pred, gold) for pred, gold in zip(prediction, label)]
        lengths_to_spearmanrs[length].extend([x.correlation for x in spearmanrs])
    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length]) 
        for length in lengths_to_spearmanrs}

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr'), 'w') as fout:
      for length in sorted(mean_spearman_for_each_length):
        fout.write(str(length) + '\t' + str(mean_spearman_for_each_length[length]) + '\n')

    with open(os.path.join(self.reporting_root, split_name + '.spearmanr-5_50-mean'), 'w') as fout:
      mean = np.mean([mean_spearman_for_each_length[x] for x in range(5,51) if x in mean_spearman_for_each_length])
      fout.write(str(mean) + '\n')

  def report_image_examples(self, prediction_batches, dataset, split_name):
    """Writes predicted and gold distance matrices to disk for the first 20
    elements of the developement set as images!

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataset: A sequence of batches of Observations
      split_name the string naming the data split: {train,dev,test}
    """
    images_printed = 0
    for prediction_batch, (data_batch, label_batch, length_batch, observation_batch) in zip(
        prediction_batches, dataset):
      for prediction, label, length, (observation, _) in zip(
          prediction_batch, label_batch,
          length_batch, observation_batch):
        length = int(length)
        prediction = prediction[:length,:length]
        label = label[:length,:length].cpu()
        words = observation.code_tokens
        fontsize = 8*( 1 + np.sqrt(len(words))/200)
        plt.clf()
        ax = sns.heatmap(label,cmap='Blues_r')
        ax.set_title('Gold Parse Distance')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va='top')
        plt.tight_layout()
        plt.savefig(os.path.join(self.reporting_root, split_name + '-gold'+str(images_printed)+'.pdf'), format='pdf',dpi=300)

        plt.clf()
        ax = sns.heatmap(prediction,cmap='Blues_r')
        ax.set_title('Predicted Parse Distance (squared)')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.reporting_root, split_name + '-pred'+str(images_printed)+'.pdf'), format='pdf',dpi=300)
        print('Printing', str(images_printed))
        images_printed += 1
        if images_printed == 20:
          return

class UnionFind:
  '''
  Naive UnionFind implementation for (slow) Prim's MST algorithm

  Used to compute minimum spanning trees for distance matrices
  '''
  def __init__(self, n):
    self.parents = list(range(n))
  def union(self, i,j):
    if self.find(i) != self.find(j):
      i_parent = self.find(i)
      self.parents[i_parent] = j
  def find(self, i):
    i_parent = i
    while True:
      if i_parent != self.parents[i_parent]:
        i_parent = self.parents[i_parent]
      else:
        break
    return i_parent


def prims_matrix_to_edges(matrix, words):
  '''
  Constructs a minimum spanning tree from the pairwise weights in matrix;
  returns the edges.

  Never lets punctuation-tagged words be part of the tree.
  '''
  pairs_to_distances = {}
  uf = UnionFind(len(matrix))
  for i_index, line in enumerate(matrix):
    for j_index, dist in enumerate(line):
      # if poses[i_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
      #   continue
      # if poses[j_index] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
      #   continue
      pairs_to_distances[(i_index, j_index)] = dist
  edges = []
  for (i_index, j_index), distance in sorted(pairs_to_distances.items(), key = lambda x: x[1]):
    if uf.find(i_index) != uf.find(j_index):
      uf.union(i_index, j_index)
      edges.append((i_index, j_index))
  return edges

def get_nopunct_argmin(prediction, words, poses):
  '''
  Gets the argmin of predictions, but filters out all punctuation-POS-tagged words
  '''
  puncts = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
  original_argmin = np.argmin(prediction)
  for i in range(len(words)):
    argmin = np.argmin(prediction)
    if poses[argmin] not in puncts:
      return argmin
    else:
      prediction[argmin] = 9000
  return original_argmin
