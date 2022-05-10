"""Contains classes describing linguistic tasks of interest on annotated data."""

import numpy as np
import torch

class Task:
  """Abstract class representing a linguistic task mapping texts to labels."""
  @staticmethod
  def labels(observation):
    """Maps an observation to a matrix of labels.
    
    Should be overriden in implementing classes.
    """
    raise NotImplementedError

class ParseDistanceTask(Task):
  """Maps observations to dependency parse distances between words."""
  @staticmethod
  def labels(observation):
    """Computes the distances between all pairs of words; returns them as a torch tensor.
     计算所有单词对之间的距离；
    Args:
      observation: a single Observation class for a sentence:
    Returns:
      A torch tensor of shape (sentence_length, sentence_length) of distances
      in the parse tree as specified by the observation annotation.
 
    """
    sigle_dict_ast=observation[0]
    code_head_indices = []  
    dict_code = []
    for item in sigle_dict_ast:
      if sigle_dict_ast[item]['parent'] == 'null':
        code_head_indices.append(0)
      else:
        code_head_indices.append(sigle_dict_ast[item]['parent'] + 1)
      if 'value' in sigle_dict_ast[item]:
        dict_code.append(int(item))
    sentence_length = len(code_head_indices)  # All observation fields must be of same length
    distances=torch.zeros((sentence_length,sentence_length))
    for i in range(sentence_length):
      for j in range(i, sentence_length):
        i_j_distance = ParseDistanceTask.distance_between_pairs(i, j, code_head_indices)
        distances[i][j] = i_j_distance
        distances[j][i] = i_j_distance

    code_sentence_length = len(dict_code)
    code_distance=torch.zeros((code_sentence_length,code_sentence_length))
    for i in range(code_sentence_length):
      for j in range(i, code_sentence_length):
        code_i = dict_code[i]
        code_j = dict_code[j]
        code_distance[i][j] = distances[code_i][code_j]
        code_distance[j][i] = distances[code_j][code_i]
    return code_distance

  @staticmethod
  def distance_between_pairs(i, j, head_indices=None):
    '''Computes path distance between a pair of words

    TODO: It would be (much) more efficient to compute all pairs' distances at once;
          this pair-by-pair method is an artefact of an older design, but
          was unit-tested for correctness...

    Args:
      observation: an Observation namedtuple, with a head_indices field.
          or None, if head_indies != None
      i: one of the two words to compute the distance between.
      j: one of the two words to compute the distance between.
      head_indices: the head indices (according to a dependency parse) of all
          words, or None, if observation != None.

    Returns:
      The integer distance d_path(i,j)
    '''
    if i == j:
      return 0
    i_path = [i + 1]
    j_path = [j + 1]
    i_head = i + 1
    j_head = j + 1
    while True:
      if not (i_head == 0 and (i_path == [i + 1] or i_path[-1] == 0)):
        i_head = head_indices[i_head - 1]
        i_path.append(i_head)
      if not (j_head == 0 and (j_path == [j + 1] or j_path[-1] == 0)):
        j_head = head_indices[j_head - 1]
        j_path.append(j_head)
      if i_head in j_path:
        j_path_length = j_path.index(i_head)
        i_path_length = len(i_path) - 1
        break
      elif j_head in i_path:
        i_path_length = i_path.index(j_head)
        j_path_length = len(j_path) - 1
        break
      elif i_head == j_head:
        i_path_length = len(i_path) - 1
        j_path_length = len(j_path) - 1
        break
    total_length = j_path_length + i_path_length
    return total_length


