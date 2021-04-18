from collections import Counter
import typing
from typing import Iterable, Dict, Sized, List, FrozenSet, Union, Optional

import numpy as np

# __all__ = ['Vocabulary']


class Vocabulary(object):
    """
    A simple vocabulary that maps strings to unique ids (and back).

    To create a vocabulary use `Vocabulary.create_vocabulary()` and pass
    a counter or an iterator of the elements that the vocabulary will contain.

    Vocabulary object usage: Assuming an initialized vocabulary `v`:

       * To get the id of an element use `v.get_id_or_unk("element")`.
       * To get the ids of a sequence, use `v.get_id_or_unk_multiple(..)`.
       * To get the size of the vocabulary use `len(v)`
       * To get the string representation for a given id use `v.get_name_for_id(the_id)`.
    """

    def __init__(self, add_unk: bool=True, add_pad: bool=False) -> None:
        self.token_to_id = {}  # type: Dict[str, int]
        self.id_to_token = []  # type: List[str]
        if add_pad:
            self.add_or_get_id(self.get_pad())
        if add_unk:
            self.add_or_get_id(self.get_unk())

    def is_equal_to(self, other) -> bool:
        """
        This would be __eq__, except that Python 3 insists that __hash__ is defined if __eq__ is
        defined, and we can't define __hash__ because the object is mutable.
        """
        if not isinstance(other, Vocabulary):
            return False
        return self.id_to_token == other.id_to_token

    def add_or_get_id(self, token: str) -> int:
        """
        Get the token's id. If the token does not exist in the
        dictionary, add it.
        """
        this_id = self.token_to_id.get(token)
        if this_id is not None:
            return this_id

        this_id = len(self.id_to_token)
        self.token_to_id[token] = this_id
        self.id_to_token.append(token)
        return this_id

    def is_unk(self, token: str) -> bool:
        return token not in self.token_to_id

    def get_id_or_unk(self, token: str) -> int:
        this_id = self.token_to_id.get(token)
        if this_id is not None:
            return this_id
        else:
            return self.token_to_id[self.get_unk()]

    def get_id_or_unk_multiple(self, tokens: List[str], pad_to_size: Optional[int] = None,
                               padding_element: int = 0) -> List[int]:
        if pad_to_size is not None:
            tokens = tokens[:pad_to_size]

        ids = [self.get_id_or_unk(t) for t in tokens]

        if pad_to_size is not None and len(ids) != pad_to_size:
            ids += [padding_element] * (pad_to_size - len(ids))

        return ids

    def get_name_for_id(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __str__(self):
        return str(self.token_to_id)

    def get_all_names(self) -> FrozenSet[str]:
        return frozenset(self.token_to_id.keys())

    def __batch_add_from_counter(self, token_counter: typing.Counter[str], count_threshold: int, max_size: int) -> None:
        """Update dictionary with elements of the token_counter"""
        for token, count in token_counter.most_common(max_size):
            if count >= count_threshold:
                self.add_or_get_id(token)
            else:
                break

    @staticmethod
    def get_unk() -> str:
        return '%UNK%'

    @staticmethod
    def get_pad() -> str:
        return '%PAD%'

    @staticmethod
    def create_vocabulary(tokens: Union[Iterable[str], typing.Counter[str]], max_size: int,
                          count_threshold: int = 5, add_unk: bool = True, add_pad: bool = False) -> 'Vocabulary':
        if isinstance(tokens, Counter):
            token_counter = tokens
        else:
            token_counter = Counter(tokens)
        vocab = Vocabulary(add_unk=add_unk, add_pad=add_pad)
        num_base_tokens = (1 if add_unk else 0) + (1 if add_pad else 0)
        vocab.__batch_add_from_counter(token_counter, count_threshold, max_size - num_base_tokens)
        return vocab

    def update(self, token_counter: typing.Counter[str], max_size: int, count_threshold: int=5):
        assert len(self) < max_size, 'Dictionary is already larger than max_size.'
        self.__batch_add_from_counter(token_counter, count_threshold=count_threshold, max_size=max_size)

    def get_empirical_distribution(self, elements: Iterable[str], dirichlet_alpha: float = 10.) -> np.ndarray:
        """Retrieve empirical distribution of elements."""
        targets = np.fromiter((self.get_id_or_unk(t) for t in elements), dtype=np.int)
        empirical_distribution = np.bincount(targets, minlength=len(self)).astype(float)
        empirical_distribution += dirichlet_alpha / len(empirical_distribution)
        return empirical_distribution / (np.sum(empirical_distribution) + dirichlet_alpha)
