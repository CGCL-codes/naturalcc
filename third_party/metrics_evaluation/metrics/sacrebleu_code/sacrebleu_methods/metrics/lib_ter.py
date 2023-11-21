"""This module implements various utility functions for the TER metric."""

# Copyright 2020 Memsource
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import List, Tuple, Dict


_COST_INS = 1
_COST_DEL = 1
_COST_SUB = 1

# Tercom-inspired limits
_MAX_SHIFT_SIZE = 10
_MAX_SHIFT_DIST = 50
_BEAM_WIDTH = 25

# Our own limits
_MAX_CACHE_SIZE = 10000
_MAX_SHIFT_CANDIDATES = 1000
_INT_INFINITY = int(1e16)

_OP_INS = "i"
_OP_DEL = "d"
_OP_NOP = " "
_OP_SUB = "s"
_OP_UNDEF = "x"

_FLIP_OPS = str.maketrans(_OP_INS + _OP_DEL, _OP_DEL + _OP_INS)


def translation_edit_rate(
    words_hyp: List[str], words_ref: List[str]
) -> Tuple[int, int]:
    """Calculate the translation edit rate.

    :param words_hyp: Tokenized translation hypothesis.
    :param words_ref: Tokenized reference translation.
    :return: tuple (number of edits, length)
    """
    n_words_ref = len(words_ref)
    n_words_hyp = len(words_hyp)
    if n_words_ref == 0:
        # FIXME: This trace here is not used?
        trace = _OP_DEL * n_words_hyp
        # special treatment of empty refs
        return n_words_hyp, 0

    cached_ed = BeamEditDistance(words_ref)
    shifts = 0

    input_words = words_hyp
    checked_candidates = 0
    while True:
        # do shifts until they stop reducing the edit distance
        delta, new_input_words, checked_candidates = _shift(
            input_words, words_ref, cached_ed, checked_candidates
        )

        if checked_candidates >= _MAX_SHIFT_CANDIDATES:
            break

        if delta <= 0:
            break
        shifts += 1
        input_words = new_input_words

    edit_distance, trace = cached_ed(input_words)
    total_edits = shifts + edit_distance

    return total_edits, n_words_ref


def _shift(
    words_h: List[str], words_r: List[str], cached_ed, checked_candidates: int
) -> Tuple[int, List[str], int]:
    """Attempt to shift words in hypothesis to match reference.

    Returns the shift that reduces the edit distance the most.

    Note that the filtering of possible shifts and shift selection are heavily
    based on somewhat arbitrary heuristics. The code here follows as closely
    as possible the logic in Tercom, not always justifying the particular design
    choices.

    :param words_h: Hypothesis.
    :param words_r: Reference.
    :param cached_ed: Cached edit distance.
    :param checked_candidates: Number of shift candidates that were already
                               evaluated.
    :return: (score, shifted_words, checked_candidates). Best shift and updated
             number of evaluated shift candidates.
    """
    pre_score, inv_trace = cached_ed(words_h)

    # to get alignment, we pretend we are rewriting reference into hypothesis,
    # so we need to flip the trace of edit operations
    trace = _flip_trace(inv_trace)
    align, ref_err, hyp_err = trace_to_alignment(trace)

    best = None

    for start_h, start_r, length in _find_shifted_pairs(words_h, words_r):
        # don't do the shift unless both the hypothesis was wrong and the
        # reference doesn't match hypothesis at the target position
        if sum(hyp_err[start_h : start_h + length]) == 0:
            continue

        if sum(ref_err[start_r : start_r + length]) == 0:
            continue

        # don't try to shift within the subsequence
        if start_h <= align[start_r] < start_h + length:
            continue

        prev_idx = -1
        for offset in range(-1, length):
            if start_r + offset == -1:
                idx = 0  # insert before the beginning
            elif start_r + offset in align:
                # Unlike Tercom which inserts *after* the index, we insert
                # *before* the index.
                idx = align[start_r + offset] + 1
            else:
                break  # offset is out of bounds => aims past reference

            if idx == prev_idx:
                continue  # skip idx if already tried

            prev_idx = idx

            shifted_words = _perform_shift(words_h, start_h, length, idx)
            assert len(shifted_words) == len(words_h)

            # Elements of the tuple are designed to replicate Tercom ranking
            # of shifts:
            candidate = (
                pre_score - cached_ed(shifted_words)[0],  # highest score first
                length,  # then, longest match first
                -start_h,  # then, earliest match first
                -idx,  # then, earliest target position first
                shifted_words,
            )

            checked_candidates += 1

            if not best or candidate > best:
                best = candidate

        if checked_candidates >= _MAX_SHIFT_CANDIDATES:
            break

    if not best:
        return 0, words_h, checked_candidates
    else:
        best_score, _, _, _, shifted_words = best
        return best_score, shifted_words, checked_candidates


def _perform_shift(words: List[str], start: int, length: int, target: int) -> List[str]:
    """Perform a shift in `words` from `start` to `target`.

    :param words: Words to shift.
    :param start: Where from.
    :param length: How many words.
    :param target: Where to.
    :return: Shifted words.
    """
    if target < start:
        # shift before previous position
        return (
            words[:target]
            + words[start : start + length]
            + words[target:start]
            + words[start + length :]
        )
    elif target > start + length:
        # shift after previous position
        return (
            words[:start]
            + words[start + length : target]
            + words[start : start + length]
            + words[target:]
        )
    else:
        # shift within the shifted string
        return (
            words[:start]
            + words[start + length : length + target]
            + words[start : start + length]
            + words[length + target :]
        )


def _find_shifted_pairs(words_h: List[str], words_r: List[str]):
    """Find matching word sub-sequences in two lists of words.

    Ignores sub-sequences starting at the same position.

    :param words_h: First word list.
    :param words_r: Second word list.
    :return: Yields tuples of (h_start, r_start, length) such that:
         words_h[h_start:h_start+length] = words_r[r_start:r_start+length]
    """
    n_words_h = len(words_h)
    n_words_r = len(words_r)
    for start_h in range(n_words_h):
        for start_r in range(n_words_r):
            # this is slightly different from what tercom does but this should
            # really only kick in in degenerate cases
            if abs(start_r - start_h) > _MAX_SHIFT_DIST:
                continue

            length = 0
            while (
                words_h[start_h + length] == words_r[start_r + length]
                and length < _MAX_SHIFT_SIZE
            ):
                length += 1

                yield start_h, start_r, length

                # If one sequence is consumed, stop processing
                if n_words_h == start_h + length or n_words_r == start_r + length:
                    break


def _flip_trace(trace):
    """Flip the trace of edit operations.

    Instead of rewriting a->b, get a recipe for rewriting b->a.

    Simply flips insertions and deletions.
    """
    return trace.translate(_FLIP_OPS)


def trace_to_alignment(trace: str) -> Tuple[Dict, List, List]:
    """Transform trace of edit operations into an alignment of the sequences.

    :param trace: Trace of edit operations (' '=no change or 's'/'i'/'d').
    :return: Alignment, error positions in reference, error positions in hypothesis.
    """
    pos_hyp = -1
    pos_ref = -1
    hyp_err = []
    ref_err = []
    align = {}

    # we are rewriting a into b
    for op in trace:
        if op == _OP_NOP:
            pos_hyp += 1
            pos_ref += 1
            align[pos_ref] = pos_hyp
            hyp_err.append(0)
            ref_err.append(0)
        elif op == _OP_SUB:
            pos_hyp += 1
            pos_ref += 1
            align[pos_ref] = pos_hyp
            hyp_err.append(1)
            ref_err.append(1)
        elif op == _OP_INS:
            pos_hyp += 1
            hyp_err.append(1)
        elif op == _OP_DEL:
            pos_ref += 1
            align[pos_ref] = pos_hyp
            ref_err.append(1)
        else:
            raise Exception(f"unknown operation {op!r}")

    return align, ref_err, hyp_err


class BeamEditDistance:
    """Edit distance with several features required for TER calculation.

        * internal cache
        * "beam" search
        * tracking of edit operations

    The internal self._cache works like this:

    Keys are words of the hypothesis. Values are tuples (next_node, row) where:

        * next_node is the cache for the next word in the sequence
        * row is the stored row of the edit distance matrix

    Effectively, caching allows to skip several rows in the edit distance
    matrix calculation and instead, to initialize the computation with the last
    matching matrix row.

    Beam search, as implemented here, only explores a fixed-size sub-row of
    candidates around the matrix diagonal (more precisely, it's a
    "pseudo"-diagonal since we take the ratio of sequence lengths into account).

    Tracking allows to reconstruct the optimal sequence of edit operations.

    :param words_ref: A list of reference tokens.
    """

    def __init__(self, words_ref: List[str]):
        """`BeamEditDistance` initializer."""
        self._words_ref = words_ref
        self._n_words_ref = len(self._words_ref)

        # first row corresponds to insertion operations of the reference,
        # so we do 1 edit operation per reference word
        self._initial_row = [
            (i * _COST_INS, _OP_INS) for i in range(self._n_words_ref + 1)
        ]

        self._cache = {}  # type: Dict[str, Tuple]
        self._cache_size = 0

        # Precomputed empty matrix row. Contains infinities so that beam search
        # avoids using the uninitialized cells.
        self._empty_row = [(_INT_INFINITY, _OP_UNDEF)] * (self._n_words_ref + 1)

    def __call__(self, words_hyp: List[str]) -> Tuple[int, str]:
        """Calculate edit distance between self._words_ref and the hypothesis.

        Uses cache to skip some of the computation.

        :param words_hyp: Words in translation hypothesis.
        :return: Edit distance score.
        """

        # skip initial words in the hypothesis for which we already know the
        # edit distance
        start_position, dist = self._find_cache(words_hyp)

        # calculate the rest of the edit distance matrix
        edit_distance, newly_created_matrix, trace = self._edit_distance(
            words_hyp, start_position, dist
        )

        # update our cache with the newly calculated rows
        self._add_cache(words_hyp, newly_created_matrix)

        return edit_distance, trace

    def _edit_distance(
        self, words_h: List[str], start_h: int, cache: List[List[Tuple[int, str]]]
    ) -> Tuple[int, List, str]:
        """Actual edit distance calculation.

        Can be initialized with the last cached row and a start position in
        the hypothesis that it corresponds to.

        :param words_h: Words in translation hypothesis.
        :param start_h: Position from which to start the calculation.
                        (This is zero if no cache match was found.)
        :param cache: Precomputed rows corresponding to edit distance matrix
                      before `start_h`.
        :return: Edit distance value, newly computed rows to update the
                 cache, trace.
        """

        n_words_h = len(words_h)

        # initialize the rest of the matrix with infinite edit distances
        rest_empty = [list(self._empty_row) for _ in range(n_words_h - start_h)]

        dist = cache + rest_empty

        assert len(dist) == n_words_h + 1

        length_ratio = self._n_words_ref / n_words_h if words_h else 1

        # in some crazy sentences, the difference in length is so large that
        # we may end up with zero overlap with previous row
        if _BEAM_WIDTH < length_ratio / 2:
            beam_width = math.ceil(length_ratio / 2 + _BEAM_WIDTH)
        else:
            beam_width = _BEAM_WIDTH

        # calculate the Levenshtein distance
        for i in range(start_h + 1, n_words_h + 1):
            pseudo_diag = math.floor(i * length_ratio)
            min_j = max(0, pseudo_diag - beam_width)
            max_j = min(self._n_words_ref + 1, pseudo_diag + beam_width)

            if i == n_words_h:
                max_j = self._n_words_ref + 1

            for j in range(min_j, max_j):
                if j == 0:
                    dist[i][j] = (dist[i - 1][j][0] + _COST_DEL, _OP_DEL)
                else:
                    if words_h[i - 1] == self._words_ref[j - 1]:
                        cost_sub = 0
                        op_sub = _OP_NOP
                    else:
                        cost_sub = _COST_SUB
                        op_sub = _OP_SUB

                    # Tercom prefers no-op/sub, then insertion, then deletion.
                    # But since we flip the trace and compute the alignment from
                    # the inverse, we need to swap order of insertion and
                    # deletion in the preference.
                    ops = (
                        (dist[i - 1][j - 1][0] + cost_sub, op_sub),
                        (dist[i - 1][j][0] + _COST_DEL, _OP_DEL),
                        (dist[i][j - 1][0] + _COST_INS, _OP_INS),
                    )

                    for op_cost, op_name in ops:
                        if dist[i][j][0] > op_cost:
                            dist[i][j] = op_cost, op_name

        # get the trace
        trace = ""
        i = n_words_h
        j = self._n_words_ref

        while i > 0 or j > 0:
            op = dist[i][j][1]
            trace = op + trace
            if op in (_OP_SUB, _OP_NOP):
                i -= 1
                j -= 1
            elif op == _OP_INS:
                j -= 1
            elif op == _OP_DEL:
                i -= 1
            else:
                raise Exception(f"unknown operation {op!r}")

        return dist[-1][-1][0], dist[len(cache) :], trace

    def _add_cache(self, words_hyp: List[str], mat: List[List[Tuple]]):
        """Add newly computed rows to cache.

        Since edit distance is only calculated on the hypothesis suffix that
        was not in cache, the number of rows in `mat` may be shorter than
        hypothesis length. In that case, we skip over these initial words.

        :param words_hyp: Hypothesis words.
        :param mat: Edit distance matrix rows for each position.
        """
        if self._cache_size >= _MAX_CACHE_SIZE:
            return

        node = self._cache

        n_mat = len(mat)

        # how many initial words to skip
        skip_num = len(words_hyp) - n_mat

        # jump through the cache to the current position
        for i in range(skip_num):
            node = node[words_hyp[i]][0]

        assert len(words_hyp[skip_num:]) == n_mat

        # update cache with newly computed rows
        for word, row in zip(words_hyp[skip_num:], mat):
            if word not in node:
                node[word] = ({}, tuple(row))
                self._cache_size += 1
            value = node[word]
            node = value[0]

    def _find_cache(self, words_hyp: List[str]) -> Tuple[int, List[List]]:
        """Find the already computed rows of the edit distance matrix in cache.

        Returns a partially computed edit distance matrix.

        :param words_hyp: Translation hypothesis.
        :return: Tuple (start position, dist).
        """
        node = self._cache
        start_position = 0
        dist = [self._initial_row]
        for word in words_hyp:
            if word in node:
                start_position += 1
                node, row = node[word]
                dist.append(row)
            else:
                break

        return start_position, dist
