import re
import ujson
from collections import Counter
from dpu_utils.codeutils import split_identifier_into_parts
import itertools

IDENTIFIER_TOKEN_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')


def sub_tokenizer(tokens: str):
    """code from https://github.com/github/CodeSearchNet/blob/e792e1caea20fbd4fba439565fe20c10d4798435/src/encoders/seq_encoder.py#L84-L92"""
    tokens = ujson.loads(tokens)
    tokens = [split_identifier_into_parts(tok) if IDENTIFIER_TOKEN_REGEX.match(tok) else [tok] for tok in tokens]
    tokens = list(itertools.chain(*tokens))
    tokens = [tok for tok in tokens if len(tok) > 0]
    return tokens


def string_sub_tokenizer(tokens: list):
    """code from https://github.com/github/CodeSearchNet/blob/e792e1caea20fbd4fba439565fe20c10d4798435/src/encoders/seq_encoder.py#L84-L92"""
    tokens = [split_identifier_into_parts(tok) if IDENTIFIER_TOKEN_REGEX.match(tok) else tok for tok in tokens]
    tokens = list(itertools.chain(*tokens))
    return tokens


def func_name_tokenizer(tokens):
    tokens = ujson.loads(tokens)
    return split_identifier_into_parts(tokens)


def lower_tokenizer(tokens: str):
    """code from https://github.com/github/CodeSearchNet/blob/e792e1caea20fbd4fba439565fe20c10d4798435/src/encoders/seq_encoder.py#L84-L92"""
    tokens = ujson.loads(tokens)
    return list(map(str.lower, tokens))


def byte_pair_counts(words, ngram_min=2, ngram_max=8):
    """ Counts space separated token character pairs:
        [('T h i s </w>', 4}] -> {'Th': 4, 'hi': 4, 'is': 4}
    """

    def count_tokens(words):
        """ Count tokens into a BPE vocab """
        token_counts = Counter(words)
        return {' '.join(token): count for token, count in token_counts.items()}

    for token, count in count_tokens(words).items():
        bp_counts = Counter()  # type: Counter
        sub_tokens = token.split(' ')
        joined_tokens = ''.join(sub_tokens)
        token_offsets = [0]
        length = 0
        for ngram in sub_tokens:
            bp_counts[ngram] += count
            length += len(ngram)
            token_offsets += [length]
        for ngram_size in range(ngram_min, min(ngram_max, len(sub_tokens)) + 1):
            for i in range(len(sub_tokens) - ngram_size + 1):
                bp_counts[joined_tokens[token_offsets[i]:token_offsets[i + ngram_size]]] += count

        yield bp_counts


def learn_bpe_vocab(words, bpe_vocab_size):
    def trim_vocab(n, vocab):
        """  Deletes all pairs below 10 * vocab size to prevent memory problems """
        pair_counts = sorted(vocab.items(), key=lambda p: -p[1])
        pairs_to_trim = [pair for pair, count in pair_counts[n:]]
        for pair in pairs_to_trim:
            del vocab[pair]

    vocab = Counter()
    for idx, byte_pair_count in enumerate(byte_pair_counts(words)):
        vocab.update(byte_pair_count)
        if (idx + 1) % 10000 == 0:
            trim_vocab(10 * bpe_vocab_size, vocab)

    sorted_bpe_counts = sorted(vocab.items(), key=lambda p: -p[1])[:bpe_vocab_size]
    return sorted_bpe_counts
