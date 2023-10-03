import re


class TokenEmbedder(object):
    STRING_LITERAL_REGEX = re.compile('^[fub]?["\'](.*)["\']$')
    STRING_LITERAL = '$StrLiteral$'
    INT_LITERAL = '$IntLiteral$'
    FLOAT_LITERAL = '$FloatLiteral$'

    @staticmethod
    def filter_literals(token: str) -> str:
        try:
            v = int(token)
            return TokenEmbedder.INT_LITERAL
        except ValueError:
            pass
        try:
            v = float(token)
            return TokenEmbedder.FLOAT_LITERAL
        except ValueError:
            pass
        string_lit = TokenEmbedder.STRING_LITERAL_REGEX.match(token)
        if string_lit:
            return TokenEmbedder.STRING_LITERAL
        return token


IGNORED_TYPES = {'typing.Any', 'Any', '', 'typing.NoReturn', 'NoReturn', 'nothing', 'None', None,
                 # Generic Type Params
                 'T', '_T', '_T0', '_T1', '_T2', '_T3', '_T4', '_T5', '_T6', '_T7'}


def ignore_type_annotation(name: str) -> bool:
    """A filter of types that should be ignored when learning or predicting"""
    if name in IGNORED_TYPES:
        return True
    if '%UNKNOWN%' in name:
        return True
    if name.startswith('_'):
        return True
    return False
