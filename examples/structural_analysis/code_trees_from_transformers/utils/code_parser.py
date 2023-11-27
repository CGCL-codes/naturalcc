import numpy as np


def not_coo_parser(score, sent):
    assert len(score) == len(sent) - 1

    if len(score) == 0:
        parse_tree = f'《T {sent[0]} 》'
    elif len(score) == 1:
        parse_tree = f'《T 《T {sent[0]} 》 《T {sent[1]} 》 》'
    elif (np.mean(list(map(lambda x: abs(x[0] - x[1]), zip(score[1:], score[:-1])))) < 0.001):
        parse_tree = "《T"
        for i in sent:
            parse_tree += f' 《T {i} 》'
        parse_tree += " 》"
    else:
        idx_max = np.argmax(score)
        l_len = len(sent[:idx_max + 1])
        r_len = len(sent[idx_max + 2:])
        if l_len > 0 and r_len > 0:
            l_tree = not_coo_parser(score[:idx_max], sent[:idx_max + 1])
            r_tree = not_coo_parser(score[idx_max + 2:], sent[idx_max + 2:])
            r_tree = f'《T 《T {sent[idx_max +1]} 》 {r_tree} 》'
            parse_tree = f'《T {l_tree} {r_tree} 》'
        else:
            if l_len == 0:
                r_tree = not_coo_parser(score[idx_max + 2:], sent[idx_max + 2:])
                r_tree = f'《T 《T {sent[idx_max +1]} 》 {r_tree} 》'
                parse_tree = r_tree
            else:
                l_tree = not_coo_parser(score[:idx_max], sent[:idx_max + 1])
                parse_tree = f'《T {l_tree} 《T {sent[idx_max + 1]} 》 》'

    return parse_tree


def parser(score, sent):
    assert len(score) == len(sent) - 1

    if len(score) == 0:
        parse_tree = f'《T {sent[0]} 》'
    elif len(score) == 1:
        parse_tree = f'《T 《T {sent[0]} 》 《T {sent[1]} 》 》'
    elif(np.mean(list(map(lambda x: abs(x[0]-x[1]), zip(score[1:], score[:-1]))))<0.001):
        parse_tree="《T"
        for i in sent:
            parse_tree+=f' 《T {i} 》'
        parse_tree+=" 》"
    else:
        idx_max = np.argmax(score)
        l_len = len(sent[:idx_max + 1])
        r_len = len(sent[idx_max + 1:])
        if l_len > 0 and r_len > 0:
            l_tree = parser(score[:idx_max], sent[:idx_max + 1])
            r_tree = parser(score[idx_max + 1:], sent[idx_max + 1:])
            parse_tree = f'《T {l_tree} {r_tree} 》'
        else:
            if l_len == 0:
                r_tree = parser(score[idx_max + 1:], sent[idx_max + 1:])
                parse_tree = r_tree
            else:
                l_tree = parser(score[:idx_max], sent[:idx_max + 1])
                parse_tree = l_tree
    return parse_tree