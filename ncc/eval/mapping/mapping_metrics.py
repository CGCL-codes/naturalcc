# -*- coding: utf-8 -*-

def platform2str(platform: str) -> str:
    """ get full platform name """
    if platform == "amd":
        return "AMD Tahiti 7970"
    elif platform == "nvidia":
        return "NVIDIA GTX 970"
    else:
        raise LookupError


def escape_suite_name(g: str) -> str:
    """ format benchmark suite name for display """
    c = g.split('-')
    if c[0] == "amd" or c[0] == "nvidia":
        return c[0].upper() + " SDK"
    if c[0] == "npb" or c[0] == "shoc":
        return c[0].upper()
    elif c[0] == "parboil" or c[0] == "polybench" or c[0] == "rodinia":
        return c[0].capitalize()
    else:
        raise LookupError


def escape_benchmark_name(g: str) -> str:
    """escape benchmark name for display"""
    c = g.split('-')
    return escape_suite_name(c[0]).split()[0] + "." + c[-2]
