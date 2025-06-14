from .base_util import BaseUtility
from abc import ABC, abstractmethod

class LanguageSpecificUtility(BaseUtility):
    def __init__(self, language: str):
        super(LanguageSpecificUtility, self).__init__(language)