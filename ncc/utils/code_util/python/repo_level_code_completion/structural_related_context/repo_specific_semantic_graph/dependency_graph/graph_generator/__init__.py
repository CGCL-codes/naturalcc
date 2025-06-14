import enum
from abc import ABC, abstractmethod, ABCMeta
from functools import wraps
from typing import Tuple

from dependency_graph.dependency_graph import DependencyGraph
from dependency_graph.models import PathLike
from dependency_graph.models.language import Language
from dependency_graph.models.repository import Repository


class GraphGeneratorType(str, enum.Enum):
    """
    Graph generator type
    """

    JEDI = "jedi"
    TREE_SITTER = "tree_sitter"


def validate_language(method):
    """
    Decorator to validate that the language of the repository as the first argument is validated.
    """

    @wraps(method)
    def wrapper(self, repo: Repository, *args, **kwargs):
        self._validate_language(repo.language)
        return method(self, repo, *args, **kwargs)

    return wrapper


class BaseDependencyGraphGeneratorMeta(ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        This metaclass ensures that any implementation of 'generate_file'/'generate' in a subclass
        will automatically have the 'validate_language' decorator applied to it.
        """
        for attr, value in namespace.items():
            if attr in ("generate_file", "generate") and callable(value):
                namespace[attr] = validate_language(value)
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class BaseDependencyGraphGenerator(ABC, metaclass=BaseDependencyGraphGeneratorMeta):
    supported_languages: Tuple[Language] = ()

    def _validate_language(self, language: Language):
        if language not in self.supported_languages:
            raise ValueError(
                f"Language {language} is not supported by graph generator {self.__class__.__name__}"
            )

    @abstractmethod
    def generate_file(
        self,
        repo: Repository,
        code: str = None,
        file_path: PathLike = None,
    ) -> DependencyGraph:
        """
        Generate Repo-Specific Semantic Graph for a file.
        Should provide either code or file_path
        """
        ...

    @abstractmethod
    def generate(self, repo: Repository) -> DependencyGraph: ...
