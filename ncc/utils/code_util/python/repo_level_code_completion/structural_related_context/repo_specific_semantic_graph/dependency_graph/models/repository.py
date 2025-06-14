from pathlib import Path
from typing import Iterable, Dict, Tuple, List

from dependency_graph.models import PathLike
from dependency_graph.models.language import Language
from dependency_graph.utils.log import setup_logger

# from git import Repo, InvalidGitRepositoryError, GitCommandError, NoSuchPathError

# Initialize logging
logger = setup_logger()


class Repository:
    # _git_repo: Repo = None
    repo_path: Path = None
    language: Language

    code_file_extensions: Dict[Language, Tuple[str]] = {
        Language.CSharp: (".cs", ".csx"),
        Language.Python: (".py", ".pyi"),
        Language.Java: (".java",),
        Language.JavaScript: (".js", ".jsx"),
        Language.TypeScript: (".ts", ".tsx"),
        Language.Kotlin: (".kt", ".kts"),
        Language.PHP: (".php",),
        Language.Ruby: (".rb",),
        Language.C: (".c", ".h"),
        Language.CPP: (".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx", ".c", ".h"),
        Language.Go: (".go", ".mod"),
        Language.Swift: (".swift",),
        Language.Rust: (".rs",),
        Language.Lua: (".lua",),
        Language.Bash: (".sh", ".bash"),
        Language.R: (".r", ".R"),
    }

    def __init__(
        self,
        repo_path: PathLike,
        language: Language,
    ) -> None:
        """Initialize the repository. It will find the code files in the repository according to the language suffixes.
        Args:
            repo_path: Path to the repository.
            language: Language of the repository.
        """
        if isinstance(repo_path, str):
            self.repo_path = Path(repo_path).expanduser().absolute().resolve()
        else:
            self.repo_path = repo_path.expanduser().absolute().resolve()

        if self.repo_path.is_file():
            raise NotADirectoryError(f"Repo path {self.repo_path} is not a directory")

        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repo path {self.repo_path} does not exist")

        if language not in self.code_file_extensions:
            raise ValueError(f"Language {language} is not supported to get code files")

        self._files: List[Path] = []
        # This will trigger the computation of files
        self.language = language

        # try:
        #     self._git_repo = Repo(repo_path)
        # except (InvalidGitRepositoryError, NoSuchPathError):
        #     # The repo is not a git repo, just ignore
        #     pass

    def _compute_files(self) -> List[Path]:
        """Compute the files based on the current language."""

        # Loop through the file extensions
        files = []
        for extension in self.code_file_extensions[self.language]:
            # Use rglob() with a pattern to match the file extension
            rglob_file_list = self.repo_path.rglob(f"*{extension}")
            rglob_file_list = [file for file in rglob_file_list if file.is_file()]

            # # Get the git-ignored files
            # ignored_files = []

            # if self._git_repo:
            #     try:
            #         ignored_files = self._git_repo.ignored(
            #             *list(self.repo_path.rglob(f"*{extension}"))
            #         )
            #     except OSError:
            #         # If the git command argument list is too long, it will raise an OSError.
            #         # In this case, we will invoke the API by iterating through the files one by one
            #         logger.warn(
            #             f"git command argument list is too long, invoking the API by iterating through the files one by one"
            #         )
            #         for file in rglob_file_list:
            #             ignored_files.extend(self._git_repo.ignored(file))
            #     except GitCommandError:
            #         pass

            # # Add the files to the set filtering out git-ignored files
            # self._files.extend(
            #     [file for file in rglob_file_list if str(file) not in ignored_files]
            # )

            files.extend(rglob_file_list)  # Add all found files
        return list(set(files))  # Remove duplicates

    @property
    def files(self) -> List[Path]:
        if not self._files:  # Compute files if they haven't been computed yet
            self._files = self._compute_files()
        return self._files

    @property
    def language(self) -> Language:
        return self._language

    @language.setter
    def language(self, value: Language):
        if value not in self.code_file_extensions:
            raise ValueError(f"Language {value} is not supported.")
        self._language = value
        self._files = self._compute_files()  # Recompute files based on new language

    @files.setter
    def files(self, file_paths: Iterable[Path]):
        """Setter to update the files in the repository."""
        self._files = list(set(file_paths))  # Remove duplicates
