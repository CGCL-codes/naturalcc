# Repo-Specific Semantic Graph

Construct a Repo-Specific Semantic Graph of a project. For simplicity, the Python package name is
called `dependency_graph`.

## Features

1. Multi-language support: Python, Java, C#, JavaScript, TypeScript.
2. Retrieval of cross-file context from the Repo-Specific Semantic Graph.
3. Serialization/Deserialization: edge list in JSON format is supported to output the graph. The graph can also be
   deserialized from JSON/Dictionary.
4. Graph visualizations: [Pyvis](https://pyvis.readthedocs.io/en/latest/)
   and [ipysigma](https://github.com/medialab/ipysigma) interactive visualizations are supported.
5. Analyze on virtual file system: the graph can be constructed from a virtual file system, which is useful for
   real-time analysis when the codebase originates from a dataset and the original file structure is either
   unavailable or cannot be restored.
6. The graph supports topological sorting, allowing the graph to be ordered according to specific relationships. This
   feature is particularly useful for sorting project files based on import dependencies.

Currently, the following Repo-Specific Semantic Graph generator types are supported, with the corresponding languages
and dependency relations:

| **Graph Generator Type** | **Supported Languages**                                                                            | **Supported Dependency Relations**                                                                                                                                    |
|--------------------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Jedi                     | Python                                                                                             | ParentOf/ChildOf, Imports/ImportedBy, BaseClassOf/DerivedClassOf, Overrides/OverriddenBy, Calls/CalledBy, Instantiates/InstantiatedBy, Uses/UsedBy, Defines/DefinedBy |
| Tree-sitter              | Python, Java, C, C++, Kotlin, C#, TypeScript, JavaScript, Go, PHP, Rust, Swift, Ruby, Lua, Bash, R | Imports/ImportedBy                                                                                                                                                    |

## Install

First, install poetry via <https://python-poetry.org/docs/#installation>, then run the following command to create a
Python virtual environment and install the dependencies:

```shell
poetry install
# Activate the virtual environment
poetry shell
```

## Usage

### Command line usage to construct a Repo-Specific Semantic Graph

See `python -m dependency_graph -h` for help:

```shell
~ python -m dependency_graph -h
usage: __main__.py [-h] -r REPO -l LANG [-g GRAPH_GENERATOR] [-f {edgelist,pyvis,ipysigma}] [-o OUTPUT_FILE]

Construct Repo-Specific Semantic Graph for a given project.

options:
  -h, --help            show this help message and exit
  -r REPO, --repo REPO  The path to a local repository.
  -l LANG, --lang LANG  The language of the parsed file.
  -g GRAPH_GENERATOR, --graph-generator GRAPH_GENERATOR
                        The code agent type to use. Should be one of the ['jedi', 'tree_sitter']. Defaults to jedi.
  -f {edgelist,pyvis,ipysigma}, --output-format {edgelist,pyvis,ipysigma}
                        The format of the output.
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        The path to the output file. If not specified, will print to stdout.
```

Example:

```shell
python -m dependency_graph -r </path/to/repo> -l python -g jedi -f edgelist -o edgelist.json
```

### Python API usage

#### Construct a Repo-Specific Semantic Graph

```python
from dependency_graph import construct_dependency_graph
from dependency_graph.models.language import Language
from dependency_graph.graph_generator import GraphGeneratorType
from dependency_graph.models.repository import Repository

dependency_graph_generator = GraphGeneratorType.JEDI

repo = Repository('/path/to/repo', Language.Python)
graph = construct_dependency_graph(repo, dependency_graph_generator)
```

#### Output

```python
from dependency_graph import output_dependency_graph

output_dependency_graph(graph, "edgelist", "graph.json")
# Or
output_dependency_graph(graph, "pyvis", "graph.html")
# Or
output_dependency_graph(graph, "ipysigma", "graph.html")
```

#### Retrieval

Get cross-file context of a specific file path:

```python
graph.as_retriever().get_cross_file_context("/path/into/repo")
```

Get cross-file definition of a file by line:

```python
graph.as_retriever().get_cross_file_definition_by_line("/path/into/repo")
```

#### Filtering

For example, get call graph of the project:

```python
from dependency_graph.models.graph_data import EdgeRelation

graph.get_related_edges(EdgeRelation.Calls)
```

#### Virtual file system

The graph can be constructed from a virtual file system, making it useful for real-time analysis. This approach is
particularly beneficial when the codebase originates from a dataset and the original file structure is either
unavailable or cannot be restored.

```python
from textwrap import dedent
from dependency_graph import construct_dependency_graph
from dependency_graph.models.language import Language
from dependency_graph.graph_generator import GraphGeneratorType
from dependency_graph.models.virtual_fs.virtual_repository import VirtualRepository

dependency_graph_generator = GraphGeneratorType.JEDI
repo = VirtualRepository(
    "repo",
    Language.Python,
    [
        (
            "src/a.py",
            dedent(
                """
                 from b import foo
                 
                 def bar():
                     print("hello")
                     foo()
                 """
            ),
        ),
        (
            "src/b.py",
            dedent(
                """
                 def foo():
                     print("hello")
                 """
            ),
        ),
        (
            "src/c.py",
            dedent(
                """
                 from pkg_not_exist import module
                 """
            ),
        ),
    ],
)
graph = construct_dependency_graph(repo, dependency_graph_generator)

```

#### Topological sorting

The graph supports topological sorting, allowing the graph to be ordered according to specific relationships. This
feature is particularly useful for sorting project files based on import dependencies. Notably, even if the import graph
is cyclic, the graph can still perform the sorting. Furthermore, the sorting process is unique, stable, and consistent.

```python
from dependency_graph import (
    construct_dependency_graph,
    GraphGeneratorType,
)
from dependency_graph.models.language import Language
from dependency_graph.models.graph_data import EdgeRelation

repo_path = "/path/to/repo"
graph = construct_dependency_graph(
    repo_path,
    GraphGeneratorType.TREE_SITTER,
    Language.Python,
)
sorted_nodes = list(graph.get_topological_sorting(EdgeRelation.ImportedBy))

```