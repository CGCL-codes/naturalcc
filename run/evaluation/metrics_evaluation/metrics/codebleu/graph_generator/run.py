import os
from argparse import ArgumentParser, Namespace
from glob import iglob
from typing import List, Dict, Iterable, Tuple

from dpu_utils.utils import ChunkWriter
from tqdm import tqdm

from graph_generator.graphgenerator import AstGraphGenerator
from graph_generator.type_lattice_generator import TypeLatticeGenerator


def get_files(root: str):
    return [
        fname
        for fname in iglob(os.path.join(root, "**/*.py"), recursive=True)
        if not os.path.isdir(fname)
    ]


def generate_graphs_for_files(
    fnames: List[str],
    skipped_files: List[str],
    errored_files: List[Tuple[str, Exception]],
    lattice: TypeLatticeGenerator,
) -> Iterable[Tuple[AstGraphGenerator, Dict, str]]:
    for fname in fnames:
        try:
            with open(fname) as f:
                generator = AstGraphGenerator(f.read(), lattice)
                graph = generator.build()
                yield generator, graph, fname
        except (SyntaxError, UnicodeDecodeError):
            skipped_files.append(fname)
        except Exception as e:
            errored_files.append((fname, e))


def run(args: Namespace):
    filenames = get_files(args.input)
    lattice = TypeLatticeGenerator(args.type_rules)
    skipped_files, errored_files = [], []
    graph_generator = generate_graphs_for_files(
        filenames, skipped_files, errored_files, lattice
    )

    os.makedirs(args.output, exist_ok=True)

    if args.format == "dot":
        for i, (generator, graph, fname) in tqdm(enumerate(graph_generator)):
            generator.to_dot(
                os.path.join(args.output, f"graph_{i}.dot"), initial_comment=fname
            )
    elif args.format == "jsonl_gz":
        with ChunkWriter(
            out_folder=args.output,
            file_prefix="all-graphs",
            max_chunk_size=5000,
            file_suffix=".jsonl.gz",
        ) as writer:
            for i, (generator, graph, fname) in tqdm(enumerate(graph_generator)):
                graph["filename"] = fname
                writer.add(graph)
    else:
        raise ValueError(f"File format {args.format} is not supported")
    print(
        f"Finished parsing. Skipped {len(skipped_files)}, failed on {len(errored_files)}"
    )
    if len(skipped_files) > 0:
        print("Skipped:")
        for fname in skipped_files:
            print(fname)
    if len(errored_files) > 0:
        print("Failed on:")
        for fname, e in errored_files:
            print(fname)
            if args.print_errors:
                print(str(e))


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Directory that will be parsed recursively.",
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Directory where the output will be stored.",
    )
    arg_parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="dot",
        help="Output format, either `dot` (default) or `jsonl_gz`.",
    )
    arg_parser.add_argument(
        "--type-rules",
        type=str,
        default="../metadata/typingRules.json",
        help="Path to json with type rules. Do not change, unless you extract data for type inference.",
    )
    arg_parser.add_argument(
        "--print-errors",
        action="store_true",
        help="If passed, the errors for all failed files will be printed.",
    )
    args = arg_parser.parse_args()
    run(args)
