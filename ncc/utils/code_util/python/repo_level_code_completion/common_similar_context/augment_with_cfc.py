import os
import json
import time
import glob
import argparse
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from .rerank_utils import lexical_ranking, SemanticReranking
from .utils import str2bool, file_distance, tokenize_nltk
from multiprocessing import Pool
from collections import defaultdict

CHUNK_SIZE = 10
SLIDING_WINDOW_SIZE = 5  # non-overlapping chunks if SLIDING_WINDOW_SIZE=CHUNK_SIZE
QUERY_LENGTH = 10  # last N lines from prompt will be query

repository_root = "/home/wanyao/talentan/cceval/raw_data/crosscodeeval_rawdata"  # get the data from authors

input_files = {
    "python": "../data/python/line_completion.jsonl",
    # "java": "../data/crosscodeeval_data/java/line_completion.jsonl",
    # "typescript": "../data/crosscodeeval_data/typescript/line_completion.jsonl",
    # "csharp": "../data/crosscodeeval_data/csharp/line_completion.jsonl"
}
output_path = "/home/wanyao/talentan/RepoFuse/data/cross_code_eval/"
file_ext = {"python": "py", "java": "java", "typescript": "ts", "csharp": "cs"}


def get_crossfile_context_from_chunks(
        args,
        prompt,
        code_chunks,
        code_chunk_ids,
        groundtruth,
        semantic_ranker
):
    assert len(code_chunks) != 0
    candidate_code_chunks = code_chunks[:args.maximum_chunk_to_rerank]
    candidate_code_chunk_ids = code_chunk_ids[:args.maximum_chunk_to_rerank]

    ranking_scores = None
    meta_data = {}

    if args.rerank:
        if args.query_type == "groundtruth":
            # oracle experiment
            prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
            groundtruth_lines = [gt for gt in groundtruth.split("\n") if gt.strip()]
            code_lines = prompt_lines + groundtruth_lines
            query = "\n".join(code_lines[-QUERY_LENGTH:])
        elif args.query_type == "last_n_lines":
            prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
            query = "\n".join(prompt_lines[-QUERY_LENGTH:])
        else:
            raise NotImplementedError

        meta_data["query"] = query
        start = time.time()

        if args.ranking_fn == "cosine_sim":
            gpu_id = (int(mp.current_process().name.split('-')[-1])) % args.num_processes
            candidate_code_chunks, candidate_code_chunk_ids, ranking_scores = semantic_ranker.rerank(
                query,
                candidate_code_chunks,
                candidate_code_chunk_ids,
                gpu_id,
                score_threshold=None
            )
        else:
            candidate_code_chunks, candidate_code_chunk_ids, ranking_scores = lexical_ranking(
                query,
                candidate_code_chunks,
                args.ranking_fn,
                candidate_code_chunk_ids,
                score_threshold=None
            )

        meta_data["latency"] = time.time() - start
        meta_data["num_candidates"] = len(candidate_code_chunks)

    top_k = min(args.maximum_cross_file_chunk, len(candidate_code_chunk_ids))
    if top_k == 0:
        return [], meta_data

    selected_chunks = []
    selected_chunks_filename = []
    selected_chunks_scores = []

    if args.use_next_chunk_as_cfc:
        # prepare an id2idx map
        assert len(candidate_code_chunks) == len(candidate_code_chunk_ids)
        id2idx = dict()
        for j, cci in enumerate(code_chunk_ids):
            id2idx[cci] = j

        total_added = 0
        for cidx, _id in enumerate(candidate_code_chunk_ids):
            fname, c_id = _id.rsplit("|", 1)
            next_id = f"{fname}|{int(c_id) + 1}"
            if next_id not in id2idx:
                to_add = code_chunks[id2idx[_id]]
            else:
                to_add = code_chunks[id2idx[next_id]]

            if to_add not in selected_chunks:
                selected_chunks.append(to_add)
                selected_chunks_filename.append(fname)
                if args.rerank:
                    selected_chunks_scores.append(ranking_scores[cidx])
                total_added += 1
                if total_added == top_k:
                    break
    else:
        selected_chunks = candidate_code_chunks[:top_k]
        selected_chunks_filename = [_id.rsplit("|", 1)[0] for _id in candidate_code_chunk_ids[:top_k]]
        if args.rerank:
            selected_chunks_scores = ranking_scores[:top_k]

    cross_file_context = []
    for idx in range(len(selected_chunks)):
        cross_file_context.append({
            "retrieved_chunk": selected_chunks[idx],
            "filename": selected_chunks_filename[idx],
            "score": selected_chunks_scores[idx] if args.rerank else None
        })

    line_start_sym = "#" if args.language == "python" else "//"
    cfc_text = f"{line_start_sym} Here are some relevant code fragments from other files of the repo:\n\n"
    for sc, scf in zip(selected_chunks, selected_chunks_filename):
        cfc_text += f"{line_start_sym} the below code fragment can be found in:\n{line_start_sym} {scf}" + "\n"
        cfc_text += "\n".join([f"{line_start_sym} {cl}" for cl in sc.strip('\n').splitlines()]) + "\n\n"

    return cross_file_context, cfc_text, meta_data


def read_project_files(repo_name, lang):
    "读取指定仓库中的源代码文件，并将文件内容存储到字典中供后续处理使用。"
    # root_dir needs a trailing slash (i.e. /root/dir/)
    # print(f"Reading files for repository: {repo_name}")
    project_context = {}
    root_dir = os.path.join(repository_root, repo_name)
    if not os.path.isdir(root_dir):
        print(f"Repository not found: {root_dir}")
        return project_context

    if lang == "typescript":
        src_files = []
        src_files += glob.glob(os.path.join(root_dir, f'src/**/*.ts'), recursive=True)
        src_files += glob.glob(os.path.join(root_dir, f'src/**/*.tsx'), recursive=True)
    else:
        src_files = glob.glob(os.path.join(root_dir, f'**/*.{file_ext[lang]}'), recursive=True)

    if len(src_files) == 0:
        return project_context

    for filename in src_files:
        if os.path.exists(filename):  # weird but some files cannot be opened to read
            if os.path.isfile(filename):
                try:
                    with open(filename, "r") as file:
                        file_content = file.read()
                except:
                    with open(filename, "rb") as file:
                        file_content = file.read().decode(errors='replace')

                fileid = os.path.relpath(filename, root_dir)
                project_context[fileid] = file_content
                # print(f"Loaded file: {fileid}")
        else:
            print(f"WARNING: File not found: {filename}")
    # print(f"Loaded {len(project_context)} files for {repo_name}")  # 总文件数量
    return project_context


def find_files_within_distance_k(current_file_path, filelist, k):
    list_of_modules = []
    module_weight = []
    for filepath in filelist:
        if filepath != current_file_path:
            dist = file_distance(filepath, current_file_path)
            if dist == -1:
                continue
            elif dist <= k:
                list_of_modules.append(filepath)
                module_weight.append(dist)

    # sorting in ascending order
    list_of_modules = [x for _, x in sorted(zip(module_weight, list_of_modules))]
    return list_of_modules


def get_cfc(example, args, semantic_ranker, repositories):
    project_context = repositories[example["metadata"]["repository"]]
    status = None
    current_filepath = example["metadata"]["file"]
    if len(project_context) == 0:
        example["crossfile_context"] = ""
        status = "project_not_found"
    else:
        current_filecontent = None
        for filepath, filecontent in project_context.items():
            if filepath == current_filepath:
                current_filecontent = filecontent
                break

        if current_filecontent is None:
            example["crossfile_context"] = {}
            print(current_filepath)
            status = "file_not_found_in_project"

        else:
            pyfiles = find_files_within_distance_k(
                example["metadata"]["file"],
                list(project_context.keys()),
                k=args.crossfile_distance
            )
            pyfiles = pyfiles[:args.maximum_cross_files]

            code_chunks = []
            code_chunk_ids = []
            for pyfile in pyfiles:
                lines = project_context[pyfile].split("\n")
                lines = [l for l in lines if l.strip()]  # removing empty lines
                c_id = 0
                for i in range(0, len(lines), SLIDING_WINDOW_SIZE):
                    c = "\n".join(lines[i:i + CHUNK_SIZE])
                    tokenized_c = tokenize_nltk(c)
                    if len(tokenized_c) > 0:
                        code_chunks.append(c)
                        code_chunk_ids.append(f"{pyfile}|{c_id}")
                        c_id += 1

            if len(code_chunks) == 0:
                example["crossfile_context"] = {}
                status = "no_crossfile_context"

            else:
                cfc, cfc_text, meta_data = get_crossfile_context_from_chunks(
                    args=args,
                    prompt=example["prompt"],
                    code_chunks=code_chunks,
                    code_chunk_ids=code_chunk_ids,
                    groundtruth=example["groundtruth"],
                    semantic_ranker=semantic_ranker
                )
                example["crossfile_context"] = {}
                example["crossfile_context"]["text"] = cfc_text
                example["crossfile_context"]["list"] = cfc

    return example, status

def process_repository(repo_name, repo_examples, args, semantic_ranker):
    # print(f"Processing repository: {repo_name}")
    project_context = read_project_files(repo_name, args.language)
    if not project_context:
        print(f"Repository {repo_name} has no valid files, skipping")
        return []
    
    # print(f"Processing {len(repo_examples)} examples in {repo_name}")  # 示例数量
    worker = partial(
        get_cfc,
        args=args,
        semantic_ranker=semantic_ranker,
        repositories={repo_name: project_context}
    )

    processed = []
    for ex in repo_examples:
        d, stat = worker(ex)
        processed.append(d)
    # print(f"Finished processing {len(processed)} examples in {repo_name}")  # 完成处理
    return processed

def process_repository_wrapper(task):
    # task 是一个包含 (repo, examples) 的元组
    repo, examples = task
    return process_repository(repo, examples, global_args, global_semantic_ranker)

def init_worker(args, semantic_ranker):
    global global_args, global_semantic_ranker
    global_args = args
    global_semantic_ranker = semantic_ranker
def attach_data(args, srcfile):

    # print(f"Reading input file: {srcfile}")
    examples = []
    with open(srcfile) as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)
        for idx, line in enumerate(tqdm(f, desc="Reading input file", total=total_lines, leave=True)):
            ex = json.loads(line)
            ex["_original_index"] = idx
            examples.append(ex)
    # print(f"Loaded {len(examples)} examples from input file")

    examples_by_repo = defaultdict(list)
    for ex in examples:
        repo_name = ex["metadata"]["repository"]
        examples_by_repo[repo_name].append(ex)
    # print(f"Found {len(examples_by_repo)} repositories to process")

    global_semantic_ranker = None
    if args.ranking_fn == "cosine_sim":
        global_semantic_ranker = SemanticReranking(
            args.ranker,
            max_sequence_length=512
        )

    tasks = [(repo, repo_examples) for repo, repo_examples in examples_by_repo.items()]
    with Pool(processes=args.num_processes, initializer=init_worker, initargs=(args, global_semantic_ranker)) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_repository_wrapper, tasks),
                        total=len(tasks), desc="Processing repositories"):
            results.append(result)
    # print(f"Finished processing {len(results)} repositories")

    all_processed = []
    for res in results:
        all_processed.extend(res)
    all_processed.sort(key=lambda x: x["_original_index"])
    for ex in all_processed:
        del ex["_original_index"]

    return all_processed


if __name__ == "__main__":
    print("Script started execution...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rerank",
        type=str2bool,
        default=True,
        help="rerank the functions"
    )
    parser.add_argument(
        "--ranker",
        type=str,
        default="sparse",
        choices=["sparse", "unixcoder", "codebert"],
        help="ranking function"
    )
    parser.add_argument(
        "--ranking_fn",
        type=str,
        default="bm25",
        choices=["tfidf", "bm25", "jaccard_sim", "cosine_sim", "edit_distance"],
        help="ranking function"
    )
    parser.add_argument(
        "--query_type",
        type=str,
        default="last_n_lines",
        choices=["last_n_lines", "groundtruth"],
        help="how to form query from prompt"
    )
    parser.add_argument(
        "--crossfile_distance",
        type=int,
        default=100,
        help="max distance to search for crossfile"
    )
    parser.add_argument(
        "--maximum_chunk_to_rerank",
        type=int,
        default=1000,
        help="max chunks to consider to rank via BM25"
    )
    parser.add_argument(
        "--maximum_cross_files",
        type=int,
        default=1000,
        help="max chunks to consider to rank via BM25"
    )
    parser.add_argument(
        "--maximum_cross_file_chunk",
        type=int,
        default=10,
        help="max chunks to return as cfc"
    )
    parser.add_argument(
        "--use_next_chunk_as_cfc",
        type=str2bool,
        default=False,
        help="use next code chunk as context"
    )
    parser.add_argument(
        "--skip_if_no_cfc",
        type=str2bool,
        default=False,
        help="skip adding examples if there is no crossfile context"
    )
    parser.add_argument(
        "--output_file_suffix",
        type=str,
        default=None,
        help="add a suffix string to the output file"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["java", "python", "typescript", "csharp"],
        help="language name"
    )
    args = parser.parse_args()

    args.output_file_suffix = "" if args.output_file_suffix is None else f"_{args.output_file_suffix}"
    if args.use_next_chunk_as_cfc:
        assert args.rerank
        assert args.query_type != "groundtruth"

    tgtfile_suffix = ""
    if args.rerank:
        tgtfile_suffix += f"_{args.ranking_fn}"

    args.num_processes = 60
    if args.ranking_fn == "cosine_sim":
        num_gpus = 7
        args.num_processes = num_gpus
        mp.set_start_method('spawn')
    tqdm_lock = mp.Manager().Lock()

    input_file = input_files[args.language]
    output_path = os.path.dirname(output_path)
    output_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_filename = output_filename + args.output_file_suffix + tgtfile_suffix + ".jsonl"
    output_file = os.path.join(output_path, output_filename)
    print("Parsed arguments:")
    print(f"Language: {args.language}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Number of processes: {args.num_processes}")

    output_examples = attach_data(args, input_file)
    with open(output_file, "w") as fw:
        for ex in output_examples:
            fw.write(json.dumps(ex))
            fw.write("\n")
