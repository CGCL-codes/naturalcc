# Data Processing Pipeline

This document describes the complete data processing pipeline for the Figma2Code project, from initial data collection to final dataset preparation.

## How to Run Scripts

Most scripts in `src/data_script/` use relative imports (e.g., `from ...configs...`), so run them from the project root with module mode:

```bash
python -m src.data_script.<submodule>.<script_name>
```

For example:

```bash
python -m src.data_script.rule_based_filtering.json_filter
```

## Overview

The data processing pipeline consists of five main stages:

1. **Raw Data Collection**: Crawl Figma file keys and download page screenshots and metadata
2. **Rule-based Filtering**: Filter samples based on JSON metadata, similarity, and LLM scoring
3. **Manual Selection and Annotation**: Human annotators filter and annotate samples using Gradio interface
4. **Test Set Sampling**: Stratified sampling and professional designer selection for test data
5. **Final Dataset Processing**: Download final dataset and process metadata

## Directory Structure

```txt
output/page_filter/
├── candidate/              # Initial downloaded pages
│   ├── page_stats.json
│   └── {filekey}/
│       ├── {filekey}.json
│       └── {page_id}.png
├── selected/              # Manually selected pages
│   └── {filekey}/
│       └── {page_id}.png
├── json_filtered/         # After JSON-based filtering
│   └── {filekey}/
│       └── summary.json
├── clipsim_filtered/      # After similarity filtering
│   ├── intra_filekey/     # Intra-group filtered
│   │   ├── {filekey}/
│   │   │   ├── {page_id}.png
│   │   │   └── summary.json
│   └── inter_filekey/     # Inter-group filtered
│       ├── {filekey}/
│       │   ├── {page_id}.png
│       │   └── summary.json
│       └── inter_filekey_removed/  # Removed samples
├── split_set/             # Split datasets for annotation
│   └── split_{i}/
│       ├── candidate/
│       ├── selected/
│       ├── trash/
│       ├── final_annotation.json
│       └── report.json
├── whitelist.csv          # Initial whitelist
├── whitelist_filter.csv   # After manual annotation
└── whitelist_categorized.csv  # With content categories
```

---

## Stage 1: Raw Data Collection

### Step 1.1: Crawl File Keys

Use `raw_data_collection/filekey_crawl.py` to crawl Figma file keys from community templates. The script extracts file keys from various Figma community categories (website templates, blog templates, web ads, etc.).

**Requirements:**

- Windows OS with Chrome browser installed
- Figma account credentials configured in `.env` or `settings.py`

**Output:** `data/file_keys.txt` containing all crawled file keys

```bash
python -m src.data_script.raw_data_collection.filekey_crawl
```

### Step 1.2: Split File Keys (Optional)

For parallel downloading by multiple annotators, use `raw_data_collection/filekeysplit.py` to split `data/file_keys.txt` into multiple files.

```bash
python -m src.data_script.raw_data_collection.filekeysplit
```

This generates `data/filekeys_split0.txt`, `data/filekeys_split1.txt`, etc. (default split count is 4 in script).

### Step 1.3: Download Page Screenshots and Metadata

Use `raw_data_collection/figma_page_filter.py` to download design page screenshots and metadata.

**Download mode:**

```bash
python -m src.data_script.raw_data_collection.figma_page_filter \
    --mode download \
    --filekeys data/filekeys.txt
```

**Options:**

- `--filekeys`: Path to file containing file keys (one per line)
- `--excluded_canvas_patterns`: Patterns to filter out based on canvas names (optional)

**Output:**

- `output/page_filter/candidate/`: Contains downloaded pages organized by filekey
- `output/page_filter/candidate/page_stats.json`: Statistics of all downloaded pages

Each filekey directory contains:

- `{filekey}.json`: Raw Figma metadata
- `{page_id}.png`: Screenshot of each page

### Step 1.4: Manual Filtering

Manually review and delete unwanted samples from `output/page_filter/candidate/` directories.

### Step 1.5: Generate Whitelist

After manual filtering, generate the initial whitelist CSV:

```bash
python -m src.data_script.raw_data_collection.figma_page_filter --mode filter
```

**Output:**

- `output/page_filter/selected/`: Contains manually selected pages
- `output/page_filter/whitelist.csv`: CSV file with selected samples

---

## Stage 2: Rule-based Filtering

### Step 2.1: JSON-based Filtering

Filter samples based on JSON metadata criteria (aspect ratio, children count, image coverage, etc.).

```bash
python -m src.data_script.rule_based_filtering.json_filter
```

**Input:** `output/page_filter/candidate/`

**Output:** `output/page_filter/json_filtered/{filekey}/summary.json`

Each `summary.json` contains:

- `file_key`: File identifier
- `retained_pages_count`: Number of retained pages
- `removed_pages_count`: Number of removed pages
- `retained_node_ids`: List of retained page IDs
- `removed_node_ids_with_reason`: List of removed page IDs with removal reasons

### Step 2.2: Similarity Filtering

Filter samples based on CLIP embeddings using intra-group and inter-group similarity.

```bash
python -m src.data_script.rule_based_filtering.similarity_filter
```

**Input:** `output/page_filter/json_filtered/`

**Output:**

- `output/page_filter/clipsim_filtered/intra_filekey/`: Samples after intra-group filtering (removes duplicates within each filekey)
- `output/page_filter/clipsim_filtered/inter_filekey/`: Samples after inter-group filtering (removes duplicates across filekeys)

Each directory contains:

- `{filekey}/{page_id}.png`: Filtered page images
- `{filekey}/summary.json`: Filtering summary

### Step 2.3: LLM Score Filtering

Filter samples based on LLM quality scoring.

```bash
python -m src.data_script.rule_based_filtering.score_filter
```

**Input:** `output/page_filter/clipsim_filtered/inter_filekey/`

**Output:**

- `output/page_filter/clipsim_filtered/inter_filekey/inter_filekey_removed/`: Low-quality samples moved here
- Updated `filter.json` files with filtering results
- `output/page_filter/clipsim_filtered/score_filter_global_summary.json`: Global filtering summary
- `output/page_filter/clipsim_filtered/score_distributions.png`: Score distributions before/after filtering

**Note:** This step directly moves low-scoring samples to the `removed` directory and updates the filter metadata.

---

## Stage 3: Manual Selection and Annotation

### Step 3.1: Split Dataset for Annotation

Split the filtered dataset into multiple parts for parallel annotation by multiple annotators.

```bash
python -m src.data_script.annotation.dataset_split_package
```

**Input:** `output/page_filter/clipsim_filtered/inter_filekey/`

**Output:**

- `output/page_filter/split_set/split_{i}/` directories (for local use)
- `output/page_filter/split_set/split_{i}_with_gradio.zip` packages (for annotators)

Each split package contains:

- `candidate/`: Assigned samples for annotation
- `selected/`: Samples selected by annotator
- `trash/`: Samples rejected by annotator
- `final_annotation.json`: Annotation results
- `report.json`: Annotation statistics
- `gradio_filter_and_annotation.py`: Gradio annotation script

### Step 3.2: Manual Annotation

Annotators extract the zip file and run the Gradio annotation interface:

```bash
python gradio_filter_and_annotation.py --base_dir <split_dir>
```

The Gradio interface allows annotators to:

- View page images
- Filter samples (select/trash)
- Add annotations (platform, content type, quality rating, etc.)

**Output:** `final_annotation.json` in each split directory

### Step 3.3: Collect Annotations

Collect all `final_annotation.json` files from annotators and place them in the corresponding `output/page_filter/split_set/split_{i}/` directories.

### Step 3.4: Generate Filtered Whitelist

Process manual annotations and generate the filtered whitelist:

```bash
python -m src.data_script.annotation.annotate_and_generate_whitelist
```

**Input:** `output/page_filter/split_set/` (with `final_annotation.json` files)

**Output:** `output/page_filter/split_set/whitelist_filter.csv`

This CSV contains samples that passed manual filtering with additional annotations.

### Step 3.5: Categorize Content Types

Map content types to broader categories:

```bash
python -m src.data_script.annotation.content_category_mapper
```

**Input:** `output/page_filter/split_set/whitelist_filter.csv`

**Output:** `output/page_filter/whitelist_categorized.csv`

This step groups detailed content types into major categories for better organization.

### Step 3.6: Compute Complexity (Optional Analysis Script)

Calculate complexity metrics and matching rates:

```bash
python -m src.data_script.annotation.complexity_compute
```

**Input:** `data/data_test/*/report.json` (uses `DATA_TEST_DIR` in code)

**Output:** `complexity_results.csv` (written to current working directory)

This generates complexity statistics and matching rates for each sample.

---

## Stage 4: Test Set Sampling and Manual Selection

### Step 4.1: Sample Test Data Candidates

Use stratified sampling to select test set candidates:

```bash
python -m src.data_script.dataset_partition.filter_test_data --mode sample --sample_num 160
```

This runs `sample_test_data()` to generate test candidates based on:

- Platform distribution
- Complexity distribution
- Quality rating distribution
- Content type distribution

**Input:** `output/page_filter/whitelist_categorized.csv`

**Output:** Test candidate samples (typically ~160 samples)

### Step 4.2: Professional Designer Selection

Professional designers review the sampled candidates and make final selections.

### Step 4.3: Generate Final Test CSV

After designer selection, generate the final test dataset CSV:

```bash
python -m src.data_script.dataset_partition.filter_test_data --mode generate
```

This runs `generate_final_csv()` to create the final test dataset.

**Output:** `output/page_filter/test_data_selected_final.csv`

---

## Stage 5: Download Final Dataset and Process Metadata

### Step 5.1: Process Final Dataset

Download the final dataset and process metadata for all selected samples:

```bash
python -m src.data_script.metadata_refinement.figma_metadata_process \
    whitelist \
    output/page_filter/test_data_selected_final.csv
```

**Input:** Final whitelist CSV file

**Output (default CLI behavior):** `output/processed_full/{file_key}_{safe_filename(node_id)}/`

Each sample directory contains:

- `raw.json`: Raw Figma JSON data
- `processed_metadata.json`: Processed and optimized metadata
- `report.json`: Detailed processing report (includes annotation/statistics/resource mappings)
- `root.png`: Rendered image of the root node
- `assets/`: Downloaded assets
  - `components_json/`: Component JSON data
  - `component_sets_json/`: Component set JSON data
  - `styles_json/`: Style JSON data
  - `image_refs/`: Image reference images
  - `svg_assets/`: SVG asset files
- `render/`: Rendered images
  - `components_img/`: Component images
  - `nodes_img/`: Node images

`report.json` key structure (important):

- `file_key`, `node_id`, `page_url`
- `annotation`: annotation fields carried from whitelist CSV (`platform`, `complexity`, `quality_rating`, `theme`, `language`, `content`, `description`, `content_original`)
- `statistics`: node/resource statistics
- `resource_info`: collected resource ids/keys before download
- `downloaded_resources`: downloaded resource path mappings
- `duplicate_mapping`: deduplication mapping when dedup is enabled

**Processing steps:**

1. Data acquisition: Fetch raw JSON and render root node image
2. Data preprocessing: Node optimization (attribute filtering, useless node removal, asset node merging, coordinate normalization)
3. Layer optimization: Redundant nesting filtering, layout recognition
4. Resource collection: Collect resource IDs and keys
5. Resource downloading: Download all necessary resource files
6. Data statistics: Generate detailed processing reports

---

## Configuration

### Environment Variables

Set the following in `.env` or `src/configs/settings.py`:

- `FIGMA_API_KEY`: Figma API key (required for most API-based downloading/metadata scripts)
- `FIGMA_EMAIL`: Your Figma account email
- `FIGMA_PASSWORD`: Your Figma account password
- `OPENROUTER_API_KEY`: API key for LLM-based filtering (optional)

### Filtering Parameters

Each filtering script has configurable parameters:

- **JSON Filter**: Aspect ratio thresholds, children count limits, image coverage thresholds
- **Similarity Filter**: CLIP similarity thresholds for intra-group and inter-group filtering
- **Score Filter**: LLM scoring thresholds and model selection

Refer to individual script files for detailed configuration options.

---

## Workflow Summary

```txt
1. Crawl file keys → data/file_keys.txt
2. Split file keys (optional) → data/filekeys_split*.txt
3. Download pages → output/page_filter/candidate/
4. Manual filtering → output/page_filter/selected/
5. Generate whitelist → output/page_filter/whitelist.csv
6. JSON filtering → output/page_filter/json_filtered/
7. Similarity filtering → output/page_filter/clipsim_filtered/inter_filekey/
8. Score filtering → output/page_filter/clipsim_filtered/inter_filekey/ (removed samples moved)
9. Split for annotation → output/page_filter/split_set/split_*.zip
10. Manual annotation → final_annotation.json files
11. Generate filtered whitelist → output/page_filter/split_set/whitelist_filter.csv
12. Categorize content → output/page_filter/whitelist_categorized.csv
13. Sample test data → test candidates
14. Designer selection → output/page_filter/test_data_selected_final.csv
15. Process final dataset → output/processed_full/{file_key}_{node_id}/
```

---

## Notes

- All paths are relative to the project root directory
- Intermediate results are preserved for debugging and reprocessing
- The pipeline supports resuming from any stage if intermediate outputs exist
- Manual filtering steps require human review and cannot be fully automated
- Ensure sufficient disk space for downloaded images and metadata (can be several GB)
