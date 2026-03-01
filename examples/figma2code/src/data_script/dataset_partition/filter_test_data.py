#!/usr/bin/env python3
"""
Filter whitelist.csv data to select a certain number of samples as test data
Adopt a stratified sampling method: divide by platform, then by complexity, then by quality, and finally by content
"""

import pandas as pd
import numpy as np
import random
import os
import shutil
import json
import zipfile
from collections import defaultdict
from pathlib import Path
def load_and_analyze_data(csv_path):
    """Load data and analyze distribution"""
    df = pd.read_csv(csv_path)
    print(f"Total data volume: {len(df)}")
    
    # Analyze the distribution of each dimension
    print("\n=== Data Distribution Analysis ===")
    
    # Platform distribution
    platform_dist = df['platform'].value_counts()
    platform_pct = df['platform'].value_counts(normalize=True) * 100
    print(f"\nPlatform distribution:")
    for platform in platform_dist.index:
        print(f"  {platform}: {platform_dist[platform]} ({platform_pct[platform]:.1f}%)")
    
    # Complexity distribution
    complexity_dist = df['complexity'].value_counts()
    complexity_pct = df['complexity'].value_counts(normalize=True) * 100
    print(f"\nComplexity distribution:")
    for complexity in complexity_dist.index:
        print(f"  {complexity}: {complexity_dist[complexity]} ({complexity_pct[complexity]:.1f}%)")
    
    # Quality distribution
    quality_dist = df['quality_rating'].value_counts().sort_index()
    quality_pct = df['quality_rating'].value_counts(normalize=True).sort_index() * 100
    print(f"\nQuality distribution:")
    for quality in quality_dist.index:
        print(f"  {quality}: {quality_dist[quality]} ({quality_pct[quality]:.1f}%)")
    
    # Content type distribution
    content_dist = df['content'].value_counts()
    content_pct = df['content'].value_counts(normalize=True) * 100
    print(f"\nContent type distribution:")
    for content in content_dist.index:
        print(f"  {content}: {content_dist[content]} ({content_pct[content]:.1f}%)")
    
    # The following two are for reference only and not as filtering criteria
    
    # Theme distribution
    theme_dist = df['theme'].value_counts()
    theme_pct = df['theme'].value_counts(normalize=True) * 100
    print(f"\nTheme distribution:")
    for theme in theme_dist.index:
        print(f"  {theme}: {theme_dist[theme]} ({theme_pct[theme]:.1f}%)")
    
    # Language distribution
    language_dist = df['language'].value_counts()
    language_pct = df['language'].value_counts(normalize=True) * 100
    print(f"\nLanguage distribution:")
    for language in language_dist.index:
        print(f"  {language}: {language_dist[language]} ({language_pct[language]:.1f}%)")
    
    return df

def organize_by_layer_structure(df, base_dir):
    """Organize images according to the directory structure <platform>_<complexity>_<quality_rating>_<content>/<filekey>"""
    img_source_base_dir = "./output/page_filter/candidate"
    
    # Clear base_dir first
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Record statistics for each layer
    layer_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    
    # Traverse the data frame and organize images according to the hierarchical structure
    for _, row in df.iterrows():
        platform = row['platform']
        complexity = row['complexity']
        quality = row['quality_rating']
        content = row['content']
        filekey = row['file_key']
        page_id = row['node_id']
        
        # Create target directory
        target_dir = os.path.join(base_dir, platform + "_" + complexity + "_" + str(quality) + "_" + content, filekey)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy image
        source_path = f"{img_source_base_dir}/{filekey}/{page_id}.png"
        target_path = os.path.join(target_dir, f"{page_id}.png")
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            # Update statistics
            layer_stats[platform][complexity][quality][content] += 1
        else:
            print(f"Warning: Source image does not exist {source_path}")
    
    return layer_stats

def hierarchical_stratified_sampling(df, num_samples):
    """Perform hierarchical stratified sampling by platform->complexity->quality->content, excluding data with quality_rating of 1"""
    df = df[df['quality_rating'] != 1]
    print(f"Data volume after filtering out quality_rating=1 for stratified sampling: {len(df)}")
    
    # Create hierarchical index
    df_grouped = df.groupby(['platform', 'complexity', 'quality_rating', 'content'])
    
    # Calculate the number of samples for each combination
    group_counts = df_grouped.size().reset_index(name='count')
    total_samples = len(df)
    
    # Calculate the target number of samples for each combination
    group_counts['target_samples'] = (group_counts['count'] / total_samples * num_samples).round().astype(int)
    
    # Ensure at least 1 sample for each combination (if present in the original data)
    group_counts['target_samples'] = group_counts.apply(
        lambda x: min(max(1, x['target_samples']), x['count']), axis=1
    )
    
    # # Adjust the total number of samples to ensure it does not exceed the target
    # while group_counts['target_samples'].sum() > num_samples:
    #     # Find the group with the most samples and reduce by 1
    #     max_idx = group_counts[group_counts['target_samples'] > 1]['target_samples'].idxmax()
    #     group_counts.loc[max_idx, 'target_samples'] -= 1
    
    # If the total number of samples is insufficient, increase it
    while group_counts['target_samples'].sum() < num_samples:
        # Find the group with the largest gap from the target ratio and add 1 sample
        group_counts['current_ratio'] = group_counts['target_samples'] / group_counts['target_samples'].sum()
        group_counts['target_ratio'] = group_counts['count'] / total_samples
        group_counts['diff'] = group_counts['target_ratio'] - group_counts['current_ratio']
        
        # Only consider groups that have additional samples to add
        valid_groups = group_counts[group_counts['target_samples'] < group_counts['count']]
        if len(valid_groups) == 0:
            break
        
        max_diff_idx = valid_groups['diff'].idxmax()
        group_counts.loc[max_diff_idx, 'target_samples'] += 1
    
    # Create sampling statistics and results
    sampling_stats = {}
    selected_indices = []

    # Actual number of required samples (excluding redundancy)
    actual_target_samples = 0
    
    # Sample each combination and add redundant samples
    for _, row in group_counts.iterrows():
        platform = row['platform']
        complexity = row['complexity']
        quality = row['quality_rating']
        content = row['content']
        target_count = row['target_samples']
        total_count = row['count']
        actual_target_samples += target_count

        # Calculate the number of redundant samples
        # Keep at least min(total group samples, 6) samples
        # For required samples between 3 and 5, keep min(total group samples, 3 * required samples)
        if target_count >= 3 and target_count <= 5:
            redundant_count = min(total_count, 3 * target_count)
        elif target_count > 5:
            # For required samples > 5, keep min(total group samples, 2 * required samples)
            redundant_count = min(total_count, 2 * target_count)
        else:
            redundant_count = min(total_count, 6)
        
        # Ensure the number of redundant samples is at least equal to the target number of samples
        redundant_count = max(redundant_count, target_count)
        
        # Get all samples for this combination
        group_df = df[
            (df['platform'] == platform) & 
            (df['complexity'] == complexity) & 
            (df['quality_rating'] == quality) & 
            (df['content'] == content)
        ]
        
        # Record statistics
        if platform not in sampling_stats:
            sampling_stats[platform] = {}
        if complexity not in sampling_stats[platform]:
            sampling_stats[platform][complexity] = {}
        if quality not in sampling_stats[platform][complexity]:
            sampling_stats[platform][complexity][quality] = {}
        
        sampling_stats[platform][complexity][quality][content] = {
            'total': len(group_df),
            'sampled': target_count,
            'redundant': redundant_count
        }
        
        if len(group_df) <= redundant_count:
            # If the number of available samples is insufficient, select all
            selected = group_df.index.tolist()
            selected_indices.extend(selected)
        else:
            # Randomly select the specified number of samples (including redundant ones)
            selected = group_df.sample(n=redundant_count, random_state=42).index.tolist()
            selected_indices.extend(selected)
    
    print(f"\n=== Stratified Sampling Strategy ===")
    print(f"Target number of samples: {num_samples}")
    print(f"Actual number of selected samples (including redundancy): {len(selected_indices)}")
    print(f"Actual number of selected samples (excluding redundancy): {actual_target_samples}")
    
    return selected_indices, sampling_stats

def save_sampling_stats(sampling_stats, output_path):
    """Save sampling statistics to a JSON file and sampling records"""
    # Save sampling statistics to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampling_stats, f, ensure_ascii=False, indent=2)
    print(f"Sampling statistics have been saved to: {output_path}")

def copy_selected_images(df, selected_indices, source_dir, target_dir, sampling_stats):
    """Copy the sampled images to the target directory according to the hierarchical structure, create a file for the required number of samples for each group, and finally compress it into a zip file"""
    selected_df = df.loc[selected_indices]
    
    # Clear target_dir first
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    # Create base directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Record all copied file paths for creating the zip
    copied_files = []
    
    # Traverse the selected data and copy the images
    for _, row in selected_df.iterrows():
        platform = row['platform']
        complexity = row['complexity']
        quality = row['quality_rating']
        quality_str = str(quality)
        content = row['content']
        filekey = row['file_key']
        page_id = row['node_id']
        
        # Handle special characters in the file name to prevent problems on Windows
        safe_page_id = page_id.replace(':', '_').replace(';', '-')
        
        # Source and target paths
        source_path = os.path.join(source_dir, platform + "_" + complexity + "_" + quality_str + "_" + content, filekey, f"{page_id}.png")
        target_path_dir = os.path.join(target_dir, platform + "_" + complexity + "_" + quality_str + "_" + content)
        target_filename = f"{filekey}--{safe_page_id}.png"
        target_path = os.path.join(target_path_dir, target_filename)
        
        # Create target directory
        os.makedirs(target_path_dir, exist_ok=True)
        
        # Copy image
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            copied_files.append(target_path)
        else:
            print(f"Warning: Source image does not exist {source_path}")
    
    # Create a <num>.txt file for each group to record the required number of samples
    for platform in sampling_stats:
        for complexity in sampling_stats[platform]:
            for quality in sampling_stats[platform][complexity]:
                for content in sampling_stats[platform][complexity][quality]:
                    # Get the target number of samples
                    target_count = sampling_stats[platform][complexity][quality][content]['sampled']
                    
                    # Create group directory path
                    group_dir = os.path.join(target_dir, platform + "_" + complexity + "_" + str(quality) + "_" + content)
                    if os.path.exists(group_dir):
                        # Create <num>.txt file
                        num_file_path = os.path.join(group_dir, f"{target_count}.txt")
                        with open(num_file_path, 'w') as f:
                            f.write(f"This group needs to select {target_count} samples")
                        copied_files.append(num_file_path)
    
    # Create zip file
    zip_path = f"{target_dir}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in copied_files:
            # Add the file to the zip, keeping the relative path
            arcname = os.path.relpath(file, os.path.dirname(target_dir))
            zipf.write(file, arcname)
    
    print(f"Copied the selected images to: {target_dir}")
    print(f"And created a compressed file: {zip_path}")
    print("Created a <num>.txt file for each group to indicate the required number of samples")

def analyze_selection(df, selected_indices):
    """Analyze selection results"""
    selected_df = df.loc[selected_indices]
    
    print(f"\n=== Selection Result Analysis ===")
    print(f"Number of selected samples: {len(selected_df)}")
    
    # Platform distribution
    platform_dist = selected_df['platform'].value_counts()
    platform_pct = selected_df['platform'].value_counts(normalize=True) * 100
    print(f"\nPlatform distribution:")
    for platform in platform_dist.index:
        print(f"  {platform}: {platform_dist[platform]} ({platform_pct[platform]:.1f}%)")
    
    # Complexity distribution
    complexity_dist = selected_df['complexity'].value_counts()
    complexity_pct = selected_df['complexity'].value_counts(normalize=True) * 100
    print(f"\nComplexity distribution:")
    for complexity in complexity_dist.index:
        print(f"  {complexity}: {complexity_dist[complexity]} ({complexity_pct[complexity]:.1f}%)")
    
    # Quality distribution
    quality_dist = selected_df['quality_rating'].value_counts().sort_index()
    quality_pct = selected_df['quality_rating'].value_counts(normalize=True).sort_index() * 100
    print(f"\nQuality distribution:")
    for quality in quality_dist.index:
        print(f"  {quality}: {quality_dist[quality]} ({quality_pct[quality]:.1f}%)")
    
    # Content type distribution
    content_dist = selected_df['content'].value_counts()
    content_pct = selected_df['content'].value_counts(normalize=True) * 100
    print(f"\nContent type distribution:")
    for content in content_dist.index:
        print(f"  {content}: {content_dist[content]} ({content_pct[content]:.1f}%)")
    
    # Theme distribution
    theme_dist = selected_df['theme'].value_counts()
    theme_pct = selected_df['theme'].value_counts(normalize=True) * 100
    print(f"\nTheme distribution:")
    for theme in theme_dist.index:
        print(f"  {theme}: {theme_dist[theme]} ({theme_pct[theme]:.1f}%)")
    
    # Language distribution
    language_dist = selected_df['language'].value_counts()
    language_pct = selected_df['language'].value_counts(normalize=True) * 100
    print(f"\nLanguage distribution:")
    for language in language_dist.index:
        print(f"  {language}: {language_dist[language]} ({language_pct[language]:.1f}%)")
    
    return selected_df

def save_selection(df, selected_indices, output_path):
    """Save selection results to a CSV file"""
    selected_df = df.loc[selected_indices]
    selected_df.to_csv(output_path, index=False)
    print(f"\nSelection results have been saved to: {output_path}")


def generate_csv_from_directory(csv_path, images_dir, output_path):
    """Generate the final test_data_selected.csv from the images in the directory and the original CSV"""
    # Load the original CSV
    df = pd.read_csv(csv_path)
    
    # Collect information about all images in the directory
    selected_images = []
    
    # Collect image information and calculate the actual number of samples per group
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith('.png'):
                # Extract information from the path, the path structure is now platform_complexity_quality_content/
                path_parts = Path(root).relative_to(images_dir).parts[0].split("_")
                platform, complexity, quality, content = path_parts
                    
                # Parse filekey and page_id from the file name
                filename = os.path.splitext(file)[0]
                if "--" in filename:
                    parts = filename.split("--", 1)
                    filekey = parts[0]
                    # Convert the safe page_id back to the original format
                    safe_page_id = parts[1]
                    page_id = safe_page_id.replace('_', ':').replace('-', ';')
                else:
                    # If the new format is not used, try to be compatible with the old format
                    print(f"  Warning: File name format is not as expected: {file}")
                    continue
                
                selected_images.append({
                    'platform': platform,
                    'complexity': complexity,
                    'quality_rating': quality,
                    'content': content,
                    'file_key': filekey,
                    'node_id': page_id
                })
    
    # Find the matching rows in the original CSV
    selected_indices = []
    for img_info in selected_images:
        matching_rows = df[
            (df['file_key'] == img_info['file_key']) &
            (df['node_id'] == img_info['node_id'])
        ]
        
        if not matching_rows.empty:
            selected_indices.extend(matching_rows.index.tolist())
        else:
            print(f"  Warning: No matching row found for: {img_info['file_key']}/{img_info['node_id']}")
    
    # Save the results
    selected_df = df.loc[selected_indices]
    selected_df.to_csv(output_path, index=False)
    print(f"\nGenerated CSV from the images in the directory: {output_path}")
    print(f"Number of selected samples: {len(selected_df)}")
    
    return selected_df

def sample_test_data(sample_num=160):
    # Set random seed to ensure reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # File paths
    input_csv = "output/page_filter/whitelist_categorized.csv"
    output_csv = "output/page_filter/test_data_selected.csv"
    sampling_stats_json = "output/page_filter/sampling_stats.json"
    
    # Directory paths
    all_images_dir = "./output/page_filter/selected_filter_by_layer"
    selected_images_dir = "./output/page_filter/selected_test_data_by_layer"
    
    print("Start organizing and filtering test data according to the hierarchical structure...")
    
    # Load and analyze data
    df = load_and_analyze_data(input_csv)
    
    # Organize all images according to the hierarchical structure
    print("\nOrganizing all images according to the hierarchical structure...")
    layer_stats = organize_by_layer_structure(df, all_images_dir)
    
    # Select test data
    print("\nPerforming stratified sampling...")
    selected_indices, sampling_stats = hierarchical_stratified_sampling(df, sample_num)
    
    # Save sampling statistics
    save_sampling_stats(sampling_stats, sampling_stats_json)
    
    # Copy the selected images to the target directory
    copy_selected_images(df, selected_indices, all_images_dir, selected_images_dir, sampling_stats)
    
    # Analyze selection results
    selected_df = analyze_selection(df, selected_indices)
    
    # Save results
    save_selection(df, selected_indices, output_csv)
    
    print("\nPlease hand it over to a professional designer to select the most suitable samples from each group and delete the rest")
    print("After completion, you can run the generate_final_csv() function to generate the final CSV file")

def generate_final_csv():
    """Generate the final CSV file"""
    input_csv = "./output/page_filter/whitelist_categorized.csv"
    output_csv = "./output/page_filter/test_data_selected_final.csv"
    images_dir = "./output/page_filter/selected_test_data_by_layer"
    
    print("Generating final CSV from images in the directory...")
    selected_df = generate_csv_from_directory(input_csv, images_dir, output_csv)
    
    # Analyze the final selection results
    if selected_df is not None and not selected_df.empty:
        analyze_selection(selected_df, selected_df.index)

if __name__ == "__main__":
    from ...configs.paths import enter_project_root
    enter_project_root()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sample", choices=["sample", "generate"])
    parser.add_argument("--sample_num", type=int, default=160)
    args = parser.parse_args()
    if args.mode == "sample":
        sample_test_data(args.sample_num)
    else:
        generate_final_csv()
