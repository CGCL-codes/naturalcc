"""Here we detect PII: Emails, IP addresses, and keys (SSH/API) and redact/anonymize them
    * we use one regex for emails and one for IP addresses
    * for keys we use detect-secrets tool, which is a combination of multiple plgins (regexes, entropy..)
    * we also add some filters on top of each tool to decrease the number of false positives
This script is adapted from https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/02_pii/pii_processor.py
"""

import argparse
import json
import logging

from datasets.utils.logging import set_verbosity_info
from datasets import load_dataset

from tqdm import tqdm
from utils.emails_ip_addresses_detection import detect_email_addresses
from utils.keys_detection import detect_keys


def postprocess_secrets(secrets):
    """Postprocess the secrets found by the scan_secrets function"""
    if secrets:
        matches = json.dumps(secrets)
        has_secrets = True
    else:
        matches = json.dumps([])
        has_secrets = False
    return matches, has_secrets


def scan_pii_batch(examples, key_detector="other"):
    """Scan a batch of examples from a dataset to detect PII
    This add two columns to the dataset:
    - secrets: (list) of secrets/PII found
    - has_secrets: (bool) whether the example contains secrets/PII
    """
    list_secrets = []
    list_has_secrets = []
    number_secrets = []
    for text in tqdm(examples["content"]):
        secrets = []
        if key_detector == "regex":
            # use a regex to detect keys + emails + ips
            secrets = secrets + detect_email_addresses(
                text, tag_types={"KEY", "EMAIL", "IP_ADDRESS"}
            )
        else:
            # detect emails and ip addresses with regexes
            secrets = secrets + detect_email_addresses(
                text, tag_types={"EMAIL", "IP_ADDRESS"}
            )
            # for keys use detect-secrets tool
            secrets = secrets + detect_keys(text)
        # to add this as new columns to datasets we need the same number of samples in each row
        # we save secrets as json strings instead of lists
        matches, has_secrets = postprocess_secrets(secrets)
        list_secrets.append(matches)
        list_has_secrets.append(has_secrets)
        number_secrets.append(len(secrets))
    return {
        "secrets": list_secrets,
        "has_secrets": list_has_secrets,
        "number_secrets": number_secrets,
    }


def parseArgs():
    parser = argparse.ArgumentParser(description="PII detection and redaction")
    parser.add_argument(
        "--dataset_name",
        default="bigcode/pii-for-code",
        type=str,
        help="HF repo name/path of the dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        type=str,
        help="Dataset split to process",
    )
    parser.add_argument(
        "--data_dir",
        default="data/python",
        type=str,
        help="Huggingface dataset dir to access",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="Batch size for the PII detection/redaction",
    )
    parser.add_argument(
        "--num_proc",
        default=96,
        type=int,
        help="Number of processes to use for the PII detection/redaction",
    )
    parser.add_argument(
        "--save_path_disk",
        default="bigcode-pii-pjj-local",
        type=str,
        help="Path to save the dataset on disk.",
    )
    return parser.parse_args()


def main():
    set_verbosity_info()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
        logging.FileHandler("pii.log"),
        logging.StreamHandler()
    ]
    )
    args = parseArgs()
    logger.info(f"** The job is running with the following arguments: **\n{args}\n **** ")

    logger.info(f" ===== Loading {args.dataset_name} =====")
    ds = load_dataset(args.dataset_name, split=args.split, data_dir=args.data_dir, num_proc=args.num_proc)

    # add id column to dataset
    logger.info(f" ===== Adding an index column =====")
    ds = ds.add_column("index", list(range(len(ds))))

    logger.info(f" ===== Applying PII detection =====")
    ds_pii = ds.map(
        scan_pii_batch, batched=True, batch_size=args.batch_size, num_proc=args.num_proc, load_from_cache_file=False
    )
    ds_pii = ds_pii.filter(lambda example: example['has_secrets'], batched=True, batch_size=args.batch_size, num_proc=args.num_proc)
    logger.info(f"Dataset info after PII detection:\n{ds_pii}")
    logger.info(f"Number of samples that contained PII: {sum(ds_pii['has_secrets'])}")
    logger.info(f"Total number of secrets found: {sum(ds_pii['number_secrets'])}")
    
    # save the final dataset
    logger.info(f" ===== Saving the dataset to disk =====")
    ds_pii.save_to_disk(args.save_path_disk)

    logger.info(f" ===== Dataset saved successfully =====")

if __name__ == "__main__":
    main()
