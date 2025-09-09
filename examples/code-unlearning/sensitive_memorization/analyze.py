import argparse
import os
from datasets import load_from_disk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Times New Roman"


def main():
    models = ['codeparrot-small', 'codegen-350M-mono']
    model_names = ['CodeParrot-small', 'CodeGen-350M-Mono']
    MA_thresholds = [0.4557, 0.4879]
    colors = ['#CCDAED', '#E2F0D9']
    fig, axs = plt.subplots(1, 2, figsize=(24, 5), constrained_layout=True)
    
    for i in range(len(models)):
        print(models[i])
        ds_pii = load_from_disk(f"./codeparrot-clean-train-secrets-probed-{models[i]}")
        print(len(ds_pii))
        ds_pii_temp = ds_pii.filter(lambda example: example['secret_mean_MA'] > MA_thresholds[i], num_proc=16)
        print(len(ds_pii_temp))
        
        ax = axs[i]
        n, bins, patches = ax.hist(ds_pii['secret_mean_MA'], bins=40, color=colors[i], edgecolor='black', alpha=0.7, linewidth=2)
        ax.axvline(MA_thresholds[i], color='black', linestyle='dashed', linewidth=3)
        ax.text(MA_thresholds[i] - 0.475, ax.get_ylim()[1] * 0.875, f'Forgetting Threshold: {MA_thresholds[i]}', color='black', fontsize=27)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xlabel(model_names[i], fontsize=30, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=27)
        if i == 0:
            ax.set_ylabel('Frequency', fontsize=30, labelpad=15)
        ax.grid(True, linestyle='--', alpha=0.7)

    fig.savefig(r"MemorizationDistribution.jpg", dpi=300, bbox_inches='tight')
    fig.savefig(r"MemorizationDistribution.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
