import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functions import get_time_interval, make_wdir

def create_hitmap(
    results_df,
    metric_name,
    output_dir="data/",
    figsize=(18, 12),
    cmap="coolwarm", 
    file_path="", 
    config={}
):

    required_cols = [
        'Blend Enter Start',
        'Blend Enter End',
        'Blend Exit Start',
        'Blend Exit End',
        metric_name
    ]

    for col in required_cols:
        if col not in results_df.columns:
            raise ValueError(f"Missing column: {col}")

    heatmap_data = results_df.pivot_table(
        index='Blend Enter Start',
        columns='Blend Exit Start',
        values=metric_name,
        aggfunc='mean'
    ).sort_index().sort_index(axis=1)

    if heatmap_data.empty or heatmap_data.isna().all().all():
        print(f"No valid data for heatmap: {metric_name}")
        return False

    enter_labels = {
        row['Blend Enter Start']: f"{row['Blend Enter Start']}-{row['Blend Enter End']}"
        for _, row in results_df[['Blend Enter Start', 'Blend Enter End']].drop_duplicates().iterrows()
    }

    exit_labels = {
        row['Blend Exit Start']: f"{row['Blend Exit Start']}-{row['Blend Exit End']}"
        for _, row in results_df[['Blend Exit Start', 'Blend Exit End']].drop_duplicates().iterrows()
    }

    heatmap_data.index = heatmap_data.index.map(enter_labels)
    heatmap_data.columns = heatmap_data.columns.map(exit_labels)

    valid_values = heatmap_data.stack().dropna()
    center = 0 if valid_values.min() < 0 else None

    plt.figure(figsize=figsize)
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=heatmap_data.size <= 100,
        fmt=".2f",
        center=center,
        cbar_kws={'label': metric_name}
    )

    plt.title(f"{metric_name} Heatmap\nBlend Enter vs Exit")
    plt.xlabel("Blend Exit Range")
    plt.ylabel("Blend Enter Range")
    plt.tight_layout()

    wdir = make_wdir(file_path, config)

    path_po = file_path.split("/")[1]
    file_path = f"{wdir}/{path_po}"
    safe_name = metric_name.replace('%', 'pct').replace('/', '_')
    path = (f"{file_path}_{safe_name}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"✓ Heatmap saved: {path}")
    return True