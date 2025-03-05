from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from load_dataset import load_lattice_dataloaders


def generate_statistics(data_dir, dimensions, distributions):
    statistics = []
    for dim in dimensions:
        for distribution in distributions:
            train_loader, val_loader, test_loader = load_lattice_dataloaders(
                data_dir=data_dir,
                dimension=dim,
                distribution_type=distribution,
                batch_size=1,
                shuffle=True,
            )
            for dataloader in [train_loader, val_loader, test_loader]:
                for batch in tqdm(dataloader, desc=f"dim={dim}, dist={distribution}"):
                    statistics.append({
                        "dimension": dim,
                        "distribution_type": distribution,
                        "shortest_vector_length": batch["shortest_vector_length"].item(),
                        "shortest_vector_length_gh": batch["shortest_vector_length_gh"].item(),
                        "shortest_lll_basis_vector_length": batch["shortest_lll_basis_vector_length"].item(),
                        "original_log_defect": batch["original_log_defect"].item(),
                        "lll_log_defect": batch["lll_log_defect"].item(),
                    })
    return statistics


def get_statistics(data_dir, dimensions, distributions, stats_filename="statistics.csv"):
    stats_file = Path(stats_filename)

    if stats_file.exists():
        print(f"Loading statistics from {stats_file}")
        df = pd.read_csv(stats_file)
    else:
        print("Generating statistics...")
        statistics = generate_statistics(data_dir, dimensions, distributions)
        df = pd.DataFrame(statistics)
        df.to_csv(stats_file, index=False)
        print(f"Statistics saved to {stats_file}")

    return df


def plot_log_defects(df, save_path=None):
    """
    Plots original log defects vs. LLL reduced log defects on a single plot,
    overlaying points with different colours and markers for each combination
    of lattice dimension and distribution type.

    Parameters:
    - df (pd.DataFrame): The dataframe must contain 'original_log_defect', 'lll_log_defect',
      'dimension', and 'distribution_type'.
    - save_path (str, optional): If provided, the figure is saved to this path.
    """
    # Ensure that 'dimension' and 'distribution_type' are treated as categorical data.
    df['dimension'] = df['dimension'].astype(str)
    df['distribution_type'] = df['distribution_type'].astype(str)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use both hue and style to differentiate groups.
    sns.scatterplot(
        data=df,
        x="original_log_defect",
        y="lll_log_defect",
        hue="dimension",          # Different colours for different dimensions.
        style="distribution_type",  # Different markers for distribution types.
        palette="Set2",
        s=100
    )

    x_diagonal = np.linspace(0, 50, 100)  # Adjust range as needed
    y_diagonal = x_diagonal
    plt.plot(x_diagonal, y_diagonal, 'r--', alpha=0.5, label="No Improvement")

    max_val = max(np.max(df["original_log_defect"]),
                  np.max(df["lll_log_defect"]))
    # Add some padding to the maximum value
    max_val = min(max_val * 1.1, 50)  # Limit to 50 to avoid extreme scales
    min_val = 0  # Defects are typically positive

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.title("Original vs. LLL Reduced Log Defects")
    plt.xlabel("Original Log Defect")
    plt.ylabel("LLL Log Defect")
    plt.legend(title="Dimension / Distribution", loc="best")

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_vector_lengths(df, save_path=None):
    """
    Creates a combined scatter plot for two comparisons on the same axes:

    1. Shortest Vector Length vs. Shortest Vector Length GH (Gaussian Heuristic)
       -- represented with circle markers ("o").
    2. Shortest Vector Length vs. Shortest LLL Basis Vector Length
       -- represented with square markers ("s").

    Data points are colored based on the combination of 'dimension' and 'distribution_type'.

    Parameters:
    - df (pd.DataFrame): Must contain the columns:
      'shortest_vector_length', 'shortest_vector_length_gh', 'shortest_lll_basis_vector_length',
      'dimension', and 'distribution_type'.
    - save_path (str, optional): If provided, the figure is saved to this path.
    """
    # Ensure categorical treatment of grouping variables.
    df['dimension'] = df['dimension'].astype(str)
    df['distribution_type'] = df['distribution_type'].astype(str)

    # Group data by the combination of dimension and distribution_type.
    groups = df.groupby(['dimension', 'distribution_type'])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a color palette for each unique group.
    group_keys = list(groups.groups.keys())
    palette = sns.color_palette("Set2", n_colors=len(group_keys))
    color_dict = {group: palette[i] for i, group in enumerate(group_keys)}

    # Plot the data for each group.
    for group, group_df in groups:
        color = color_dict[group]
        group_label = f"{group[0]}_{group[1]}"

        # Plot for Gaussian Heuristic: circle markers ("o")
        ax.scatter(
            group_df["shortest_vector_length"],
            group_df["shortest_vector_length_gh"],
            color=color,
            marker='o',
            s=100,
            edgecolor='k',
            label=f"{group_label} GH"
        )

        # Plot for LLL Basis: square markers ("s")
        ax.scatter(
            group_df["shortest_vector_length"],
            group_df["shortest_lll_basis_vector_length"],
            color=color,
            marker='s',
            s=100,
            edgecolor='k',
            label=f"{group_label} LLL"
        )

    # To avoid duplicate legend entries, collect unique handles.
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(),
              title="Group (Dim_Dist & Measurement)", loc="best")

    # Plot a diagonal (equality) line for reference.
    x_diagonal = np.linspace(0, 100, 100)  # Adjust the range if necessary.
    ax.plot(x_diagonal, x_diagonal, 'r--', alpha=0.5, label="Equality Line")

    # Determine axis limits based on the maximum values across both measurements.
    max_val = max(np.max(df["shortest_vector_length"]),
                  np.max(df["shortest_vector_length_gh"]),
                  np.max(df["shortest_lll_basis_vector_length"]))
    max_val = min(max_val * 1.1, 100)  # Add padding and cap at 100.
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.set_title("Shortest Vector Length vs. Gaussian Heuristic (GH) or LLL")
    ax.set_xlabel("Shortest Vector Length")
    ax.set_ylabel("GH or LLL")

    if save_path:
        plt.savefig(save_path)

    plt.show()


def main():
    data_dir = Path("random_bases")
    dimensions = [4, 6, 8, 12, 16]
    distributions = ["uniform"]

    df = get_statistics(data_dir, dimensions, distributions)

    plot_log_defects(df, "log_defects_overlayed.png")
    plot_vector_lengths(df, "lengths_overlayed.png")


if __name__ == "__main__":
    main()
