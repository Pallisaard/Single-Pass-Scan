from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

HEATMAP_FILENAMES = ["heatmaps-b32.txt", "heatmaps-b64.txt", "heatmaps-b128.txt",
             "heatmaps-b256.txt", "heatmaps-b512.txt", "heatmaps-b1024.txt"]
HEATMAP_FILENAMES = [f"heatmaps/{filename}" for filename in HEATMAP_FILENAMES]
Q_VALUES = [2, 4, 7, 8, 10, 13, 16, 20, 24, 30, 32, 40]
B_VALUES = [32, 64, 128, 256, 512, 1024]
N_VALUES = [i for i in range(10, 31, 1)]

PLOT_FILENAMES = ["graphs/results-256-30-int32.csv", "graphs/results-256-10-int32.csv", "graphs/results-1024-10-int32.csv"]
PLOT_COLUMNS = ["cudaMemcpy", "naiveMemcpy", "cpu", "AuxBlock", "SeqLookback", "ParLookback", "scanIncAdd"]
PLOT_LABELS = ["cudaMemcpy", "Naive Memcpy", "CPU Sequential Scan", "Aux Block", "Sequential Lookback", "Parallel Lookback", "Base GPU Scan"]


def get_plot_info() -> Tuple[List[str], List[str], List[str]]:
    """
    Returns a tuple of lists containing the filenames, column names and labels for the plot data.

    Returns:
        Tuple[List[str], List[str], List[str]]: A tuple of lists containing the filenames, column names and labels.
    """
    return PLOT_FILENAMES, PLOT_COLUMNS, PLOT_LABELS


def get_BQN_values(skip_every_second_n: bool = False) -> Tuple[List[int], List[int], List[int]]:
    """
    Returns a tuple of lists containing the B, Q and N values for which there is data.

    Args:
        skip_every_second_n (bool, optional): Whether to skip every second N value. Defaults to False.

    Returns:
        Tuple[List[int], List[int], List[int]]: A tuple of lists containing the B, Q and N values.
    """

    b_values = B_VALUES.copy()
    q_values = Q_VALUES.copy()
    n_values = N_VALUES.copy()
    if skip_every_second_n:
        n_values = [i for i in range(10, 31, 2)]

    return b_values, q_values, n_values


def get_heatmap_filenames() -> List[str]:
    """
    Returns a list of filenames for the heatmap data.

    Returns:
        List[str]: A list of filenames for the heatmap data.
    """
    return HEATMAP_FILENAMES


def replace_nan(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Replaces NaN values in a given column of a pandas DataFrame with the average of the previous and next non-NaN values.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to be processed.
        column (str): The name of the column to be processed.

    Returns:
        pd.DataFrame: The DataFrame with NaN values replaced.
    """
    for i in range(1, len(df[column]) - 1):
        if np.isnan(df[column][i]):
            df.loc[i, column] = (df[column][i - 1] + df[column][i + 1]) / 2
    return df


def load_experiment_data(filename: str) -> pd.DataFrame:
    """
    Load experiment data from a CSV file and return it as a pandas DataFrame.

    Args:
        filename (str): The path to the CSV file containing the experiment data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the experiment data.
    """
    df = pd.read_csv(filename, index_col=False)
    df = replace_nan(df, "scanIncAdd")
    return df


def print_to_latex(df: pd.DataFrame) -> None:
    """
    Prints a pandas DataFrame in LaTeX table format.

    Parameters:
        df (pd.DataFrame): The DataFrame to be printed.
    """
    latex_table = df.to_latex(index=False, float_format="%.2f")
    print(latex_table)


def plot_experiment_data(df: pd.DataFrame,
                         columns: List[str],
                         labels: List[str],
                         title: Optional[str] = None,
                         filename: Optional[str] = None):
    """
    Plots the experiment data contained in the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the experiment data.
        columns (List[str]): A list of column names to plot.
        labels (List[str]): A list of labels for each column to plot.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        filename (Optional[str], optional): The filename to save the plot to. Defaults to None.
    """
    plt.figure(figsize=(6, 4))
    plt.xlabel('Array size (2^n)')
    plt.ylabel('GB/s')
    plt.xticks(np.arange(10, 31, 1))
    plt.yticks(np.arange(0, 1501, 200))
    plt.grid(color="lightgray")
    for i in range(len(columns)):
        plt.plot(df["N"], df[columns[i]], label=labels[i])
    plt.legend()

    if title:
        plt.title(title)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def load_heatmap_data(filenames: List[str], skip_every_second_n: bool = False) -> np.ndarray:
    """
    Load heatmap data from a list of filenames.

    Args:
        filenames (List[str]): A list of filenames to load data from.
        skip_second_n (bool, optional): Whether to skip the second N value in each file. Defaults to False.

    Returns:
        np.ndarray: A array containing the loaded data. The array has shape (B, Q, N).
    """
    data = []
    for filename in filenames:
        with open(filename) as f:
            lines = f.readlines()
            data.append([])
            for line in lines:
                if line.startswith("Q"):
                    data[-1].append([])
                elif line.startswith("i"):
                    data[-1][-1].append(float(line.split(" - ")[1]))

    data = np.array(data)

    if skip_every_second_n:
        data = data[:, :, ::2]

    return data


def slice_data(data_3d: np.ndarray, d: Tuple[int, int], k: int) -> np.ndarray:
    """
    Slice a 3D numpy array along the given dimensions and return a 2D numpy array.

    Args:
        data_3d (np.ndarray): The 3D numpy array to slice.
        d (Tuple[int, int]): A tuple of two integers representing the dimensions to slice along.
        k (int): The index of the slice to extract.

    Returns:
        np.ndarray: A 2D numpy array representing the slice of the 3D array along the given dimensions and index.
    """
    d1, d2 = d
    if d == (0, 1) or d == (1, 0):
        data_2d = data_3d[:, :, k]
    elif d == (1, 2) or d == (2, 1):
        data_2d = data_3d[k, :, :]
    else:
        data_2d = data_3d[:, k, :]

    if d1 > d2:
        return data_2d
    return data_2d.T

def plot_heatmap_from_data(
    data: np.ndarray, 
    x_values: List[int], 
    y_values: List[int], 
    x_label: str, 
    y_label: str, 
    title: Optional[str] = None, 
    interpolation: str = "nearest", 
    colorbar: bool = False, 
    clim: Optional[Tuple[int, int]] = None, 
    filename: Optional[str] = None, 
    return_ax: bool = False, 
    ax: Optional[plt.Axes] = None, 
    figsize: Tuple[int, int] = (8, 6), 
    dpi: int = 300
):
    """
    Plots a heatmap from a 2D numpy array.

    Args:
        data (np.ndarray): The 2D numpy array to plot.
        x_values (List[int]): The values to use for the x-axis.
        y_values (List[int]): The values to use for the y-axis.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        interpolation (str, optional): The interpolation method to use. Defaults to "nearest".
        colorbar (bool, optional): Whether to show a colorbar. Defaults to False.
        clim (Optional[Tuple[int, int]], optional): The limits for the colorbar. Defaults to None.
        filename (Optional[str], optional): The filename to save the plot to. Defaults to None.
        return_ax (bool, optional): Whether to return the Axes object. Defaults to False.
        ax (Optional[plt.Axes], optional): The Axes object to use for the plot. Defaults to None.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (8, 6).
        dpi (int, optional): The DPI of the figure. Defaults to 300.

    Returns:
        Optional[plt.Axes]: The Axes object if `return_ax` is True, else None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if clim is None:
        clim = (0, data.max())

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels(x_values)
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels(y_values)
    ax.set_title(title)
    img = ax.imshow(data, cmap="winter", interpolation=interpolation)
    img.set_clim(clim)
    if colorbar:
        ax.figure.colorbar(img, ax=ax, label="GB/s")

    # Write the value of each cell in the heatmap
    for i in range(len(y_values)):
        for j in range(len(x_values)):
            if data[i, j] > clim[1] * 0.8:
                ax.text(j, i, data[i, j], ha="center", va="center", color="darkblue", fontsize=7)
            else:
                ax.text(j, i, data[i, j], ha="center", va="center", color="white", fontsize=7)

    if filename:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")

    if return_ax:
        return ax
    return None

def plot_heatmap(
    array_data: np.ndarray,
    dim_names: Tuple[int, int],
    B: Optional[int] = None,
    Q: Optional[int] = None,
    N: Optional[int] = None,
    bqn_values: Optional[Tuple[List[int], List[int], List[int]]] = None,
    *args,
    **kwargs
):
    """
    Plots a heatmap of a 2D slice of a 3D array, where the slice is defined by two dimensions
    specified by their names, and the third dimension is specified by either B, Q, or N.

    Args:
        array_data (np.ndarray): The 3D array to plot a slice of.
        dim_names (Tuple[int, int]): A tuple of two integers specifying the names of the two dimensions to plot.
        B (Optional[int]): The value of the B dimension to slice the array with. If specified, Q and N must be None.
        Q (Optional[int]): The value of the Q dimension to slice the array with. If specified, B and N must be None.
        N (Optional[int]): The value of the N dimension to slice the array with. If specified, B and Q must be None.
        bqn_values (Optional[Tuple[List[int], List[int], List[int]]]): A tuple of lists containing the B, Q and N values for which there is data. Defaults to None.
        *args, **kwargs: Additional arguments to pass to the `plot_heatmap_from_data` function.

    Returns:
        None
    """
    d_map = {"B": 0, "Q": 1, "N": 2}
    if bqn_values is None:
        bqn_values = get_BQN_values()
    data_B_values, data_Q_values, data_N_values = bqn_values
    d1_n, d2_n = dim_names
    d1, d2 = d_map[d1_n], d_map[d2_n]
    if B:
        k = data_B_values.index(B)
    elif Q:
        k = data_Q_values.index(Q)
    else:
        k = data_N_values.index(N)
    data_2d = slice_data(array_data, (d1, d2), k)

    plot_heatmap_from_data(data_2d, bqn_values[d1], bqn_values[d2], d1_n, d2_n, *args, **kwargs)