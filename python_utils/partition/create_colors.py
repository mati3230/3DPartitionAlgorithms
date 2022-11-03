import random
import numpy as np
import argparse


def main(n_colors=10000, filename="../colors.npz"):
    """Create some hex code colors and save them in rgb format.

    Parameters
    ----------
    n_colors : int
        Number of colors that should be created.
    filename : str
        Absolute or relative path of file where the colors will be saved.
        Usually the file should be saved in the root of the project where the
        environment is used.
    """
    col_mat = np.ndarray((n_colors, 3), np.int32)
    # create n_colors hex colors
    hex_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(n_colors)]
    # transform the hex code colors to rgb colors
    for i in range(n_colors):
        hex_color = hex_colors[i][1:]
        color = list(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        color = np.array(color, np.int32)
        col_mat[i, :] = color
    np.savez(filename, colors=col_mat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_colors",
        type=int,
        default=10000,
        help="Number of colors that should be created.")
    parser.add_argument(
        "--filename",
        type=str,
        default="../colors.npz",
        help="Absolute or relative path of file where the colors will be saved.")
    args = parser.parse_args()
    main(args.n_colors, args.filename)
