"""Visualize the redundancy of the dataset."""

import matplotlib.pyplot as plt
import numpy as np
from fastcan.narx import make_time_shift_features, make_time_shift_ids
from fastcan.utils import mask_missing_values
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # Plot raw time-sereis data
    u = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * u)
    
    fonts = 14
    plt.plot(u, y)
    plt.xlabel("Time (s)", fontsize=fonts-1)
    plt.ylabel("Amplitude", fontsize=fonts-1)
    # Set tick label font size
    plt.tick_params(axis='both', labelsize=fonts-1)
    plt.xlim(0, 1)
    plt.savefig("redundancy_raw_data.png", bbox_inches="tight")
    plt.close()

    # Plot the redundancy in column-wise (feature-wise)
    time_ids = make_time_shift_ids(1, 20, False)
    Y = make_time_shift_features(y.reshape(-1, 1), time_ids)
    Y, u = mask_missing_values(Y, u)  # Mask missing values caused by time shifts
    Y_corr = np.corrcoef(Y, rowvar=False)

    plt.imshow(Y_corr, extent=[1, Y_corr.shape[1], Y_corr.shape[0], 1])
    cb = plt.colorbar()
    cb.ax.set_title("$r$", fontsize=fonts)
    cb.ax.tick_params(labelsize=fonts)  # Colorbar tick label font size

    ticks = np.arange(1, len(time_ids), step=4)
    plt.xticks(ticks)
    plt.yticks(ticks)
    # Set tick label font size
    plt.tick_params(axis='both', labelsize=fonts)
    plt.xlabel("Feature index", fontsize=fonts)
    plt.ylabel("Feature index", fontsize=fonts)
    plt.savefig("redundancy_feature.png", bbox_inches="tight")
    plt.close()

    # Plot the redundancy in row-wise (sample-wise)
    pca = PCA(2).fit(Y)

    pcs = pca.transform(Y)

    plt.scatter(pcs[:, 0], pcs[:, 1], c=u)
    cb = plt.colorbar()
    cb.ax.set_title("Time (s)", fontsize=fonts)
    cb.ax.tick_params(labelsize=fonts)  # Colorbar tick label font size

    plt.xlabel("The first principle component", fontsize=fonts)
    plt.ylabel("The second principle component", fontsize=fonts)
    plt.tick_params(axis='both', labelsize=fonts)
    plt.savefig("redundancy_sample.png", bbox_inches="tight")
    plt.close()
