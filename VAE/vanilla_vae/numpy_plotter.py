import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import math


def plot_a_numpy_array(a_numpy_array):
    if len(a_numpy_array.shape) != 2:
        side_size = int(math.sqrt(a_numpy_array.shape[0]))
        print(a_numpy_array.reshape(side_size, side_size).shape)
        plt.imshow(a_numpy_array.reshape(side_size, side_size))
        plt.show()
    else:
        plt.imshow(a_numpy_array)
        plt.show()


def plot_a_numpy_file(file_name, to_save=False):
    samples = np.loadtxt(file_name, delimiter=',')

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    if to_save:
        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/' + file_name.replace('original_np_outs/', '').replace('.out', '') + '.png')
        plt.close(fig)
    else:
        plt.show()


# plot_a_numpy_file('original_np_outs/original_samples_99000.out')