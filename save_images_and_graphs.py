import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image


def save_images(log_dir):
    npz_file = glob.glob(f"{log_dir}/../sampling/*.npz")[0]

    data = np.load(npz_file)
    images = data['arr_0']

    images_dir = os.path.join(log_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    for idx, img_array in enumerate(images):
        img = Image.fromarray(img_array)
        img.save(os.path.join(images_dir, f'image_{idx}.png'))


def save_plots(log_dir):
    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(log_dir, '../training/progress.csv'))

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Loss plot
    axs[0].plot(df['step'], df['loss'], label='Loss')
    axs[0].set_title('Loss vs. Step')
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    axs[0].legend()

    # MSE plot
    axs[1].plot(df['step'], df['mse'], label='MSE', color='r')
    axs[1].set_title('MSE vs. Step')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('MSE')
    axs[1].grid(True)
    axs[1].legend()

    # Training time plot
    axs[2].plot(df['step'], df['training time'], label='Training Time', color='g')
    axs[2].set_title('Training Time vs. Step')
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('Training Time (s)')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(plots_dir, 'training_plots.png'))
    plt.close()  # Close the figure to free memory


def main(args):
    log_dir = os.path.join('./logs', args.exp, 'results')
    save_images(log_dir)
    save_plots(log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp")
    args = parser.parse_args()

    main(args)
