import os
import pickle
from PIL import Image


def main():
    data_dir = './celeb64'
    os.mkdir(data_dir)
    with open('celeb_data.pkl', 'rb') as f:
        images = pickle.load(f)

    for idx, img_array in enumerate(images):
        # Convert the NumPy array to a PIL Image
        img = Image.fromarray(img_array)
        # Save the image to disk
        img.save(os.path.join(data_dir, f'image_{idx}.png'))


if __name__ == '__main__':
    main()
