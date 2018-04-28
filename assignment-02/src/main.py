from PIL import Image, ImageOps
import numpy as np
import itertools

class DetailedImage(object):
    def __init__(self, raw_content):
        self.raw_content = raw_content
        self.rows, self.columns = raw_content.size

    def raw_image_copy(self):
        return self.raw_content.copy()

    def meta_copy(self):
        return self.raw_image_copy(), itertools.product(range(self.rows), range(self.columns))

def load_image(path):
    raw_image = Image.open(path)
    return DetailedImage(raw_content=raw_image)

def show_image(detailed_image):
    detailed_image.raw_content.show()

def rgb_to_y_band(image):
    image, image_range = image.meta_copy()
    pixels = image.load()

    for i, j in image_range:
        R, G, B = pixels[i, j]
        Y_B = int(16 + (65.481 * R)/256 + (122.233 * G)/256 + (24.966 * B)/256)
        pixels[i, j] = (Y_B, Y_B, Y_B)

    return DetailedImage(image)

def simulate(image):
    show_image(image)
    y_band_image = rgb_to_y_band(image)

    show_image(y_band_image)

def main():
    LENA_IMAGE_PATH = "../assets/images/lena.jpg"
    LENA_128_IMAGE_PATH = "../assets/images/lena_128.jpg"
    LENA_256_IMAGE_PATH = "../assets/images/lena_256.jpg"
    BABOON_256_IMAGE_PATH = "../assets/images/baboon_256.jpg"
    PEPERS_256_IMAGE_PATH = "../assets/images/pepers_256.jpg"
    TEST_256_IMAGE_PATH = "../assets/images/test_256.jpg"

    lena = load_image(LENA_IMAGE_PATH)
    simulate(image=lena)

if __name__ == '__main__':
    main()
