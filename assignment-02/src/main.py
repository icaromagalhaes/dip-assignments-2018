from PIL import Image, ImageOps
import numpy as np
import itertools
import functools
import operator

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
        Y_B = int(R * 0.299 + G * 0.587 + B * 0.114)
        pixels[i, j] = (Y_B, Y_B, Y_B)

    return DetailedImage(image)


def histogram_expansion(image):
    def gxy(fxy, min, max):
        return int(((fxy - min) * 255)/(max-min))

    image, image_range = image.meta_copy()
    pixels = image.load()
    mimax, _, _ = image.getextrema()
    min, max = mimax[0], mimax[1]
    
    for i, j in image_range:
        fxy, _, _ = pixels[i, j]

        expanded_px = gxy(fxy, min, max)
        pixels[i, j] = (expanded_px, expanded_px, expanded_px)
       
    return DetailedImage(image)


def histogram_equalization(image):
    """
    Create a uniform distribution of grayscale values in
    the output image by applying a non linear mapping based on
    Pillow's histogram equalization algorithm [IM]
    """
    image, image_range = image.meta_copy()
    pixels = image.load()
    range_256 = range(256)
    original_histogram = image.histogram()
    lookup_table = []
    for band_idx in range(0, len(original_histogram), 256):
        histo = [_f for _f in original_histogram[band_idx:band_idx+256] if _f]
        if len(histo) <= 1:
            lookup_table.extend(list(range_256))
        else:
            step = (functools.reduce(operator.add, histo) - histo[-1]) // 255
            if not step:
                lookup_table.extend(list(range_256))
            else:
                n = step // 2
                for i in range_256:
                    lookup_table.append(n // step)
                    n = n + original_histogram[i+band_idx]
    return DetailedImage(image.point(lookup_table))


def simulate(image):
    show_image(image)
    
    y_band_image = rgb_to_y_band(image)
    show_image(y_band_image)

    expanded = histogram_expansion(y_band_image)
    show_image(expanded)

    equalized = histogram_equalization(y_band_image)
    show_image(equalized)

    equalized_and_expanded = histogram_expansion(equalized)
    expanded_and_equalized = histogram_equalization(expanded)

    show_image(equalized_and_expanded)
    show_image(expanded_and_equalized)


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
