from PIL import Image
import numpy as np

class DetailedImage(object):
    def __init__(self, raw_content):
        self.raw_content = raw_content
        self.rows, self.columns = raw_content.size

    def meta_copy(self):
        return self.raw_content.copy(), self.rows, self.columns


def load_image(path):
    raw_image = Image.open(path)
    return DetailedImage(raw_content=raw_image)

def show_image(detailed_image):
    detailed_image.raw_content.show()


def negative(image):
    image, rows, columns = image.meta_copy()

    for i in range(rows):
        for j in range(columns):
            R, G, B = image.getpixel((i, j))
            image.putpixel((i, j), (255 - r, 255 - g, 255 - b))
    
    return DetailedImage(image)

def negative(image):
    image, rows, columns = image.meta_copy()

    for i in range(rows):
        for j in range(columns):
            r, g, b = image.getpixel((i, j))
            image.putpixel((i, j), (255 - r, 255 - g, 255 - b))
    
    return DetailedImage(image)

def rgb_to_yiq(image):
    image, rows, columns = image.meta_copy()

    for i in range(rows):
        for j in range(columns):
            R, G, B = image.getpixel((i, j))

            Y, I, Q = (
                int(0.299 * R + 0.587 * G + 0.114 * B),
                int(0.596 * R - 0.275 * G - 0.321 * B),
                int(0.212 * R - 0.523 * G + 0.311 * B)
            )

            image.putpixel((i, j), (Y, I, Q))

    return DetailedImage(image)
    
LENA_IMAGE_PATH = "../assets/images/lena.jpg"

def main():
    lena = load_image(LENA_IMAGE_PATH)
    show_image(lena)
    show_image(negative(lena))
    show_image(rgb_to_yiq(lena))

if __name__ == '__main__':
    main()
