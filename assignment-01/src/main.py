from PIL import Image
import numpy as np
import itertools

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
    pixels = image.load()

    for i, j in itertools.product(range(rows), range(columns)):
        R, G, B = pixels[i, j]
        pixels[i, j] = (255 - R, 255 - G, 255 - B)

    return DetailedImage(image)

def rgb_to_yiq(image):
    image, rows, columns = image.meta_copy()
    pixels = image.load()

    for i, j in itertools.product(range(rows), range(columns)):
        R, G, B = pixels[i, j]

        Y, I, Q = (
            int(0.299 * R + 0.587 * G + 0.114 * B),
            int(0.596 * R - 0.275 * G - 0.321 * B),
            int(0.212 * R - 0.523 * G + 0.311 * B)
        )

        pixels[i, j] = (Y, I, Q)

    return DetailedImage(image)

def yiq_to_rgb(image):
    image, rows, columns = image.meta_copy()
    pixels = image.load()

    for i, j in itertools.product(range(rows), range(columns)):
        Y, I, Q = pixels[i, j]

        R, G, B = (
            int(1.000 * Y + 0.956 * I + 0.621 * Q),
			int(1.000 * Y - 0.272 * I - 0.647 * Q),
			int(1.000 * Y - 1.106 * I + 1.703 * Q)
        )

        pixels[i, j] = (R, G, B)

    return DetailedImage(image)

def band_red(image, monocromatic=False):
    image, rows, columns = image.meta_copy()
    pixels = image.load()

    for i, j in itertools.product(range(rows), range(columns)):
        R, _, _ = pixels[i, j]
        pixels[i, j] = (R, R, R) if monocromatic else (R, 0, 0)

    return DetailedImage(image)

def band_green(image, monocromatic=False):
    image, rows, columns = image.meta_copy()
    pixels = image.load()

    for i, j in itertools.product(range(rows), range(columns)):
        _, G, _ = pixels[i, j]
        pixels[i, j] = (G, G, G) if monocromatic else (0, G, 0)

    return DetailedImage(image)

def band_blue(image, monocromatic=False):
    image, rows, columns = image.meta_copy()
    pixels = image.load()

    for i, j in itertools.product(range(rows), range(columns)):
        _, _, B = pixels[i, j]
        pixels[i, j] = (B, B, B) if monocromatic else (0, 0, B)

    return DetailedImage(image)

def limited_rgb(R, G, B):
    limit = lambda value: 255 if value > 255 else (
        0 if value < 0 else value
    )
    return limit(R), limit(G), limit(B)

def brightness_control_additive(image, C):
    image, rows, columns = image.meta_copy()
    pixels = image.load()

    for i, j in itertools.product(range(rows), range(columns)):
        R, G, B = pixels[i, j]
        R, G, B = limited_rgb(R + C, G + C, B + C)
        pixels[i, j] = (R, G, B)

    return DetailedImage(image)


def brightness_control_multiplicative(image, C):
    image, rows, columns = image.meta_copy()
    pixels = image.load()

    for i, j in itertools.product(range(rows), range(columns)):
        R, G, B = pixels[i, j]
        R, G, B = limited_rgb(R * C, G * C, B * C)
        pixels[i, j] = (R, G, B)

    return DetailedImage(image)

def band_y_threshold(image, M=None):
    image, rows, columns = rgb_to_yiq(image).meta_copy()
    pixels = image.load()

    # If the user has not specified the parameter,
    #   the mean Y should be calculated [IM]
    if M == None:
        M = np.mean(np.array(image)[:, :, 0].flatten())

    for i, j in itertools.product(range(rows), range(columns)):
        Y, I, Q = pixels[i, j]
        pixels[i, j] = ((255 if Y >= M else 0), 0, 0)

    return yiq_to_rgb(DetailedImage(image))

LENA_IMAGE_PATH = "../assets/images/lena.jpg"

def simulate(image):
    # Show image [IM]
    show_image(image)

    # Negative filter [IM]
    show_image(negative(image))

    # RGB to YIQ and YIQ to RGB [IM]
    yiq_image = rgb_to_yiq(image)
    show_image(yiq_image)

    rgb_image = yiq_to_rgb(yiq_image)
    show_image(rgb_image)

    # R, G and B bands for colored and monocromatic images [IM]
    show_image(band_red(image))
    show_image(band_red(image, monocromatic=True))

    show_image(band_green(image))
    show_image(band_green(image, monocromatic=True))

    show_image(band_blue(image))
    show_image(band_blue(image, monocromatic=True))

    # Additive and multiplicative brightness control [IM]
    show_image(brightness_control_additive(image, 100))
    show_image(brightness_control_multiplicative(image, 2))

    # Threshold over Y [IM]
    show_image(band_y_threshold(image))
    show_image(band_y_threshold(image, 100))

def main():
    # Load and show image [IM]
    lena = load_image(LENA_IMAGE_PATH)

    simulate(image=lena)

if __name__ == '__main__':
    main()
