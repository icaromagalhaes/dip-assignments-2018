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


def negative(image):
    image, image_range = image.meta_copy()
    pixels = image.load()

    for i, j in image_range:
        R, G, B = pixels[i, j]
        pixels[i, j] = (255 - R, 255 - G, 255 - B)

    return DetailedImage(image)


def negative_luminance(image):
    image, image_range = image.meta_copy()
    pixels = image.load()

    for i, j in image_range:
        Y, I, Q = pixels[i, j]
        pixels[i, j] = (255 - Y, 0, 0)

    return DetailedImage(image)


def rgb_to_yiq(image):
    image, image_range = image.meta_copy()
    pixels = image.load()

    for i, j in image_range:
        R, G, B = pixels[i, j]

        Y, I, Q = (
            int(0.299 * R + 0.587 * G + 0.114 * B),
            int(0.596 * R - 0.275 * G - 0.321 * B),
            int(0.212 * R - 0.523 * G + 0.311 * B)
        )

        pixels[i, j] = (Y, I, Q)

    return DetailedImage(image)


def yiq_to_rgb(image):
    image, image_range = image.meta_copy()
    pixels = image.load()

    for i, j in image_range:
        Y, I, Q = pixels[i, j]

        R, G, B = (
            int(1.000 * Y + 0.956 * I + 0.621 * Q),
			int(1.000 * Y - 0.272 * I - 0.647 * Q),
			int(1.000 * Y - 1.106 * I + 1.703 * Q)
        )

        pixels[i, j] = (R, G, B)

    return DetailedImage(image)


def red_band(image, monocromatic=False):
    image, image_range = image.meta_copy()
    pixels = image.load()

    for i, j in image_range:
        R, _, _ = pixels[i, j]
        pixels[i, j] = (R, R, R) if monocromatic else (R, 0, 0)

    return DetailedImage(image)


def green_band(image, monocromatic=False):
    image, image_range = image.meta_copy()
    pixels = image.load()

    for i, j in image_range:
        _, G, _ = pixels[i, j]
        pixels[i, j] = (G, G, G) if monocromatic else (0, G, 0)

    return DetailedImage(image)


def blue_band(image, monocromatic=False):
    image, image_range = image.meta_copy()
    pixels = image.load()

    for i, j in image_range:
        _, _, B = pixels[i, j]
        pixels[i, j] = (B, B, B) if monocromatic else (0, 0, B)

    return DetailedImage(image)


def luminance_band(image, monocromatic=False): # Y_band
    image, image_range = rgb_to_yiq(image).meta_copy()
    pixels = image.load()

    for i, j in image_range:
        Y, _, _ = pixels[i, j]
        pixels[i, j] = (Y, Y, Y) if monocromatic else (Y, 0, 0)

    return DetailedImage(image)


def limited_rgb(R, G, B):
    limit = lambda value: 255 if value > 255 else (
        0 if value < 0 else value
    )
    return limit(R), limit(G), limit(B)


def brightness_control_additive(image, C):
    image, image_range = image.meta_copy()
    pixels = image.load()

    for i, j in image_range:
        R, G, B = pixels[i, j]
        R, G, B = limited_rgb(R + C, G + C, B + C)
        pixels[i, j] = (R, G, B)

    return DetailedImage(image)


def brightness_control_multiplicative(image, C):
    image, image_range = image.meta_copy()
    pixels = image.load()

    for i, j in image_range:
        R, G, B = pixels[i, j]
        R, G, B = limited_rgb(R * C, G * C, B * C)
        pixels[i, j] = (R, G, B)

    return DetailedImage(image)


def y_band_threshold(image, M=None):
    image, image_range = rgb_to_yiq(image).meta_copy()
    pixels = image.load()

    # If the user has not specified the parameter,
    #   the mean Y should be calculated [IM]
    if M == None:
        M = np.mean(np.array(image)[:, :, 0].flatten())

    for i, j in image_range:
        Y, I, Q = pixels[i, j]
        pixels[i, j] = ((255 if Y >= M else 0), 0, 0)

    return yiq_to_rgb(DetailedImage(image))


def mean_filter(image, kernel_size=3):
    def integer_mask_mean(component):
        return np.mean(component).astype(int)

    image, image_range = image.meta_copy()
    pixels = image.load()
    npimage = np.array(image)

    for i, j in image_range:
        mask = npimage[i: i + kernel_size, j: j + kernel_size]
        pixels[j, i] = (
            integer_mask_mean(mask[:, :, 0]), # R
            integer_mask_mean(mask[:, :, 1]), # G
            integer_mask_mean(mask[:, :, 2])  # B
        )

    return DetailedImage(image)


def median_filter(image, kernel_size=3):
    def integer_mask_median(component):
        return np.median(component).astype(int)

    image, image_range = image.meta_copy()
    pixels = image.load()
    npimage = np.array(image)

    for i, j in image_range:
        mask = npimage[i: i + kernel_size, j: j + kernel_size]
        pixels[j, i] = (
            integer_mask_median(mask[:, :, 0].flatten()), # R
            integer_mask_median(mask[:, :, 1].flatten()), # G
            integer_mask_median(mask[:, :, 2].flatten())  # B
        )

    return DetailedImage(image)


def masked(image, mask):
    rows, columns = image.rows, image.columns
    image, _ = image.meta_copy()
    pixels = image.load()
    npimage = np.array(image)
    mask_dimm = mask[0].size
    border_limit = mask_dimm - 1

    row_range = range(border_limit, rows - border_limit)
    col_range = range(border_limit, columns - border_limit)
    image_range = itertools.product(row_range, col_range)

    for i, j in image_range:
        newMask = npimage[i: i + mask_dimm, j: j + mask_dimm]

        masked_R = int(np.sum(mask * newMask[:, :, 0]))
        masked_G = int(np.sum(mask * newMask[:, :, 1]))
        masked_B = int(np.sum(mask * newMask[:, :, 2]))

        pixels[j, i] = (masked_R, masked_G, masked_B)

    image = ImageOps.crop(image, border=mask_dimm)
    return DetailedImage(image)


def simulate(image):
    # Show image [IM]
    show_image(image)

    # RGB to YIQ and YIQ to RGB [IM]
    yiq_image = rgb_to_yiq(image)
    show_image(yiq_image)

    rgb_image = yiq_to_rgb(yiq_image)
    show_image(rgb_image)

    # R, G and B bands for colored and monocromatic images [IM]
    show_image(red_band(image))
    show_image(red_band(image, monocromatic=True))

    show_image(green_band(image))
    show_image(green_band(image, monocromatic=True))

    show_image(blue_band(image))
    show_image(blue_band(image, monocromatic=True))

    y_only_band = luminance_band(image)
    y_only_band_mono = luminance_band(image, monocromatic=True)
    show_image(y_only_band)
    show_image(luminance_band(image, monocromatic=True))

    back_to_rgb_image = yiq_to_rgb(y_only_band)
    back_to_rgb_image_mono = yiq_to_rgb(y_only_band_mono)
    show_image(back_to_rgb_image)
    show_image(back_to_rgb_image_mono)

    # Negative [IM]
    show_image(negative(image))
    negative_luminance_img = negative_luminance(yiq_image)
    show_image(negative_luminance_img)

    negative_luminance_rgb_img = yiq_to_rgb(negative_luminance_img)
    show_image(negative_luminance_rgb_img)

    # Additive and multiplicative brightness control [IM]
    show_image(brightness_control_additive(image, 100))
    show_image(brightness_control_multiplicative(image, 2))

    # Threshold over Y [IM]
    show_image(y_band_threshold(image))
    show_image(y_band_threshold(image, 100))

    # Mean and Median filters - slow operations for large images [IM]
    show_image(mean_filter(image, kernel_size=3)) # just slow [IM]
    show_image(median_filter(image, kernel_size=3)) # little bit slower than mean [IM]

    # Laplacian filter [IM]
    laplacian_mask = np.array(
        [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]]
    )
    show_image(masked(image, laplacian_mask))

    # Sobel filter [IM]
    sobel_over_x_mask = np.array(
        [[-1,  0,  1],
         [-2,  0,  2],
         [-1,  0,  1]]
    )
    show_image(masked(image, sobel_over_x_mask))

    sobel_over_y_mask = np.array(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
    )
    show_image(masked(image, sobel_over_y_mask))

    # Suggested filters [IM]
    custom_filter_1 = np.array(
        [[ 0, -1,  0],
         [-1,  5, -1],
         [ 0, -1,  0]]
    )
    show_image(masked(image, custom_filter_1))

    custom_filter_2 = np.array(
        [[0,  0,  0],
         [0,  1,  0],
         [0,  0, -1]]
    )
    show_image(masked(image, custom_filter_2))


def main():
    LENA_IMAGE_PATH = "../assets/images/lena.jpg"
    LENA_128_IMAGE_PATH = "../assets/images/lena_128.jpg"
    LENA_256_IMAGE_PATH = "../assets/images/lena_256.jpg"
    BABOON_256_IMAGE_PATH = "../assets/images/baboon_256.jpg"
    PEPERS_256_IMAGE_PATH = "../assets/images/pepers_256.jpg"
    TEST_256_IMAGE_PATH = "../assets/images/test_256.jpg"

    # Load image and simulate [IM]
    lena = load_image(PEPERS_256_IMAGE_PATH)
    simulate(image=lena)

if __name__ == '__main__':
    main()
