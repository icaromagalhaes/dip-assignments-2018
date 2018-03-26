from PIL import Image
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

    for i, j in itertools.product(range(rows), range(columns)):
        R, G, B = image.getpixel((i, j))
        image.putpixel((i, j), (255 - r, 255 - g, 255 - b))
    
    return DetailedImage(image)

def negative(image):
    image, rows, columns = image.meta_copy()

    for i, j in itertools.product(range(rows), range(columns)):
        r, g, b = image.getpixel((i, j))
        image.putpixel((i, j), (255 - r, 255 - g, 255 - b))
    
    return DetailedImage(image)

def rgb_to_yiq(image):
    image, rows, columns = image.meta_copy()

    for i, j in itertools.product(range(rows), range(columns)):
        R, G, B = image.getpixel((i, j))

        Y, I, Q = (
            int(0.299 * R + 0.587 * G + 0.114 * B),
            int(0.596 * R - 0.275 * G - 0.321 * B),
            int(0.212 * R - 0.523 * G + 0.311 * B)
        )

        image.putpixel((i, j), (Y, I, Q))

    return DetailedImage(image)

def yiq_to_rgb(image):
    image, rows, columns = image.meta_copy()

    for i, j in itertools.product(range(rows), range(columns)):
        Y, I, Q = image.getpixel((i, j))

        R, G, B = (
            int((1.000 * Y + 0.956 * I + 0.621 * Q)),
			int((1.000 * Y - 0.272 * I - 0.647 * Q)),
			int((1.000 * Y - 1.106 * I + 1.703 * Q))
        )

        image.putpixel((i, j), (R, G, B))

    return DetailedImage(image)

def band_red(image, monocromatic=False):
    image, rows, columns = image.meta_copy()

    for i, j in itertools.product(range(rows), range(columns)):
        R, _, _ = image.getpixel((i, j))
        image.putpixel(
            (i, j),
            (R, R, R) if monocromatic else (R, 0, 0)
        )
            
    return DetailedImage(image)

def band_green(image, monocromatic=False):
    image, rows, columns = image.meta_copy()
    
    for i, j in itertools.product(range(rows), range(columns)):
        _, G, _ = image.getpixel((i, j))
        image.putpixel(
            (i, j),
            (G, G, G) if monocromatic else (0, G, 0)
        )
    return DetailedImage(image)

def band_blue(image, monocromatic=False):
    image, rows, columns = image.meta_copy()
    
    for i, j in itertools.product(range(rows), range(columns)):
        _, _, B = image.getpixel((i, j))
        image.putpixel(
            (i, j),
            (B, B, B) if monocromatic else (0, 0, B)
        )
    return DetailedImage(image)

def limited_rgb(R, G, B):
    limit = lambda value: 255 if value > 255 else (
        0 if value < 0 else value
    )
    return limit(R), limit(G), limit(B)

def brightness_control_additive(image, C):
    image, rows, columns = image.meta_copy()

    for i, j in itertools.product(range(rows), range(columns)):
        R, G, B = image.getpixel((i, j))
        R, G, B = limited_rgb(R + C, G + C, B + C)
        image.putpixel((i, j), (R, G, B))
    
    return DetailedImage(image)


def brightness_control_multiplicative(image, C):
    image, rows, columns = image.meta_copy()

    for i, j in itertools.product(range(rows), range(columns)):
        R, G, B = image.getpixel((i, j))
        R, G, B = limited_rgb(R * C, G * C, B * C)
        image.putpixel((i, j), (R, G, B))
    
    return DetailedImage(image)

def band_y_threshold(image, M=None):
    image, rows, columns = rgb_to_yiq(image).meta_copy()
    
    # If the user has not specified the parameter,
    #   the mean should be calculated [IM]
    if M == None:
        yellow_band_summation = 0
        for i, j in itertools.product(range(rows), range(columns)):
            Y, _, _ = image.getpixel((i, j))
            yellow_band_summation += Y
        
        # The final mean is calculated [IM]
        M = yellow_band_summation / (rows * columns)

    for i, j in itertools.product(range(rows), range(columns)):
        Y, I, Q = image.getpixel((i, j))
        image.putpixel(
            (i, j),
            # Taking the limiar decision [IM]
            ((255 if Y >= M else 0), 0, 0)
        )
    
    return yiq_to_rgb(DetailedImage(image))


LENA_IMAGE_PATH = "../assets/images/lena.jpg"

def main():
    # Load and show image [IM]
    lena = load_image(LENA_IMAGE_PATH)
    # show_image(lena)

    # Negative filter [IM]
    show_image(negative(lena))
    
    # RGB to YIQ and YIQ to RGB [IM]
    yiq_lena = rgb_to_yiq(lena)
    show_image(yiq_lena)

    rgb_lena = yiq_to_rgb(yiq_lena)
    show_image(rgb_lena)

    # R, G and B bands for colored and monocromatic images [IM]
    show_image(band_red(lena))
    show_image(band_red(lena, monocromatic=True))

    show_image(band_green(lena))
    show_image(band_green(lena, monocromatic=True))

    show_image(band_blue(lena))
    show_image(band_blue(lena, monocromatic=True))
    
    # Additive and multiplicative brightness control [IM]
    show_image(brightness_control_additive(lena, 100))
    show_image(brightness_control_multiplicative(lena, 2))

    # Threshold over Y [IM]
    show_image(band_y_threshold(lena))
    show_image(band_y_threshold(lena, 100))

if __name__ == '__main__':
    main()
