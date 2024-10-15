import numpy as np
import cv2
def skin_detection(image_path,threshold_percentage=3):
    # Convert image to ARGB color space
    image = cv2.imread(image_path)
    argb = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    height, width, _ = argb.shape

    # Initialize skin mask
    skin_mask = np.zeros((height, width), dtype=np.uint8)

    # Define thresholds for skin detection
    min_hue = 0
    max_hue = 50
    min_saturation = 0.23
    max_saturation = 0.68
    min_red = 95
    min_green = 40
    min_blue = 20
    min_rg_diff = 15
    min_rb_diff = 15
    min_alpha = 15

    for y in range(height):
        for x in range(width):
            # Get ARGB values for the current pixel
            alpha = argb[y, x, 3]
            red = argb[y, x, 2]
            green = argb[y, x, 1]
            blue = argb[y, x, 0]

            # Convert RGB to HSV
            hsv = cv2.cvtColor(np.uint8([[[blue, green, red]]]), cv2.COLOR_BGR2HSV)
            hue = hsv[0, 0, 0]  # Hue value
            saturation = hsv[0, 0, 1] / 255.0  # Saturation value

            # Check if pixel is within skin color range
            if (min_hue <= hue <= max_hue) and \
                    (min_saturation <= saturation <= max_saturation) and \
                    (red > min_red) and \
                    (green > min_green) and \
                    (blue > min_blue) and \
                    (red > green) and \
                    (red > blue) and \
                    (abs(red - green) > min_rg_diff) and \
                    (abs(red - blue) > min_rb_diff) and \
                    (alpha > min_alpha):
                skin_mask[y, x] = 255  # Skin pixel

    # Count the number of non-zero pixels (skin)
    skin_pixels = cv2.countNonZero(skin_mask)

    # Calculate the percentage of skin pixels in the image
    height, width = skin_mask.shape
    total_pixels = height * width
    skin_percentage = (skin_pixels / total_pixels) * 100

    # Determine if there is skin or not
    if skin_percentage >= threshold_percentage:
        return True
    else:
        return False