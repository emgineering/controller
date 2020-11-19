import cv2
import numpy as np
import random
from plate_locator import *

# recommended val: 10-25 (higher = lower image qual)
def blur(img, val):
    if val > 0:
        out = cv2.GaussianBlur(img, None, val)
        return out
    return img

# recommended val: 0 - 170 (higher is lower qual)
def dim(img, val):
    out = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out[:,:,2] = (out[:,:,2].astype('float32') * (255-val) / 255)
    out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR).astype('uint8')
    return out

# Adds copies of an edge to the periphery of an image
def pad_axis(image, axis, value):
    npad = [(0, 0)] * image.ndim
    npad[axis] = value
    return np.pad(image, npad, mode='edge')
    
# Translates an image horizontally, filling empty space with copies of the outer edge.
def h_shift(img, val):
    _, width, _ = img.shape
    if val > 0:
        out = pad_axis(img[:, val:], 1, (0, val))
    else:
        out = pad_axis(img[:, :width - abs(val)], 1, (abs(val), 0))
    return out

# Translates an image vertically, filling empty space with copies of the outer edge
def v_shift(img, val):
    height, _, _ = img.shape
    if val > 0:
        out = pad_axis(img[val:], 0, (0, val))
    else:
        out = pad_axis(img[:height - abs(val)], 0, (abs(val), 0))
    return out

# Given a source image of a plate, generates an approximation of how it would look
# when captured from gazebo.
def apply_perspective(plate, intensity, direction):
    scale = 800/1800.0
    squashed = cv2.resize(plate, None, fx=1, fy=scale)

    height, width = squashed.shape[:2]
    
    pillar = np.zeros((height, width//3, 3), np.uint8)
    pillar[:,:] = (255, 0, 0)

    pillarboxed_plate = np.hstack([pillar, squashed, pillar])
    full_width = pillarboxed_plate.shape[1]

    bg_height = height + intensity * 2

    background = np.zeros((bg_height, full_width, 3), np.uint8)
    # make bg green
    background[:,:] = (0, 255,0)
    # paste pillarboxed plate in the middle
    background[(bg_height - height) // 2 : (bg_height + height)//2] = pillarboxed_plate

    # calculate corners of skewed box
    if direction == 1:
        height1 = 0
        height2 = intensity
    else:
        height2 = 0
        height1 = intensity

    pt_A = [0, height1]
    pt_B = [0, bg_height - 1 - height1]
    pt_C = [full_width - 1, bg_height - 1 - height2]
    pt_D = [full_width - 1, height2]

    dst_size = 200
    src_points = np.float32([pt_A, pt_B, pt_C, pt_D])
    dst_points = np.float32(
        [[0, 0],
        [0, dst_size - 1],
        [dst_size - 1, dst_size - 1],
        [dst_size - 1, 0]]
    )

    # apply transform
    transform = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(background, transform, (dst_size, dst_size))

    return warped


# Applies a set of predefined perspective abberations, blurs, and dimming to a source image.
# Returns 'None' in the event that the perspective distortion could not be reversed.
def distort_image(image):
    # Initialize variables
    persp_intensity = random.randint(100, 500)
    persp_dir = random.randint(0, 1)
    dim_intensity = random.randint(0, 160)
    initial_blur_intensity = random.randint(0,2)
    vshift_size = random.randint(-50, 50)
    secondary_blur_intensity = random.randint(0,8)

    image = apply_perspective(image, persp_intensity, persp_dir)
    image = dim(image, dim_intensity)
    image = blur(image, initial_blur_intensity)

    extracted, _ = find_plate(image)
    if extracted is None:
        return None

    extracted = v_shift(extracted, vshift_size)
    extracted = blur(extracted, secondary_blur_intensity)

    return extracted



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    def show_plot(image):
        plt.figure()
        plt.imshow(image)
        plt.show()


    # Show breakdown of a single image
    image_name = sys.argv[1]
    image = cv2.imread(image_name)

    # Initialize variables
    persp_intensity = random.randint(100, 500)
    persp_dir = random.randint(0, 1)
    dim_intensity = random.randint(0, 160)
    initial_blur_intensity = random.randint(0,2)
    vshift_size = random.randint(-50, 50)
    secondary_blur_intensity = random.randint(0,8)

    # Print variables
    print("persp_intensity: {}".format(persp_intensity))
    print("persp_dir: {}".format(persp_dir))
    print("dim_intensity: {}".format(dim_intensity))
    print("initial_blur_intensity: {}".format(initial_blur_intensity))
    print("secondary_blur_intensity: {}".format(secondary_blur_intensity))
    print("vshift_size: {}".format(vshift_size))

    print("Displaying source image")
    show_plot(image)

    print("Applying perspective")
    image = apply_perspective(image, persp_intensity, persp_dir)
    show_plot(image)

    print("Applying dimming")
    image = dim(image, dim_intensity)
    show_plot(image)

    print("Applying initial blur")
    image = blur(image, initial_blur_intensity)
    show_plot(image)

    extracted, _ = find_plate(image)

    if extracted is None:
        print("Was unable to recover plate")
    else:
        extracted = v_shift(extracted, vshift_size)
        print("Showing extracted plate")
        show_plot(extracted)

        print("Applying secondary blur")
        extracted = blur(extracted, secondary_blur_intensity)
        show_plot(extracted)