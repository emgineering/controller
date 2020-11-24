
# The main function here is 'find_plate(img)'

# param: img is a raw camera feed image
# returns: image with only the white rectangle with
#           parking spot number and license plate,
#           or None if not found

import cv2
import numpy as np


def get_gray_mask(image):
    # https://stackoverflow.com/questions/23680498/detect-gray-things-with-opencv

    low_thresh = 90
    high_thresh = 210
    low = image[:,:,0] > low_thresh
    high = image[:,:,0] < high_thresh

    bg = image[:,:,0] == image[:,:,1] # B == G
    gr = image[:,:,1] == image[:,:,2] # G == R

    mask = np.bitwise_and(np.bitwise_and(bg, gr), np.bitwise_and(low,high))
    return mask

def get_blue_mask(image):

    low_red = image[:,:,0] > image[:,:,2] 
    low_green = image[:,:,0] > image[:,:,1] 

    mask = np.bitwise_and(low_red, low_green)
    return mask


def locate_blue_gray_split(image):
    output = image[:]

    blue_mask = get_blue_mask(image) * 255
    gray_mask = get_gray_mask(image) * 255
    
    blue_blur = cv2.blur(blue_mask,(8,1))
    gray_blur = cv2.blur(gray_mask, (8,1))

    overlap = np.bitwise_and(blue_blur, gray_blur).astype('uint8')

    # Turn up minLineLength if edge is not getting detected
    # Turn up maxLineGap if the top and bottom sections are not getting connected
    lines = cv2.HoughLinesP(overlap,1,np.pi/180,40,minLineLength=20,maxLineGap=80)
    
    # Enable this block to visualize where the lines are being detected:
    #for i in range(len(lines)):
    #    for x1,y1,x2,y2 in lines[i]:
    #        cv2.line(output,(x1,y1),(x2,y2),(0,0,255),3)
    #plt.figure()
    #plt.imshow(output)

    return lines

# adapted from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

def order_points(pts):

	rect = np.zeros((4, 2), dtype = "float32")

	s = np.sum(pts, axis = 1)
	rect[0] = pts[np.argmin(s)] # lowest sum is top left
	rect[2] = pts[np.argmax(s)] # largest sum is bottom right

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)] # lowest diff is bottom left
	rect[3] = pts[np.argmax(diff)] # largest diff is top right

	return rect

def find_plate(image):
    lines = locate_blue_gray_split(image)

    if lines is not None and len(lines) == 2:

        p1 = lines[0][0][:2]
        p2 = lines[0][0][2:]
        p3 = lines[1][0][:2]
        p4 = lines[1][0][2:]

        rect = order_points((p1,p2,p3,p4))
        (tl, tr, br, bl) = rect

        h_distance = np.abs(tl[0] - tr[0])

        # dimensions of original generated plate textures
        maxWidth = 600
        maxHeight = 1800

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)

        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped, h_distance
    else:
        return None, None
