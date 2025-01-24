import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

# File path for the image
im_path = r'C:\Users\hp\Desktop\Projects\Projects\OCR\original_image.jpeg'

# Function to display an image using matplotlib
def display(im_path):
    dpi = 80
    im_data = cv2.imread(im_path)
    height, width, _ = im_data.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB))
    plt.show()

# Load and display the original image
img = cv2.imread(im_path)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Function to convert an image to grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to remove noise from an image
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

# Function to thin font in an image
def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return cv2.bitwise_not(image)

# Function to thicken font in an image
def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return cv2.bitwise_not(image)

# Function to get skew angle
def get_skew_angle(cv_image):
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    min_area_rect = cv2.minAreaRect(largest_contour)
    angle = min_area_rect[-1]
    return angle + 90 if angle < -45 else angle

# Function to rotate image
def rotate_image(cv_image, angle):
    (h, w) = cv_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(cv_image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Function to deskew image
def deskew(cv_image):
    angle = get_skew_angle(cv_image)
    return rotate_image(cv_image, -angle)

# Function to remove borders from an image
def remove_borders(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y+h, x:x+w]

# Function to add borders to an image
def add_borders(image, border_size=150, color=(255, 255, 255)):
    return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=color)

# Image manipulations
inverted_image = cv2.bitwise_not(img)
gray_image = grayscale(img)
thresh, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
no_noise_image = noise_removal(binary_image)
thin_image = thin_font(no_noise_image)
thick_image = thick_font(no_noise_image)
deskewed_image = deskew(img)
removed_borders_image = remove_borders(no_noise_image)
image_with_borders = add_borders(removed_borders_image)

# Save manipulated images
cv2.imwrite(r"C:\Users\hp\Desktop\Projects\Projects\OCR\inverted_image.jpeg", inverted_image)
cv2.imwrite(r"C:\Users\hp\Desktop\Projects\Projects\OCR\gray_image.jpeg", gray_image)
cv2.imwrite(r"C:\Users\hp\Desktop\Projects\Projects\OCR\binary_image.jpeg", binary_image)
cv2.imwrite(r"C:\Users\hp\Desktop\Projects\Projects\OCR\no_noise_image.jpeg", no_noise_image)
cv2.imwrite(r"C:\Users\hp\Desktop\Projects\Projects\OCR\thin_image.jpeg", thin_image)
cv2.imwrite(r"C:\Users\hp\Desktop\Projects\Projects\OCR\thick_image.jpeg", thick_image)
cv2.imwrite(r"C:\Users\hp\Desktop\Projects\Projects\OCR\deskewed_image.jpeg", deskewed_image)
cv2.imwrite(r"C:\Users\hp\Desktop\Projects\Projects\OCR\removed_borders_image.jpeg", removed_borders_image)
cv2.imwrite(r"C:\Users\hp\Desktop\Projects\Projects\OCR\image_with_borders.jpeg", image_with_borders)

# OCR using pytesseract
ocr_result = pytesseract.image_to_string(img)
print("OCR Result:", ocr_result)