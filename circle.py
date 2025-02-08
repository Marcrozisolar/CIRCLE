import cv2
import numpy as np

# Load the image
image = cv2.imread("D:\Mark\Pictures\cans.jpg")
if image is None:
    print("Error: Image not loaded. Check the file path.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocessing: Enhance and blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection
edges = cv2.Canny(blurred, 30, 100)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Refine contours
refined_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if area > 100 and 0.6 < circularity <= 1.5:  # Looser thresholds
            refined_contours.append(contour)

# Create a mask and draw refined contours
mask = np.zeros_like(gray)
cv2.drawContours(mask, refined_contours, -1, 255, -1)

overlay = image.copy()
cv2.drawContours(overlay, refined_contours, -1, (0, 255, 0), 2)

# Display the mask
cv2.imshow("Mask", mask)
cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
