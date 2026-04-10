import cv2
import numpy as np

# STEP 1: Load the images
# We usually convert to grayscale because color doesn't always help 
# for simple shape matching and it's much faster.
main_image = cv2.imread('./data/Photo2.jpg')
gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template = cv2.imread('./data/Template2.jpg', 0) # '0' loads as grayscale

# Get the width and height of the template to draw the box later
w, h = template.shape[::-1]

# STEP 2: Perform Template Matching
# We use CCOEFF_NORMED for the best accuracy in real-world lighting
result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)

# STEP 3: Find the "Brightest" point (The Best Match)
# minMaxLoc returns the min value, max value, and their coordinates
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# For CCOEFF, the 'max_loc' is the top-left corner of our best match
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# STEP 4: Draw the result
cv2.rectangle(main_image, top_left, bottom_right, (0, 255, 0), 3)

# Show the results
cv2.imshow('Result Map (The Heatmap)', result)
cv2.imshow('Detected Object', main_image)
cv2.waitKey(0)
cv2.destroyAllWindows()