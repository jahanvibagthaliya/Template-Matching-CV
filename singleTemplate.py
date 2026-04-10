import cv2
import numpy as np
from matplotlib import pyplot as plt

# STEP 1: Load the images
# We usually convert to grayscale because color doesn't always help 
# for simple shape matching and it's much faster.
img_rgb = cv2.imread('./data/Photo1.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('./data/Template1.jpg',0)  # '0' loads as grayscale

# Get the width and height of the template to draw the box later
w, h = template.shape[::-1]

# STEP 2: Perform Template Matching
# We use CCOEFF_NORMED for the best accuracy in real-world lighting
result = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

# STEP 3: Find the "Brightest" point (The Best Match)
threshold = 0.98
loc = np.where(result >= threshold)

# STEP 4: Draw the result
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

# Show the results
cv2.imshow('Result Map (The Heatmap)', result)      
cv2.imshow('result', img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()