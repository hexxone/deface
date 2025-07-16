import cv2
import numpy as np

img = cv2.imread('examples/city.jpg')
height, width, layers = img.shape
size = (width,height)

out = cv2.VideoWriter('examples/city.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 1, size)

for i in range(10):
    out.write(img)

out.release()
