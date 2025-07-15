import numpy as np
import cv2

# Create a black image
width, height = 640, 480
fps = 24
duration = 5  # seconds
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('examples/test_video.mp4', fourcc, fps, (width, height))

for i in range(fps * duration):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw a white square that moves across the screen
    x = int(100 + 200 * i / (fps * duration))
    y = int(100 + 100 * i / (fps * duration))
    w, h = 50, 50
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)
    out.write(frame)

out.release()
