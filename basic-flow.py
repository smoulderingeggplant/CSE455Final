import cv2
import numpy as np

def draw_flow(image, flow, step=16):
    h, w = image.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_, _) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def main():
    # Load two consecutive frames
    frame1 = cv2.imread("frame1.png", cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread("frame2.png", cv2.IMREAD_GRAYSCALE)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Draw the optical flow on the second frame
    flow_image = draw_flow(frame2, flow)

    # Save the optical flow image
    cv2.imwrite("optical_flow.png", flow_image)

if __name__ == "__main__":
    main()