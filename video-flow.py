import cv2
import os
import numpy as np

eps = 10e-2

def extract_frames_and_optical_flow(video_path, output_folder):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    # Create a folder to save the frames and optical flow images
    os.makedirs(output_folder, exist_ok=True)
    # Create a CSV file path within the output folder
    csv_file = os.path.join(output_folder, "flow_vectors.csv")
    # Open the CSV file to write flow vectors
    with open(csv_file, 'w') as f:
        f.write("Frame,x,y,Flow_x,Flow_y\n")
        # Number of frames gone through in loop
        frame_count = 0
        # Optical flow parameters
        prev_frame = None
        while True:
            # Read the next frame
            ret, frame = video.read()
            # If there are no more frames, break the loop
            if not ret:
                break
            # Save the frame as an image
            frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            # Calculate optical flow if not the first frame
            if prev_frame is not None:
                # Convert frames to grayscale
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Save the optical flow image
                flow_image_path = os.path.join(output_folder, f"optical_flow_{frame_count:04d}.png")
                flow_vis = draw_flow(frame_gray, flow)
                cv2.imwrite(flow_image_path, flow_vis)
                
                # Write rounded flow vectors to CSV file
                for y in range(0, flow.shape[0], 16):
                    for x in range(0, flow.shape[1], 16):
                        dx, dy = flow[y, x]
                        if (abs(dx) > eps and abs(dy) > eps):
                            f.write(f"{frame_count},{x},{y},{round(dx, 2)},{round(dy, 2)}\n")
                
            prev_frame = frame.copy()
            # Increment frame count
            frame_count += 1
    # Release the video object
    video.release()

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

for i in range(1, 15):

    video_name= "video" + str(i) + ".mp4"
    video_path = "Videos/" + video_name
    output_folder = "output_video" + str(i)

    # Extract frames from the video, calculate optical flow, and save flow vectors
    extract_frames_and_optical_flow(video_path, output_folder)