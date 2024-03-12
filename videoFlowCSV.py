import cv2
import os
import numpy as np

eps = 10e-2

def extract_frames_and_optical_flow(video_path, output_folder, file_name):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    # Create a folder to save the frames and optical flow images
    os.makedirs(output_folder, exist_ok=True)
    # Create a CSV file path within the output folder
    csv_file = os.path.join(output_folder, file_name)
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
            # Calculate optical flow if not the first frame
            if prev_frame is not None:
                # Convert frames to grayscale
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
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

for i in range(1, 33):
    letter = "S"
    video_name= letter + str(i) + ".mp4"
    video_path = "Videos/" + video_name
    output_folder = "CSVs"
    file_name = letter + str(i) + ".csv"

    # Extract frames from the video, calculate optical flow, and save flow vectors
    extract_frames_and_optical_flow(video_path, output_folder, file_name)