# -*- coding: utf-8 -*-
"""extract-video-frames.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qGAGA9usQjKD_7xcHKIA5f-LGUR3rPG3
"""

!pip install pytube

from pytube import YouTube
import cv2
import os

def download_and_extract_frames(video_url, output_directory="/content/drive/MyDrive/Videoset-frame/6", frame_rate=1):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Download video
    yt = YouTube(video_url)
    video_stream = yt.streams.filter(res="480p").first() #If video resolution has 480p value algoritm choose this resolution
    video_file_path = os.path.join(output_directory, "ytvit.mp4")
    video_stream.download(output_path=output_directory, filename="ytvit.mp4")

    # Extract frames
    cap = cv2.VideoCapture(video_file_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // frame_rate

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Save frame as an image every 'frame_interval' frames
        if frame_count % frame_interval == 0:
            # Resize the frame to 640x480
            resized_frame = cv2.resize(frame, (640, 480))

            frame_filename = os.path.join(output_directory, f"frame_{frame_count // frame_interval}.jpg")
            cv2.imwrite(frame_filename, resized_frame)

            # Get and print the dimensions of the saved frame
            height, width, _ = resized_frame.shape
            print(f"Frame {frame_count // frame_interval} Dimensions: {width}x{height}")

    cap.release()

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=UvLr-T9-zxk"
    output_directory = "/content/drive/MyDrive/Videoset-frame/6"

    download_and_extract_frames(video_url, output_directory)

