
import cv2

def combine_videos(input_files, output_file):
    # Initialize variables
    frames_written = 0
    frame_width, frame_height = None, None
    fps = None

    # Create VideoWriter object for output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Codec for MP4 format
    out = None

    try:
        for input_file in input_files:
            # Open input video file
            cap = cv2.VideoCapture(input_file)

            # Get frame width, height, and frame rate from first video file
            if frame_width is None or frame_height is None:
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Create VideoWriter object for output video
                out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

            # Read and write frames from input video to output video
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                    frames_written += 1
                else:
                    break

            # Release input video capture object
            cap.release()

        # Release output video writer object
        if out is not None:
            out.release()

        print(f"Combined {len(input_files)} videos into {output_file} with {frames_written} frames.")

    except Exception as e:
        if out is not None:
            out.release()
        print(f"Error occurred: {str(e)}")

input_files = ["ittr_0_prediction.mp4", "ittr_1_prediction.mp4", "ittr_2_prediction.mp4", "ittr_3_prediction.mp4"]
output_file = "combined_video.mp4"
combine_videos(input_files, output_file)


import imageio

def convert_to_gif(input_file, output_file, fps=10):
    try:
        # Read the video file using imageio
        reader = imageio.get_reader(input_file)

        # Initialize a writer to create the GIF
        writer = imageio.get_writer(output_file, fps=fps)

        # Iterate over each frame in the video and add it to the GIF writer
        for frame in reader:
            writer.append_data(frame)

        # Close the writer to save the GIF
        writer.close()

        print(f"Conversion completed. GIF saved as {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = "combined_video.mp4"
output_file = "output.gif"
convert_to_gif(input_file, output_file)