import cv2
import sys
import face_recognition
from scenedetect import detect, AdaptiveDetector
import numpy as np
import os
from tqdm import tqdm

def process_side_by_side(video_path, output_interviewer, output_interviewee):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    aspect_ratio = width / height
    content_width = width // 2
    content_height = int(content_width / aspect_ratio)
    content_start_height = (height - content_height) // 2
    content_end_height = content_start_height + content_height
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video resolution: {width}x{height}")
    print(f"Content resolution: {content_width}x{content_height}")


    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_interviewer = cv2.VideoWriter(output_interviewer, fourcc, fps, (width, height))
    out_interviewee = cv2.VideoWriter(output_interviewee, fourcc, fps, (width, height))

    for _ in tqdm(range(total_frames), leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        # Split the frame into left and right halves
        left_frame = frame[content_start_height:content_end_height, :content_width]
        right_frame = frame[content_start_height:content_end_height:, content_width:]

        # Resize the frames
        left_frame = cv2.resize(left_frame, (width, height))
        right_frame = cv2.resize(right_frame, (width, height))

        # Write the frames
        out_interviewer.write(left_frame)
        out_interviewee.write(right_frame)

    cap.release()
    out_interviewer.release()
    out_interviewee.release()
    print("Processing side-by-side video completed.")

def process_switching_video_via_scenedetect(video_path, output_1, output_2):
    scene_list = detect(video_path=video_path, detector=AdaptiveDetector(), show_progress=True)

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_out_1 = cv2.VideoWriter(output_1, fourcc, fps, (width, height))
    vid_out_2 = cv2.VideoWriter(output_2, fourcc, fps, (width, height))

    cur_output = output_1
    for i, scene in tqdm(enumerate(scene_list), leave=False):
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()

        for _ in tqdm(range(start_frame, end_frame), leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            if cur_output == output_1:
                vid_out_1.write(frame)
                vid_out_2.write(np.zeros((height, width, 3), dtype=np.uint8))
            else:
                vid_out_1.write(np.zeros((height, width, 3), dtype=np.uint8))
                vid_out_2.write(frame)
        
        # switch output
        cur_output = output_2 if cur_output == output_1 else output_1

    cap.release()
    vid_out_1.release()
    vid_out_2.release()

def process_switching_video(video_path, output_interviewer, output_interviewee, interviewer_image, interviewee_image):
    # Load reference images and get face encodings
    interviewer_image = face_recognition.load_image_file(interviewer_image)
    interviewer_encoding = face_recognition.face_encodings(interviewer_image)[0]

    interviewee_image = face_recognition.load_image_file(interviewee_image)
    interviewee_encoding = face_recognition.face_encodings(interviewee_image)[0]

    known_encodings = [interviewer_encoding, interviewee_encoding]
    known_names = ["interviewer", "interviewee"]

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_interviewer = cv2.VideoWriter(output_interviewer, fourcc, fps, (width, height))
    out_interviewee = cv2.VideoWriter(output_interviewee, fourcc, fps, (width, height))

    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    for _ in tqdm(range(frames), leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color to RGB color
        rgb_small_frame = np.array(small_frame[:, :, ::-1])

        # Find face encoding in the current frame
        face_encodings = face_recognition.face_encodings(rgb_small_frame)[0]

        # Compare the face encoding with known encodings
        person_in_frame = None
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                person_in_frame = known_names[first_match_index]
                break

        # Write frames accordingly
        if person_in_frame == "interviewer":
            out_interviewer.write(frame)
            out_interviewee.write(black_frame)
        elif person_in_frame == "interviewee":
            out_interviewer.write(black_frame)
            out_interviewee.write(frame)
        else:
            # If no face is recognized, write black frames
            out_interviewer.write(black_frame)
            out_interviewee.write(black_frame)

    cap.release()
    out_interviewer.release()
    out_interviewee.release()

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <side_by_side_video_dir> <switching_video_dir> <output_dir>")
        sys.exit(1)

    side_by_side_video_dir = sys.argv[1]
    switching_video_dir = sys.argv[2]
    output_dir = sys.argv[3]

    if not os.path.exists(side_by_side_video_dir):
        print(f"Directory {side_by_side_video_dir} does not exist.")
        sys.exit(1)
    
    if not os.path.exists(switching_video_dir):
        print(f"Directory {switching_video_dir} does not exist.")
        sys.exit(1)

    side_by_side_videos = [f for f in os.listdir(side_by_side_video_dir) if f.endswith(".mp4")]
    switching_videos = [f for f in os.listdir(switching_video_dir) if f.endswith(".mp4")]

    for video in tqdm(side_by_side_videos):
        video_path = os.path.join(side_by_side_video_dir, video)

        pnum = video.split(".")[0]

        output_interviewee = os.path.join(output_dir, pnum + "_P.mp4") # P for participant P###_P.mp4
        output_interviewer = os.path.join(output_dir, pnum + "_I.mp4") # I for interviewer P###_I.mp4
        process_side_by_side(video_path, output_interviewer, output_interviewee)



    for video in tqdm(switching_videos):
        video_path = os.path.join(switching_video_dir, video)

        pnum = video.split(".")[0]

        output_interviewee = os.path.join(output_dir, pnum + "_P.mp4") # P for participant P###_P.mp4
        output_interviewer = os.path.join(output_dir, pnum + "_I.mp4") # I for interviewer P###_I.mp4
        output_1 = os.path.join(output_dir, pnum + "_1.mp4")
        output_2 = os.path.join(output_dir, pnum + "_2.mp4")

        interviewee_face = os.path.join(switching_video_dir, pnum + "_P.png")
        interviewer_face = os.path.join(switching_video_dir, pnum + "_I.png")

        # process_switching_video(video_path, output_interviewer, output_interviewee, interviewer_face, interviewee_face)
        process_switching_video_via_scenedetect(video_path, output_1, output_2)

if __name__ == "__main__":
    main()