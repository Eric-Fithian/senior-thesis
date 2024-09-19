# OpenFace feature extraction through docker container

### Shell Instructions for OpenFace Feature Extraction:
# Docker Command:
# $ docker run -it -v $(pwd)/data/interim/video:/data -v $(pwd)/data/processed:/output algebr/openface:latest
# $ build/bin/FeatureExtraction -f /data/P{###}_{P/I}.mp4 -out_dir /output -2Dfp -3Dfp -pdmparams -pose -aus -gaze
# -f /data/P028_P.mp4 -f /data/P028_I.mp4 -f /data/P029_P.mp4 -f /data/P029_I.mp4 -f /data/P031_P.mp4 -f /data/P031_I.mp4 -f /data/P032_P.mp4 -f /data/P032_I.mp4 -f /data/P033_P.mp4 -f /data/P033_I.mp4 -f /data/P034_P.mp4 -f /data/P034_I.mp4 -f /data/P035_P.mp4 -f /data/P035_I.mp4 -f /data/P036_P.mp4 -f /data/P036_I.mp4 -f /data/P037_P.mp4 -f /data/P037_I.mp4 -f /data/P038_P.mp4 -f /data/P038_I.mp4 -f /data/P039_P.mp4 -f /data/P039_I.mp4 -f /data/P040_P.mp4 -f /data/P040_I.mp4 -f /data/P041_P.mp4 -f /data/P041_I.mp4
# 30 ish minutes per video file


from tqdm import tqdm
from deepface import DeepFace
import cv2
import os
import numpy as np

# result = DeepFace.represent(
#     img_path="../data/raw/Video/Switching/P001_P.png",
#     enforce_detection=False,
#     model_name="Facenet",
#     detector_backend="opencv",
#     max_faces=1
# )[0]

# result

# result['embedding']
# result['face_confidence']
# result['facial_area']

def extract_deepface_embeddings(video_path, output_path, model_name="VGG-Face", detector_backend="opencv", frame_skip=1):
    # detector_backend: opencv, dlib are faster than ssd and mtcnn but less accurate

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    embeddings_list = []
    face_confidence_list = []
    face_areas_list = []
    frame_numbers = []

    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}") as pbar:
        frame_count = 0
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1
            pbar.update(1)

            if frame_count % frame_skip != 0:
                continue

            try:
                result = DeepFace.represent(
                    img_path=frame,
                    enforce_detection=False,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    max_faces = 1
                )[0]

                if result:
                    embeddings_list.append(result['embedding'])
                    face_confidence_list.append(result['face_confidence'])
                    face_areas_list.append(result['facial_area'])
                    frame_numbers.append(frame_count)

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")

    cap.release()

    video_basename = os.path.basename(video_path).split('.')[0]
    output_file = os.path.join(output_path, f'{video_basename}_embeddings.npz')
    np.savez(
        output_file,
        embeddings=np.array(embeddings_list),
        face_confidences=np.array(face_confidence_list),
        face_areas=np.array(face_areas_list),
        frame_numbers=np.array(frame_numbers)
    )

    print(f"Embeddings saved to {output_file}")

def extract_bulk(video_dir, output_dir, model_name="VGG-Face", detector_backend="opencv", frame_skip=1):
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    print('video files to process:', video_files)

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)

        print(f"Processing {video_path} ... to {output_dir}")
        extract_deepface_embeddings(video_path, output_dir, model_name, detector_backend, frame_skip)

# model_name = 'Facenet'
model_name = 'VGG-Face'
detector_backend = 'opencv'
frame_skip = 3

extract_bulk(
    video_dir='../data/interim/video',
    output_dir='../data/processed/deepface',
    model_name=model_name,
    detector_backend=detector_backend,
    frame_skip=frame_skip
)

data = np.load("../data/processed/deepface/P005_I_embeddings.npz")

data['embeddings'][0]
