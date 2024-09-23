import cv2
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from deepface import DeepFace

def extract_deepface_embeddings(video_path, output_path, model_name="VGG-Face", detector_backend="opencv", frame_skip=1, show_progress=False):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    embeddings_list = []
    face_confidence_list = []
    face_areas_list = []
    frame_numbers = []

    if show_progress:
        pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")
    else:
        pbar = None

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if pbar:
            pbar.update(1)

        if frame_count % frame_skip != 0:
            continue

        try:
            # Use DeepFace.represent to get embedding
            result = DeepFace.represent(
                img_path=frame,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
                align=True,
                normalization='base'
            )

            if len(result) > 0:
                embedding = result[0]['embedding']
                face_confidence = result[0].get('face_confidence', 1.0)
                facial_area = result[0].get('facial_area', {})

                embeddings_list.append(embedding)
                face_confidence_list.append(face_confidence)
                face_areas_list.append(facial_area)
                frame_numbers.append(frame_count)
            else:
                # No face detected
                continue

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            continue

    cap.release()

    if pbar:
        pbar.close()

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

def process_video(args):
    video_path, output_dir, model_name, detector_backend, frame_skip = args
    print(f"Processing {video_path} ...")
    extract_deepface_embeddings(video_path, output_dir, model_name, detector_backend, frame_skip, show_progress=False)

def extract_bulk(video_dir, output_dir, model_name="VGG-Face", detector_backend="opencv", frame_skip=1):
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    print('Video files to process:', video_files)

    # Prepare tasks for multiprocessing
    tasks = []
    for video_file in video_files:
        video_basename = video_file.split('.')[0]
        output_file = os.path.join(output_dir, f'{video_basename}_embeddings.npz')
        if os.path.exists(output_file):
            print(f"Embeddings for {video_file} already exist. Skipping ...")
            continue

        video_path = os.path.join(video_dir, video_file)
        tasks.append((video_path, output_dir, model_name, detector_backend, frame_skip))

    # Use multiprocessing Pool to process videos in parallel
    pool = Pool(processes=cpu_count())
    total_tasks = len(tasks)
    with tqdm(total=total_tasks, desc="Total Progress") as pbar:
        for _ in pool.imap_unordered(process_video, tasks):
            pbar.update(1)
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Parameters
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