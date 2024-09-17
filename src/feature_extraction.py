# OpenFace feature extraction through docker container

### CLI Instructions:
# Docker Command:
# docker run -it -v $(pwd)/data/interim/video:/data -v $(pwd)/data/processed:/output algebr/openface:latest
# build/bin/FeatureExtraction -f /data/P{###}_{P/I}.mp4 -out_dir /output


import os
import subprocess
import sys
from tqdm import tqdm


def extract_features(video_path, output_dir):
    video_id = os.path.basename(video_path).split(".")[0]
    video_basename = os.path.basename(video_path)

    command = f"docker run --platform linux/amd64 -v {os.path.abspath(video_dir)}:/data -v {os.path.abspath(output_dir)}:/output --rm algebr/openface:latest build/bin/FeatureExtraction -f /data/{video_basename} -out_dir {output_dir}"
    subprocess.run(command, shell=True)

video_dir = '../data/raw/Video/Side-By-Side'

video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

for video_file in video_files[:1]:
    video_name = video_file.split(".")[0]
    extract_features(os.path.join(video_dir, video_file), f'data/processed/{video_name}')

