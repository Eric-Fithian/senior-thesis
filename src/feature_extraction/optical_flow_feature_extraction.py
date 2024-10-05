import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from scipy.stats import skew, kurtosis
from multiprocessing import Pool, cpu_count

def extract_optical_flow_features(video_path, output_dir, resize_dim=None, show_progress=True):
    """
    Processes a video to compute optical flow using RLOF, extracts statistical features from each flow frame,
    and saves the features to a CSV file.
    
    Parameters:
    - video_path (str): Path to the input video file.
    - output_dir (str): Directory to save the CSV file containing optical flow features.
    - resize_dim (tuple, optional): New size for frames (width, height). If None, original size is used.
    - show_progress (bool): Whether to show a progress bar. Default is True.
    """
    
    # Validate input parameters
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Derive CSV filename from video filename if not provided
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    csv_filename = f'{video_basename}_rlof.csv'
    
    output_file = os.path.join(output_dir, csv_filename)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Read the first frame
    ret, frame_prev = cap.read()
    if not ret:
        raise ValueError("Cannot read the first frame of the video.")
    
    # Optionally resize frame
    if resize_dim is not None:
        frame_prev = cv2.resize(frame_prev, resize_dim)
    
    # Initialize list to store features
    feature_list = []
    
    # Get total number of frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 1  # Already read the first frame
    
    # Initialize progress bar
    if show_progress:
        pbar = tqdm(total=total_frames - 1, desc="Processing Optical Flow Frames")
    
    frame_idx = 2  # Starting from the second frame
    
    while True:
        ret, frame_next = cap.read()
        if not ret:
            break
        
        # Optionally resize frame
        if resize_dim is not None:
            frame_next = cv2.resize(frame_next, resize_dim)
        
        # Calculate optical flow using RLOF
        flow = cv2.optflow.calcOpticalFlowDenseRLOF(frame_prev, frame_next, None, None)
        
        # Compute magnitude and angle from flow
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
        
        # Flatten the magnitude and angle for statistical calculations
        flat_mag = mag.flatten()
        flat_ang = ang.flatten()
        
        # Avoid division by zero in entropy calculation
        mag_sum = flat_mag.sum()
        if mag_sum == 0:
            norm_mag = np.zeros_like(flat_mag, dtype=np.float32)
        else:
            norm_mag = flat_mag / mag_sum
        
        ang_sum = flat_ang.sum()
        if ang_sum == 0:
            norm_ang = np.zeros_like(flat_ang, dtype=np.float32)
        else:
            norm_ang = flat_ang / ang_sum
        
        # Compute statistical features for magnitude
        mag_features = {
            'frame': frame_idx,
            'mag_total': np.sum(flat_mag),
            'mag_mean': np.mean(flat_mag),
            'mag_median': np.median(flat_mag),
            'mag_std': np.std(flat_mag),
            'mag_variance': np.var(flat_mag),
            'mag_skewness': skew(flat_mag),
            'mag_kurtosis': kurtosis(flat_mag),
            'mag_min': np.min(flat_mag),
            'mag_max': np.max(flat_mag),
            'mag_range': np.max(flat_mag) - np.min(flat_mag),
            'mag_percentile_10': np.percentile(flat_mag, 10),
            'mag_percentile_25': np.percentile(flat_mag, 25),
            'mag_percentile_75': np.percentile(flat_mag, 75),
            'mag_percentile_90': np.percentile(flat_mag, 90),
            'mag_iqr': np.percentile(flat_mag, 75) - np.percentile(flat_mag, 25),
            'mag_entropy': -np.sum(norm_mag * np.log2(norm_mag + 1e-10)),  # Added epsilon to avoid log(0)
            'mag_energy': np.sum(flat_mag ** 2),
            'mag_mode': float(pd.Series(flat_mag).mode()[0]) if not pd.Series(flat_mag).mode().empty else 0,
            'mag_unique_values': len(np.unique(flat_mag))
        }
        
        # Compute statistical features for angle
        ang_features = {
            'ang_total': np.sum(flat_ang),
            'ang_mean': np.mean(flat_ang),
            'ang_median': np.median(flat_ang),
            'ang_std': np.std(flat_ang),
            'ang_variance': np.var(flat_ang),
            'ang_skewness': skew(flat_ang),
            'ang_kurtosis': kurtosis(flat_ang),
            'ang_min': np.min(flat_ang),
            'ang_max': np.max(flat_ang),
            'ang_range': np.max(flat_ang) - np.min(flat_ang),
            'ang_percentile_10': np.percentile(flat_ang, 10),
            'ang_percentile_25': np.percentile(flat_ang, 25),
            'ang_percentile_75': np.percentile(flat_ang, 75),
            'ang_percentile_90': np.percentile(flat_ang, 90),
            'ang_iqr': np.percentile(flat_ang, 75) - np.percentile(flat_ang, 25),
            'ang_entropy': -np.sum(norm_ang * np.log2(norm_ang + 1e-10)),  # Added epsilon to avoid log(0)
            'ang_energy': np.sum(flat_ang ** 2),
            'ang_mode': float(pd.Series(flat_ang).mode()[0]) if not pd.Series(flat_ang).mode().empty else 0,
            'ang_unique_values': len(np.unique(flat_ang))
        }
        
        # Combine magnitude and angle features
        combined_features = {**mag_features, **ang_features}
        
        # Append to the feature list
        feature_list.append(combined_features)
        
        # Update for next iteration
        frame_prev = frame_next
        frame_idx += 1
        processed_frames += 1
        
        if show_progress:
            pbar.update(1)
    
    if show_progress:
        pbar.close()
    cap.release()
    
    features_df = pd.DataFrame(feature_list)
    features_df.to_csv(output_file, index=False)
    print(f"Optical flow features saved to {output_file}")
    
    
def process_video(args):
    video_path, output_dir, resize_dim = args
    print(f"Processing {video_path} ...")
    extract_optical_flow_features(video_path, output_dir, resize_dim, show_progress=False)

def extract_bulk(video_dir, output_dir, resize_dim):
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    print('video files to process:', video_files)

    tasks = []
    for video_file in video_files:
        # check if video has already been processed
        video_basename = video_file.split('.')[0]
        output_file = os.path.join(output_dir, f'{video_basename}_rlof.csv')
        if os.path.exists(output_file):
            print(f"RLOF features for {video_file} already exist. Skipping ...")
            continue

        video_path = os.path.join(video_dir, video_file)
        tasks.append((video_path, output_dir, resize_dim))
    
    # Use multiprocessing Pool to process videos in parallel
    pool = Pool(processes=cpu_count())
    total_tasks = len(tasks)
    with tqdm(total=total_tasks, desc="Total Progress") as pbar:
        for _ in pool.imap_unordered(process_video, tasks):
            pbar.update(1)
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    video_dir = '../../data/interim/video'
    output_dir = '../../data/processed/rlof'
    resize_dim = (320, 240)
    
    extract_bulk(video_dir, output_dir, resize_dim)
    

# # Example usage
# video_path = '../../data/interim/video/P005_I.mp4' 
# output_dir = '../../data/processed/rlof'           

# # Extract RLOF features
# # make resize small so it can run faster
# extract_optical_flow_features(video_path, output_dir, resize_dim=(320, 240), show_progress=True)
