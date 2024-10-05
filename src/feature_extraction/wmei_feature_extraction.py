import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from scipy.stats import skew, kurtosis
from multiprocessing import Pool, cpu_count

def extract_wmei_statistics(video_path, output_dir, alpha=0.7, threshold=30, resize_dim=None):
    """
    Processes a video to compute Weighted Motion Energy Images (WMEI) frame-by-frame,
    extracts statistical features from each WMEI frame, and saves the features to a CSV file.
    
    Parameters:
    - video_path (str): Path to the input video file.
    - output_dir (str): Directory to save the CSV file containing WMEI features.
    - alpha (float): Weighting factor for WMEI update. Default is 0.7.
    - threshold (int): Threshold value for motion detection. Default is 30.
    - resize_dim (tuple, optional): New size for frames (width, height). If None, original size is used.
    """
    
    # Validate input parameters
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Derive CSV filename from video filename
    video_basename = os.path.basename(video_path).split('.')[0]
    csv_filename = f'{video_basename}_wmei_stats.csv'
    
    output_file = os.path.join(output_dir, csv_filename)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Read the first two frames
    ret, frame1 = cap.read()
    if not ret:
        raise ValueError("Cannot read the first frame of the video.")
    
    ret, frame2 = cap.read()
    if not ret:
        raise ValueError("Cannot read the second frame of the video.")
    
    # Optionally resize frames
    if resize_dim is not None:
        frame1 = cv2.resize(frame1, resize_dim)
        frame2 = cv2.resize(frame2, resize_dim)
    
    # Initialize WMEI as a float32 array for precision
    h, w = frame1.shape[:2]
    wmei = np.zeros((h, w), dtype=np.float32)
    
    # List to store feature dictionaries
    feature_list = []
    
    # Get total number of frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 2  # Already read two frames
    
    # Initialize progress bar
    # pbar = tqdm(total=total_frames - 2, desc="Processing WMEI Frames")
    
    frame_idx = 2  # IMPORTANT: This was 3 (which is incorrect) and the files that were generated are off by one on their frame index. They are 1 more than what they should be.
    
    while True:
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference between consecutive frames
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply binary threshold to highlight significant motion
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Update WMEI using the weighted sum
        wmei = alpha * wmei + (1 - alpha) * thresh.astype(np.float32)
        
        # Extract statistical features
        flat_wmei = wmei.flatten()
        
        # Avoid division by zero in entropy calculation
        wmei_sum = flat_wmei.sum()
        if wmei_sum == 0:
            normalized_wmei = np.zeros_like(flat_wmei, dtype=np.float32)
        else:
            normalized_wmei = flat_wmei / wmei_sum
        
        # Compute statistical features
        total_motion = np.sum(flat_wmei)
        mean_motion = np.mean(flat_wmei)
        median_motion = np.median(flat_wmei)
        std_motion = np.std(flat_wmei)
        variance_motion = np.var(flat_wmei)
        skewness_motion = skew(flat_wmei)
        kurtosis_motion = kurtosis(flat_wmei)
        min_motion = np.min(flat_wmei)
        max_motion = np.max(flat_wmei)
        range_motion = max_motion - min_motion
        percentile_10 = np.percentile(flat_wmei, 10)
        percentile_25 = np.percentile(flat_wmei, 25)
        percentile_75 = np.percentile(flat_wmei, 75)
        percentile_90 = np.percentile(flat_wmei, 90)
        iqr_motion = percentile_75 - percentile_25
        entropy_motion = -np.sum(normalized_wmei * np.log2(normalized_wmei + 1e-10))  # Added epsilon to avoid log(0)
        energy_motion = np.sum(flat_wmei ** 2)
        mode_motion_series = pd.Series(flat_wmei).mode()
        mode_motion = float(mode_motion_series.iloc[0]) if not mode_motion_series.empty else 0
        unique_values = len(np.unique(flat_wmei))
        
        # Compile features into a dictionary
        feature_dict = {
            'frame': frame_idx,
            'total_motion': total_motion,
            'mean_motion': mean_motion,
            'median_motion': median_motion,
            'std_motion': std_motion,
            'variance_motion': variance_motion,
            'skewness_motion': skewness_motion,
            'kurtosis_motion': kurtosis_motion,
            'min_motion': min_motion,
            'max_motion': max_motion,
            'range_motion': range_motion,
            'percentile_10': percentile_10,
            'percentile_25': percentile_25,
            'percentile_75': percentile_75,
            'percentile_90': percentile_90,
            'iqr_motion': iqr_motion,
            'entropy_motion': entropy_motion,
            'energy_motion': energy_motion,
            'mode_motion': mode_motion,
            'unique_values': unique_values
        }
        
        feature_list.append(feature_dict)
        
        # Update frames for next iteration
        frame1 = frame2
        ret, frame2 = cap.read()
        frame_idx += 1
        processed_frames += 1
        # pbar.update(1)
        
        if not ret:
            break
    
    # pbar.close()
    cap.release()
    
    # Create a DataFrame from the feature list
    features_df = pd.DataFrame(feature_list)
    
    # Save features to CSV
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
    
    
def process_video(args):
    video_path, output_dir = args
    print(f"Processing {video_path} ...")
    extract_wmei_statistics(video_path, output_dir)

def extract_bulk(video_dir, output_dir):
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    print('video files to process:', video_files)

    tasks = []
    for video_file in video_files:
        # check if video has already been processed
        video_basename = video_file.split('.')[0]
        output_file = os.path.join(output_dir, f'{video_basename}_wmei_stats.csv')
        if os.path.exists(output_file):
            print(f"WMEI features for {video_file} already exist. Skipping ...")
            continue

        video_path = os.path.join(video_dir, video_file)
        tasks.append((video_path, output_dir))
    
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
    output_dir = '../../data/processed/wmei'
    
    extract_bulk(video_dir, output_dir)
    

# # Example usage
# video_path = '../../data/interim/video/P005_I.mp4' 
# output_dir = '../../data/processed/wmei'           

# # Extract WMEI features
# extract_wmei_statistics(video_path, output_dir)


# # Extract WMEI features for all videos in a directory
# video_dir = '../../data/interim/video'
# output_dir = '../../data/processed/wmei'

# extract_bulk(video_dir, output_dir)
