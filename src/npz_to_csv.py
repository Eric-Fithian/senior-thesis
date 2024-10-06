import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def convert_npz_to_csv(npz_path, csv_dir, include_face_area=True):
    """
    Convert a .npz file containing embeddings and related data to a CSV file.

    Parameters:
    - npz_path (str): Path to the input .npz file.
    - csv_path (str): Path to the output CSV file.
    - include_face_area (bool): Whether to include face_area in the CSV.
    """
    csv_filename = os.path.splitext(os.path.basename(npz_path))[0] + '.csv'
    csv_path = os.path.join(csv_dir, csv_filename)

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            frame_numbers = data['frame_numbers']
            face_confidences = data['face_confidences']
            embeddings = data['embeddings']
            face_areas = data['face_areas'] if 'face_areas' in data else None

        # Prepare data dictionary
        data_dict = {
            'frame': frame_numbers,
            'face_confidence': face_confidences
        }

        # Add embeddings
        embedding_dim = embeddings.shape[1] if embeddings.ndim > 1 else 1
        for i in tqdm(range(embedding_dim), leave=False, desc="Adding embeddings"):
            data_dict[f'emb{i}'] = embeddings[:, i] if embeddings.ndim > 1 else embeddings

        # Add face_area if requested and available
        if include_face_area and face_areas is not None:
            # Assuming face_area is a list of dictionaries with keys 'x', 'y', 'w', 'h'
            face_area_x = []
            face_area_y = []
            face_area_w = []
            face_area_h = []
            for area in face_areas:
                if isinstance(area, dict):
                    face_area_x.append(area.get('x', np.nan))
                    face_area_y.append(area.get('y', np.nan))
                    face_area_w.append(area.get('w', np.nan))
                    face_area_h.append(area.get('h', np.nan))
                else:
                    face_area_x.append(np.nan)
                    face_area_y.append(np.nan)
                    face_area_w.append(np.nan)
                    face_area_h.append(np.nan)

            data_dict['face_area_x'] = face_area_x
            data_dict['face_area_y'] = face_area_y
            data_dict['face_area_w'] = face_area_w
            data_dict['face_area_h'] = face_area_h

        # Create DataFrame
        df = pd.DataFrame(data_dict)

        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Successfully converted {os.path.basename(npz_path)} to {os.path.basename(csv_path)}")

    except Exception as e:
        print(f"Failed to convert {npz_path}: {e}")

def process_all_npz(input_dir, output_dir, include_face_area=True):
    """
    Process all .npz files in the input directory and convert them to CSV.

    Parameters:
    - input_dir (str): Directory containing .npz files.
    - output_dir (str): Directory where CSV files will be saved.
    - include_face_area (bool): Whether to include face_area in the CSV.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]

    if not npz_files:
        print(f"No .npz files found in {input_dir}.")
        return

    print(f"Found {len(npz_files)} .npz files in {input_dir}. Converting to CSV...")

    for npz_file in tqdm(npz_files, desc="Converting .npz to CSV"):
        npz_path = os.path.join(input_dir, npz_file)
        csv_filename = os.path.splitext(npz_file)[0] + '.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        convert_npz_to_csv(npz_path, csv_path, include_face_area=include_face_area)

    print("All files have been converted.")

convert_npz_to_csv('../data/processed/deepface/P006_I_embeddings.npz', '../data/processed/deepface/csv', include_face_area=True)

df = pd.read_csv('../data/processed/deepface/csv/P006_I_embeddings.csv')
df.info()
df.head()

#get second row, sort it in desc and print it
df.iloc[1].sort_values(ascending=False)

npz_path = '../data/processed/wmei/P006_I_wmei.npz'
npz_data = np.load(npz_path)

npz_data['frame_numbers'][0:10]
npz_data['wmei_features'][0:10]