import av
import numpy as np
import pandas as pd
import os
import glob
import zipfile
from tqdm import tqdm  # for progress bar

def extract_zip_if_needed(zip_path, extract_dir):
    """
    Check if zip file needs to be extracted and extract if necessary.
    """
    chunk_path = os.path.join(extract_dir, 'Chunk_1')
    
    if not os.path.exists(zip_path):
        print(f"Warning: Zip file not found at {zip_path}")
        return False
        
    if os.path.exists(extract_dir) and os.path.exists(chunk_path):
        print(f"Data already extracted at {extract_dir}")
        return True
        
    print(f"\nExtracting {zip_path} to {extract_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size of files to extract
            total_size = sum(file.file_size for file in zip_ref.filelist)
            extracted_size = 0
            
            # Extract each file with progress
            for file in zip_ref.filelist:
                zip_ref.extract(file, extract_dir)
                extracted_size += file.file_size
                progress = (extracted_size / total_size) * 100
                print(f"\rExtraction progress: {progress:.1f}%", end="", flush=True)
        
        print("\nExtraction completed successfully!")
        return True
        
    except zipfile.BadZipFile:
        print(f"Error: The file {zip_path} is not a valid zip file")
        return False
    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        return False

def sync_video_and_steering_with_more_state_data(segment):
    """
    Sync current video frame and sensor data with future steering angle (t+200ms)
    """
    base_path = os.path.join(segment, 'processed_log')
    
    # Load all sensor data
    steering_times = np.load(os.path.join(base_path, 'CAN', 'steering_angle', 't'))
    steering_angles = np.load(os.path.join(base_path, 'CAN', 'steering_angle', 'value'))
    speed_times = np.load(os.path.join(base_path, 'CAN', 'speed', 't'))
    speed_values = np.load(os.path.join(base_path, 'CAN', 'speed', 'value'))
    gyro_times = np.load(os.path.join(base_path, 'IMU', 'gyro', 't'))
    gyro_values = np.load(os.path.join(base_path, 'IMU', 'gyro', 'value'))
    accel_times = np.load(os.path.join(base_path, 'IMU', 'accelerometer', 't'))
    accel_values = np.load(os.path.join(base_path, 'IMU', 'accelerometer', 'value'))
    frame_times = np.load(os.path.join(segment, 'global_pose', 'frame_times'))
    frame_velocities = np.load(os.path.join(segment, 'global_pose', 'frame_velocities'))

    synced_data = []
    
    # Function to get closest measurement for any sensor
    def get_measurement_at_time(times, values, target_time):
        idx = np.argmin(np.abs(times - target_time))
        return values[idx]
    
    # Process each frame
    for frame_idx in range(len(frame_times)):
        current_time = frame_times[frame_idx]
        future_time = current_time + 0.2  # 200ms in the future
        
        # Check if we have steering data for future time
        if future_time > steering_times[-1]:
            break
            
        # Get current measurements (inputs)
        data_point = {
            'frame_idx': frame_idx,
            'frame_time': current_time,
            
            # Current sensor data (inputs)
            'speed': get_measurement_at_time(speed_times, speed_values, current_time),
            'gyro_x': get_measurement_at_time(gyro_times, gyro_values, current_time)[0],
            'gyro_y': get_measurement_at_time(gyro_times, gyro_values, current_time)[1],
            'gyro_z': get_measurement_at_time(gyro_times, gyro_values, current_time)[2],
            'accel_x': get_measurement_at_time(accel_times, accel_values, current_time)[0],
            'accel_y': get_measurement_at_time(accel_times, accel_values, current_time)[1],
            'accel_z': get_measurement_at_time(accel_times, accel_values, current_time)[2],
            'velocity_x': frame_velocities[frame_idx][0],
            'velocity_y': frame_velocities[frame_idx][1],
            'velocity_z': frame_velocities[frame_idx][2],
            'current_steering': get_measurement_at_time(steering_times, steering_angles, current_time),
            
            # Future steering angle (target)
            'future_steering': get_measurement_at_time(steering_times, steering_angles, future_time)
        }
        
        synced_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(synced_data)
    
    tqdm.write(f"\nProcessed {len(df)} frames from {segment}")
    return df

if __name__ == "__main__":
    # Base paths
    data_dir = 'data'
    comma2k19_dir = os.path.join(data_dir, 'comma2k19')
    extract_dir = os.path.join(data_dir, 'extracted')  # at same level as comma2k19
    chunk_path = os.path.join(extract_dir, 'Chunk_1')
    zip_path = os.path.join(comma2k19_dir, 'Chunk_1.zip')

    # Create necessary directories
    os.makedirs('data_synced', exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    # Check and extract the zip file if needed
    if not os.path.exists(chunk_path):
        print("Chunk_1 directory not found, checking for zip file...")
        success = extract_zip_if_needed(zip_path, extract_dir)
        if not success:
            print("Error: Could not find or extract the dataset. Please ensure Chunk_1.zip is in the data/comma2k19 directory.")
            exit(1)

    # Find all segment directories
    segments = []
    for root, dirs, files in os.walk(chunk_path):
        if 'processed_log' in dirs:  # This indicates we're in a segment directory
            segments.append(root)  # Don't add trailing slash

    print(f"Found {len(segments)} segments to process")

    # Process each segment with progress bar
    for segment in tqdm(segments, position=0, leave=True):
        try:
            # Get the segment number (last folder in path)
            segment_num = os.path.basename(segment)
            
            # Get the parent folder name (contains identifier and timestamp)
            parent_folder = os.path.basename(os.path.dirname(segment))
            
            # Create output filename by combining parent folder and segment number
            output_file = os.path.join('data_synced', f'{parent_folder}_{segment_num}.csv')
            
            tqdm.write(f"Processing: {segment}")
            
            # Process the segment
            df = sync_video_and_steering_with_more_state_data(segment)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            tqdm.write(f"Saved {len(df)} frames to {output_file}")
            
        except Exception as e:
            tqdm.write(f"Error processing segment {segment}: {str(e)}")
            continue

    print("\nProcessing complete!")
    print(f"Files saved in data_synced/")
