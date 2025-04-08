import av
import numpy as np
import pandas as pd
import os
import glob
import zipfile
from tqdm import tqdm  # for progress bar
from PIL import Image
import concurrent.futures
from functools import partial

# Configuration constants
TARGET_FPS = 22.0  # Desired FPS (will be capped by original video FPS)
CHUNK_NAME = "Chunk_1"  # Which chunk to process (e.g., "Chunk_1", "Chunk_2", etc.)

print(f"Processing {CHUNK_NAME} at {TARGET_FPS} FPS")

def extract_zip_if_needed(zip_path, extract_dir):
    """
    Check if zip file needs to be extracted and extract if necessary.
    """
    chunk_path = os.path.join(extract_dir, CHUNK_NAME)
    
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

def save_frame_image(frame, output_dir, frame_idx):
    """
    Save a video frame as an image file.
    
    Args:
        frame: Video frame from av container
        output_dir: Directory to save the image
        frame_idx: Frame index to use as filename
    """
    # Convert frame to RGB numpy array
    frame_array = frame.to_ndarray(format='rgb24')
    
    # Convert to PIL Image and resize
    frame_pil = Image.fromarray(frame_array)
    frame_pil = frame_pil.resize((400, 240), Image.Resampling.BILINEAR)
    
    # Save the image
    frame_path = os.path.join(output_dir, f"{frame_idx}.jpg")
    frame_pil.save(frame_path, quality=95)

def get_video_fps(video_path):
    """Get the original video frame rate"""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)  # or stream.guessed_rate if average_rate is None
        container.close()
        return fps
    except Exception as e:
        print(f"Error getting video FPS: {str(e)}")
        return None

def sync_video_and_steering_with_more_state_data(segment):
    """
    Sync current video frame and sensor data with future steering angle (t+200ms)
    with downsampling to specified FPS and frame saving
    """
    base_path = os.path.join(segment, 'processed_log')
    video_path = os.path.join(segment, 'video.hevc')
    
    # Get original video FPS
    original_fps = get_video_fps(video_path)
    if original_fps is None:
        raise ValueError(f"Could not determine FPS for video: {video_path}")
        
    # Use the lower of target FPS or original FPS
    effective_fps = min(TARGET_FPS, original_fps)
    print(f"\nVideo: {video_path}")
    print(f"Original FPS: {original_fps:.2f}")
    print(f"Target FPS: {TARGET_FPS:.2f}")
    print(f"Effective sampling FPS: {effective_fps:.2f}")
    
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

    # Create directory for frame images
    segment_name = os.path.basename(os.path.dirname(segment))
    subsegment_name = os.path.basename(segment)
    frames_dir = os.path.join('data_synced', f'{segment_name}_{subsegment_name}_frames')
    os.makedirs(frames_dir, exist_ok=True)

    synced_data = []
    
    def get_measurement_at_time(times, values, target_time):
        idx = np.argmin(np.abs(times - target_time))
        return values[idx]
    
    # Open video
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    
    # Process frames with downsampling
    frame_map = {}  # Map frame times to frame indices
    interval = 1.0 / effective_fps  # Time interval between frames
    
    for frame_idx, frame in enumerate(container.decode(video=0)):
        frame_time = frame_times[frame_idx]
        # Round to nearest interval based on effective FPS
        frame_time_rounded = np.floor(frame_time * effective_fps) / effective_fps
        
        # Only keep first frame for each interval
        if frame_time_rounded not in frame_map:
            frame_map[frame_time_rounded] = frame_idx
            future_time = frame_time + 0.2
            
            if future_time > steering_times[-1]:
                continue
                
            # Save frame image
            save_frame_image(frame, frames_dir, frame_idx)
            
            # Find previous steering angles (100ms and 200ms earlier)
            # Previous angle at least 100ms earlier
            prev_100ms_threshold = frame_time - 0.10
            prev_100ms_candidates = np.where(steering_times <= prev_100ms_threshold)[0]
            if len(prev_100ms_candidates) > 0:
                prev_100ms_idx = prev_100ms_candidates[-1]
                prev_100ms_angle = steering_angles[prev_100ms_idx]
                prev_100ms_time = steering_times[prev_100ms_idx]
            else:
                prev_100ms_angle = 0
                prev_100ms_time = 0

            # Previous angle at least 200ms earlier
            prev_200ms_threshold = frame_time - 0.20
            prev_200ms_candidates = np.where(steering_times <= prev_200ms_threshold)[0]
            if len(prev_200ms_candidates) > 0:
                prev_200ms_idx = prev_200ms_candidates[-1]
                prev_200ms_angle = steering_angles[prev_200ms_idx]
                prev_200ms_time = steering_times[prev_200ms_idx]
            else:
                prev_200ms_angle = 0
                prev_200ms_time = 0
            
            # Get current measurements
            data_point = {
                'frame_idx': frame_idx,
                'frame_time': frame_time,
                'speed': get_measurement_at_time(speed_times, speed_values, frame_time),
                'gyro_x': get_measurement_at_time(gyro_times, gyro_values, frame_time)[0],
                'gyro_y': get_measurement_at_time(gyro_times, gyro_values, frame_time)[1],
                'gyro_z': get_measurement_at_time(gyro_times, gyro_values, frame_time)[2],
                'accel_x': get_measurement_at_time(accel_times, accel_values, frame_time)[0],
                'accel_y': get_measurement_at_time(accel_times, accel_values, frame_time)[1],
                'accel_z': get_measurement_at_time(accel_times, accel_values, frame_time)[2],
                'velocity_x': frame_velocities[frame_idx][0],
                'velocity_y': frame_velocities[frame_idx][1],
                'velocity_z': frame_velocities[frame_idx][2],
                'current_steering': get_measurement_at_time(steering_times, steering_angles, frame_time),
                'future_steering': get_measurement_at_time(steering_times, steering_angles, future_time),
                'steering_angle_prev_100ms': prev_100ms_angle,
                'steering_time_prev_100ms': prev_100ms_time,
                'steering_angle_prev_200ms': prev_200ms_angle,
                'steering_time_prev_200ms': prev_200ms_time
            }
            synced_data.append(data_point)
    
    container.close()

    # Convert to DataFrame
    df = pd.DataFrame(synced_data)
    
    tqdm.write(f"\nProcessed {len(df)} frames from {segment}")
    tqdm.write(f"Original frames: {len(frame_times)}")
    tqdm.write(f"Sampling rate: {effective_fps:.2f} FPS")
    return df

def process_single_segment(segment):
    """
    Process a single segment - helper function for parallel processing
    """
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
        
        return True, segment
        
    except Exception as e:
        tqdm.write(f"Error processing {segment}: {str(e)}")
        return False, segment

if __name__ == "__main__":
    # Base paths
    data_dir = 'data'
    comma2k19_dir = os.path.join(data_dir, 'comma2k19')
    
    # Check both potential locations for extracted data
    extract_dir = os.path.join(data_dir, 'extracted')  # at same level as comma2k19
    alt_extract_dir = os.path.join(comma2k19_dir, 'extracted')  # inside comma2k19
    
    chunk_path = os.path.join(extract_dir, CHUNK_NAME)
    alt_chunk_path = os.path.join(alt_extract_dir, CHUNK_NAME)
    
    # Check if data exists in alternate location
    if os.path.exists(alt_chunk_path):
        print(f"Found data in alternate location: {alt_chunk_path}")
        chunk_path = alt_chunk_path
        extract_dir = alt_extract_dir
    
    zip_path = os.path.join(comma2k19_dir, f'{CHUNK_NAME}.zip')

    # Allow setting custom paths via environment variables
    custom_data_path = os.environ.get('COMMA2K19_DATA_PATH')
    if custom_data_path:
        print(f"Using custom data path: {custom_data_path}")
        zip_path = os.path.join(custom_data_path, f'{CHUNK_NAME}.zip')

    # Create necessary directories
    os.makedirs('data_synced', exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    # Check and extract the zip file if needed
    if not os.path.exists(chunk_path):
        print(f"{CHUNK_NAME} directory not found, checking for zip file...")
        success = extract_zip_if_needed(zip_path, extract_dir)
        if not success:
            print(f"Error: Could not find or extract the dataset. Please ensure {CHUNK_NAME}.zip is in the data/comma2k19 directory.")
            exit(1)

    # Find all segment directories
    segments = []
    for root, dirs, files in os.walk(chunk_path):
        if 'processed_log' in dirs:  # This indicates we're in a segment directory
            segments.append(root)  # Don't add trailing slash

    print(f"Found {len(segments)} segments to process")

    # Calculate optimal number of workers
    # Use min of (CPU count - 1) or 4 to avoid overwhelming the system
    max_workers = min(os.cpu_count() - 1 or 1, 4)
    print(f"Processing with {max_workers} workers")

    # Process segments in parallel
    successful_segments = 0
    failed_segments = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a progress bar for all segments
        futures = list(tqdm(
            executor.map(process_single_segment, segments),
            total=len(segments),
            desc="Processing segments",
            position=0,
            leave=True
        ))
        
        # Count successes and failures
        for success, segment in futures:
            if success:
                successful_segments += 1
            else:
                failed_segments.append(segment)

    # Print summary
    print("\nProcessing complete!")
    print(f"Successfully processed: {successful_segments}/{len(segments)} segments")
    if failed_segments:
        print(f"Failed segments ({len(failed_segments)}):")
        for segment in failed_segments:
            print(f"  - {segment}")
    print(f"Files saved in data_synced/")
