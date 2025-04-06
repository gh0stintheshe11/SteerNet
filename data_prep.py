import av
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm  # for progress bar

def sync_video_and_steering_with_more_state_data(segment):
    """
    Sync current video frame and sensor data with future steering angle (t+200ms)
    """
    base_path = segment + 'processed_log/'
    
    # Load all sensor data
    steering_times = np.load(base_path + 'CAN/steering_angle/t')
    steering_angles = np.load(base_path + 'CAN/steering_angle/value')
    speed_times = np.load(base_path + 'CAN/speed/t')
    speed_values = np.load(base_path + 'CAN/speed/value')
    gyro_times = np.load(base_path + 'IMU/gyro/t')
    gyro_values = np.load(base_path + 'IMU/gyro/value')
    accel_times = np.load(base_path + 'IMU/accelerometer/t')
    accel_values = np.load(base_path + 'IMU/accelerometer/value')
    frame_times = np.load(segment + 'global_pose/frame_times')
    frame_velocities = np.load(segment + 'global_pose/frame_velocities')

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
    # Base path for Chunk_1
    chunk_path = 'data/comma2k19/extracted/Chunk_1'

    # Create data_synced directory in root if it doesn't exist
    os.makedirs('data_synced', exist_ok=True)

    # Find all segment directories
    segments = []
    for root, dirs, files in os.walk(chunk_path):
        if 'processed_log' in dirs:  # This indicates we're in a segment directory
            segments.append(root + '/')

    print(f"Found {len(segments)} segments to process")

    # Process each segment with progress bar
    for segment in tqdm(segments, position=0, leave=True):
        try:
            # Extract identifier from path (keep original path for processing)
            path_parts = segment.split('/')
            parent_dir = path_parts[-3]  # Gets b0c9d2329ad1606b|2018-07-27--06-03-57
            seg_num = path_parts[-2]     # Gets the segment number (3)
            
            # Get only the timestamp part after the |
            timestamp = parent_dir.split('|')[1]  # Gets 2018-07-27--06-03-57
            
            # Create output filename
            output_file = os.path.join('data_synced', f'{parent_dir}_{seg_num}.csv')
            
            tqdm.write(f"Processing: {segment}")
            
            # Process the segment using your existing function (using original path with |)
            df = sync_video_and_steering_with_more_state_data(segment)
            
            # Save to CSV with safe filename
            df.to_csv(output_file, index=False)
            tqdm.write(f"Saved {len(df)} frames to {output_file}")
            
        except Exception as e:
            tqdm.write(f"Error processing segment {segment}: {str(e)}")
            continue

    print("\nProcessing complete!")
    print(f"Files saved in data_synced/")
