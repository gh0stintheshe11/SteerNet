import av
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm  # for progress bar

def sync_video_and_steering_with_more_state_data(segment):
    """
    Sync video frames with comprehensive car state data at current time, 100ms ago, and 200ms ago
    """
    base_path = segment + 'processed_log/'
    
    # Load all sensor data
    # CAN data
    steering_times = np.load(base_path + 'CAN/steering_angle/t')
    steering_angles = np.load(base_path + 'CAN/steering_angle/value')
    speed_times = np.load(base_path + 'CAN/speed/t')
    speed_values = np.load(base_path + 'CAN/speed/value')
    wheel_speed_times = np.load(base_path + 'CAN/wheel_speed/t')
    wheel_speed_values = np.load(base_path + 'CAN/wheel_speed/value')  # Individual wheel speeds
    
    # IMU data
    gyro_times = np.load(base_path + 'IMU/gyro/t')
    gyro_values = np.load(base_path + 'IMU/gyro/value')  # [x,y,z] rotation rates
    accel_times = np.load(base_path + 'IMU/accelerometer/t')
    accel_values = np.load(base_path + 'IMU/accelerometer/value')  # [x,y,z] acceleration
    
    # Frame data
    frame_times = np.load(segment + 'global_pose/frame_times')
    frame_velocities = np.load(segment + 'global_pose/frame_velocities')

    synced_data = []
    
    for frame_idx, frame_time in enumerate(frame_times):
        # Define the three time points we want data for
        current_time = frame_time
        time_100ms_ago = frame_time - 0.10
        time_200ms_ago = frame_time - 0.20
        
        # Function to get closest measurement for any sensor
        def get_measurement_at_time(times, values, target_time):
            idx = np.argmin(np.abs(times - target_time))
            return values[idx]
        
        # Get all measurements for each time point
        data_point = {
            'frame_idx': frame_idx,
            'frame_time': frame_time,
            
            # Current time measurements
            # Steering and speed
            'steering_angle': get_measurement_at_time(steering_times, steering_angles, current_time),
            'speed': get_measurement_at_time(speed_times, speed_values, current_time),
            
            # Wheel speeds (individual wheels)
            'wheel_speeds': get_measurement_at_time(wheel_speed_times, wheel_speed_values, current_time),
            
            # IMU data
            'gyro_x': get_measurement_at_time(gyro_times, gyro_values, current_time)[0],
            'gyro_y': get_measurement_at_time(gyro_times, gyro_values, current_time)[1],
            'gyro_z': get_measurement_at_time(gyro_times, gyro_values, current_time)[2],
            'accel_x': get_measurement_at_time(accel_times, accel_values, current_time)[0],
            'accel_y': get_measurement_at_time(accel_times, accel_values, current_time)[1],
            'accel_z': get_measurement_at_time(accel_times, accel_values, current_time)[2],
            
            # Vehicle velocities
            'velocity_x': frame_velocities[frame_idx][0],
            'velocity_y': frame_velocities[frame_idx][1],
            'velocity_z': frame_velocities[frame_idx][2],
            
            # 100ms ago measurements
            'steering_angle_100ms': get_measurement_at_time(steering_times, steering_angles, time_100ms_ago),
            'speed_100ms': get_measurement_at_time(speed_times, speed_values, time_100ms_ago),
            'wheel_speeds_100ms': get_measurement_at_time(wheel_speed_times, wheel_speed_values, time_100ms_ago),
            'gyro_x_100ms': get_measurement_at_time(gyro_times, gyro_values, time_100ms_ago)[0],
            'gyro_y_100ms': get_measurement_at_time(gyro_times, gyro_values, time_100ms_ago)[1],
            'gyro_z_100ms': get_measurement_at_time(gyro_times, gyro_values, time_100ms_ago)[2],
            'accel_x_100ms': get_measurement_at_time(accel_times, accel_values, time_100ms_ago)[0],
            'accel_y_100ms': get_measurement_at_time(accel_times, accel_values, time_100ms_ago)[1],
            'accel_z_100ms': get_measurement_at_time(accel_times, accel_values, time_100ms_ago)[2],
            
            # 200ms ago measurements
            'steering_angle_200ms': get_measurement_at_time(steering_times, steering_angles, time_200ms_ago),
            'speed_200ms': get_measurement_at_time(speed_times, speed_values, time_200ms_ago),
            'wheel_speeds_200ms': get_measurement_at_time(wheel_speed_times, wheel_speed_values, time_200ms_ago),
            'gyro_x_200ms': get_measurement_at_time(gyro_times, gyro_values, time_200ms_ago)[0],
            'gyro_y_200ms': get_measurement_at_time(gyro_times, gyro_values, time_200ms_ago)[1],
            'gyro_z_200ms': get_measurement_at_time(gyro_times, gyro_values, time_200ms_ago)[2],
            'accel_x_200ms': get_measurement_at_time(accel_times, accel_values, time_200ms_ago)[0],
            'accel_y_200ms': get_measurement_at_time(accel_times, accel_values, time_200ms_ago)[1],
            'accel_z_200ms': get_measurement_at_time(accel_times, accel_values, time_200ms_ago)[2],
        }
        
        synced_data.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame(synced_data)
    
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
            
            # Create safe filename by replacing special chars with underscore
            safe_name = f"{timestamp.replace('-', '_')}_{seg_num}"
            
            # Create output filename
            output_file = os.path.join('data_synced', f'{safe_name}.csv')
            
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
