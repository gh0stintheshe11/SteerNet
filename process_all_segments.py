import av
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def sync_video_and_steering(segment):
    """
    Sync video frames with steering angle data
    """
    # Load steering data
    steering_times = np.load(segment + 'processed_log/CAN/steering_angle/t')
    steering_angles = np.load(segment + 'processed_log/CAN/steering_angle/value')
    
    # Load frame times
    frame_times = np.load(segment + 'global_pose/frame_times')
    
    print(f"Number of frames: {len(frame_times)}")
    print(f"Number of steering measurements: {len(steering_times)}")
    print(f"\nFrame time range: {frame_times[0]:.2f} to {frame_times[-1]:.2f}")
    print(f"Steering time range: {steering_times[0]:.2f} to {steering_times[-1]:.2f}")
    
    # Create a list to store synchronized data
    synced_data = []
    
    # For each frame time, find the closest steering angle measurement
    for frame_idx, frame_time in enumerate(frame_times):
        # Find the closest steering measurement
        closest_idx = np.argmin(np.abs(steering_times - frame_time))
        
        synced_data.append({
            'frame_idx': frame_idx,
            'frame_time': frame_time,
            'steering_time': steering_times[closest_idx],
            'steering_angle': steering_angles[closest_idx],
            'time_diff': abs(frame_time - steering_times[closest_idx])
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(synced_data)
    
    print("\nSync Statistics:")
    print(f"Average time difference: {df['time_diff'].mean()*1000:.2f} ms")
    print(f"Max time difference: {df['time_diff'].max()*1000:.2f} ms")
    
    return df


def process_all_segments(base_path='data/comma2k19/extracted/Chunk_1/'):
    """
    Process all segments in the dataset by traversing through all subdirectories.
    
    Args:
        base_path (str): Path to the Chunk_1 directory
        
    Returns:
        dict: Dictionary containing results for each processed segment
    """
    # Create output directory if it doesn't exist
    output_dir = 'data/generated'
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results
    all_results = {}
    
    # Get all immediate subdirectories in the base path (the b0c9d... folders)
    main_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Process each main directory
    for main_dir in tqdm(main_dirs, desc="Processing main directories"):
        main_path = os.path.join(base_path, main_dir)
        
        # Get all segment subdirectories (the numbered folders)
        segment_dirs = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
        
        # Process each segment
        for segment in segment_dirs:
            segment_path = os.path.join(main_path, segment, '')  # Empty string at end adds trailing slash
            
            try:
                # Check if required files exist
                steering_time_path = os.path.join(segment_path, 'processed_log/CAN/steering_angle/t')
                steering_angle_path = os.path.join(segment_path, 'processed_log/CAN/steering_angle/value')
                frame_times_path = os.path.join(segment_path, 'global_pose/frame_times')
                
                if not all(os.path.exists(p) for p in [steering_time_path, steering_angle_path, frame_times_path]):
                    print(f"Skipping {segment_path} - missing required files")
                    continue
                
                # Process the segment
                print(f"\nProcessing: {segment_path}")
                df = sync_video_and_steering(segment_path)
                
                # Store results
                all_results[segment_path] = {
                    'dataframe': df,
                    'avg_time_diff': df['time_diff'].mean(),
                    'max_time_diff': df['time_diff'].max(),
                    'n_frames': len(df)
                }
                
                clean_main_dir = main_dir.replace("|", "")
                csv_filename = os.path.join(output_dir, f"synced_steer_data_{clean_main_dir}_SEG{segment}.csv")
                df.to_csv(csv_filename, index=False)
                print(f"Saved segment data to: {csv_filename}")
                
            except Exception as e:
                print(f"Error processing {segment_path}: {str(e)}")
                continue
    
    # Create a summary DataFrame
    summary_data = []
    for segment_path, result in all_results.items():
        summary_data.append({
            'segment_path': segment_path,
            'n_frames': result['n_frames'],
            'avg_time_diff_ms': result['avg_time_diff'] * 1000,
            'max_time_diff_ms': result['max_time_diff'] * 1000
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'sync_steer_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nProcessing complete. Summary saved to: {summary_path}")
    
    return all_results

def process_all_segments_short(base_path='data/comma2k19/extracted/Chunk_1/'):
    """
    Process all segments in the dataset and downsample to 1 Hz.
    
    Args:
        base_path (str): Path to the Chunk_1 directory
        
    Returns:
        dict: Dictionary containing results for each processed segment
    """
    # Create output directory if it doesn't exist
    output_dir = 'data/generated'
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results
    all_results = {}
    
    # Get all immediate subdirectories in the base path (the b0c9d... folders)
    main_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Process each main directory
    for main_dir in tqdm(main_dirs, desc="Processing main directories"):
        main_path = os.path.join(base_path, main_dir)
        
        # Get all segment subdirectories (the numbered folders)
        segment_dirs = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
        
        # Process each segment
        for segment in segment_dirs:
            segment_path = os.path.join(main_path, segment, '')  # Empty string at end adds trailing slash
            
            try:
                # Check if required files exist
                steering_time_path = os.path.join(segment_path, 'processed_log/CAN/steering_angle/t')
                steering_angle_path = os.path.join(segment_path, 'processed_log/CAN/steering_angle/value')
                frame_times_path = os.path.join(segment_path, 'global_pose/frame_times')
                
                if not all(os.path.exists(p) for p in [steering_time_path, steering_angle_path, frame_times_path]):
                    print(f"Skipping {segment_path} - missing required files")
                    continue
                
                # Process the segment
                print(f"\nProcessing: {segment_path}")
                df = sync_video_and_steering(segment_path)
                
                # Downsample to 1 Hz by rounding frame_time to nearest second and taking first occurrence
                df['frame_time_rounded'] = np.floor(df['frame_time'])
                df_short = df.groupby('frame_time_rounded').first().reset_index()
                
                # Drop the rounded time column as it's no longer needed
                df_short = df_short.drop('frame_time_rounded', axis=1)
                
                # Store results
                all_results[segment_path] = {
                    'dataframe': df_short,
                    'avg_time_diff': df_short['time_diff'].mean(),
                    'max_time_diff': df_short['time_diff'].max(),
                    'n_frames': len(df_short)
                }
                
                clean_main_dir = main_dir.replace("|", "")
                csv_filename = os.path.join(output_dir, f"synced_steer_data_short_{clean_main_dir}_SEG{segment}.csv")
                df_short.to_csv(csv_filename, index=False)
                print(f"Saved downsampled segment data to: {csv_filename}")
                
                # Print downsampling statistics
                reduction_ratio = (len(df) - len(df_short)) / len(df) * 100
                print(f"Downsampling reduced data by {reduction_ratio:.1f}% ({len(df)} → {len(df_short)} points)")
                
            except Exception as e:
                print(f"Error processing {segment_path}: {str(e)}")
                continue
    
    # Create a summary DataFrame
    summary_data = []
    for segment_path, result in all_results.items():
        summary_data.append({
            'segment_path': segment_path,
            'n_frames': result['n_frames'],
            'avg_time_diff_ms': result['avg_time_diff'] * 1000,
            'max_time_diff_ms': result['max_time_diff'] * 1000
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'sync_steer_summary_short.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nProcessing complete. Summary saved to: {summary_path}")
    
    return all_results

if __name__ == "__main__":
    # Process all segments with both full and downsampled versions
    print("Processing full dataset...")
    results_full = process_all_segments()
    
    print("\nProcessing downsampled dataset...")
    results_short = process_all_segments_short()
    
    # Print overall statistics for both versions
    print("\nOverall Statistics (Full Dataset):")
    summary_path_full = os.path.join('data/generated', 'sync_steer_summary.csv')
    summary_df_full = pd.read_csv(summary_path_full)
    print(f"Total segments processed: {len(summary_df_full)}")
    print(f"Average time difference across all segments: {summary_df_full['avg_time_diff_ms'].mean():.2f} ms")
    print(f"Maximum time difference across all segments: {summary_df_full['max_time_diff_ms'].max():.2f} ms")
    print(f"Total frames processed: {summary_df_full['n_frames'].sum()}")
    
    print("\nOverall Statistics (Downsampled Dataset):")
    summary_path_short = os.path.join('data/generated', 'sync_steer_summary_short.csv')
    summary_df_short = pd.read_csv(summary_path_short)
    print(f"Total segments processed: {len(summary_df_short)}")
    print(f"Average time difference across all segments: {summary_df_short['avg_time_diff_ms'].mean():.2f} ms")
    print(f"Maximum time difference across all segments: {summary_df_short['max_time_diff_ms'].max():.2f} ms")
    print(f"Total frames processed: {summary_df_short['n_frames'].sum()}") 