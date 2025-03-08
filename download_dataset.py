import libtorrent as lt
import time
import os
import requests
import zipfile

# Define paths
SAVE_DIR = "data/comma2k19"
TORRENT_PATH = os.path.join(SAVE_DIR, "comma2k19.torrent")
TORRENT_URL = "https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent"
EXTRACT_DIR = os.path.join(SAVE_DIR, "extracted")

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

def list_available_chunks(info):
    """List all available chunks in the torrent"""
    print("\nAvailable chunks:")
    chunks = []
    for idx in range(info.num_files()):
        file_path = info.files().file_path(idx)
        if file_path.endswith('.zip'):
            chunks.append(file_path)
            print(f"{len(chunks)}. {file_path}")
    return chunks

def get_user_choice(chunks):
    """Get user choice for download mode and chunk"""
    while True:
        print("\nDownload options:")
        print("1. Download all chunks")
        print("2. Download specific chunk")
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            return "all", None
        elif choice == "2":
            while True:
                try:
                    chunk_num = int(input(f"Enter chunk number (1-{len(chunks)}): "))
                    if 1 <= chunk_num <= len(chunks):
                        return "specific", chunks[chunk_num - 1]
                    else:
                        print("Invalid chunk number. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            print("Invalid choice. Please enter 1 or 2.")

# Step 1: Download the Torrent File
if not os.path.exists(TORRENT_PATH):
    print("Downloading the torrent file...")
    response = requests.get(TORRENT_URL)
    if response.status_code == 200:
        with open(TORRENT_PATH, "wb") as file:
            file.write(response.content)
        print(f"Torrent file saved: {TORRENT_PATH}")
    else:
        print("Failed to download the torrent file. Check the URL.")
        exit()

# Step 2: Download chunks using libtorrent
print("Loading torrent session...")
try:
    # Configure session
    session = lt.session()
    
    # Configure settings using the new API
    settings = {
        'alert_mask': lt.alert.category_t.all_categories,
        'enable_dht': True,
        'announce_to_all_tiers': True,
        'enable_outgoing_utp': True,
        'enable_incoming_utp': True,
        'download_rate_limit': 0,
        'upload_rate_limit': 0,
        'listen_interfaces': '0.0.0.0:6881',  # Replace listen_on with listen_interfaces
    }
    session.apply_settings(settings)

    info = lt.torrent_info(TORRENT_PATH)
    params = {
        "ti": info,
        "save_path": SAVE_DIR,
        "flags": lt.torrent_flags.sequential_download
    }
    handle = session.add_torrent(params)

    # Get available chunks and user choice
    chunks = list_available_chunks(info)
    download_mode, selected_chunk = get_user_choice(chunks)
    downloaded_files = []

    # Set file priorities based on user choice
    for idx in range(info.num_files()):
        file_path = info.files().file_path(idx)
        if download_mode == "all":
            if file_path.endswith('.zip'):
                handle.file_priority(idx, 1)
                downloaded_files.append(os.path.join(SAVE_DIR, file_path))
                print(f"Download: {file_path}")
        else:  # specific chunk
            if selected_chunk == file_path:
                handle.file_priority(idx, 1)
                downloaded_files.append(os.path.join(SAVE_DIR, file_path))
                print(f"Download: {file_path}")
            else:
                handle.file_priority(idx, 0)

    if not downloaded_files:
        print("Error: No files selected for download.")
        exit()

    print("\nStarting download...")

    # Monitor the download progress and exit properly
    while (handle.status().state != lt.torrent_status.seeding):
        status = handle.status()
        progress = status.progress * 100
        speed = status.download_rate / 1024  # Convert speed to KB/s
        
        print(f"\rProgress: {progress:.2f}% | Speed: {speed:.2f} KB/s", end="", flush=True)

        if progress >= 100.0:
            break

        time.sleep(1)

    print("\nDownload complete!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit(1)
finally:
    # Remove the torrent from the session to clean up
    if 'handle' in locals():
        session.remove_torrent(handle)
    print("Torrent session closed.")

# Flush disk cache to ensure the file is fully written
time.sleep(2)

# Process each downloaded file
for downloaded_file in downloaded_files:
    print(f"\nProcessing: {downloaded_file}")
    
    # Verify file exists
    if os.path.exists(downloaded_file):
        print("Download verified successfully!")
        
        # Extract the zip file
        print(f"\nExtracting {downloaded_file} to {EXTRACT_DIR}...")
        try:
            with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
                # Get total size of files to extract
                total_size = sum(file.file_size for file in zip_ref.filelist)
                extracted_size = 0
                
                # Extract each file with progress
                for file in zip_ref.filelist:
                    zip_ref.extract(file, EXTRACT_DIR)
                    extracted_size += file.file_size
                    progress = (extracted_size / total_size) * 100
                    print(f"\rExtraction progress: {progress:.1f}%", end="", flush=True)
            
            print("\nExtraction completed successfully!")
            
            # Optionally remove the zip file to save space
            #os.remove(downloaded_file)
            #print("Removed zip file to save space")
            
        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file")
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
    else:
        print(f"Warning: File not found: {downloaded_file}")