# SteerNet

This project evaluates four architectures for short-term steering prediction: a baseline CNN inspired by NVIDIA’s PilotNet, an enhanced CNN with more temporal steering history, a MobileNetV2-based CNN, and an RNN with ConvLSTM. All models performed similarly, with RNN demonstrating the highest accuracy (MAE 0.57°).

> [!IMPORTANT]  
> The report is written based on the result of the [previous version](https://github.com/gh0stintheshe11/SteerNet/tree/52d8f3934dbdb8a67bdfaafc7b5af43fbc6916fe) of the project. Some of the results does not match exactly with the results in the report. However, the overall conclusions remain the same. 

## Table of Contents

- [SteerNet](#steernet)
  - [Table of Contents](#table-of-contents)
  - [Project Sturcture](#project-sturcture)
  - [Downlaod the Dataset](#downlaod-the-dataset)

## Project Sturcture

```bash
SteerNet/
├── data/ # original dataset
├── data_synced/ # dataset after syncing the sensor data with video frames
├── model_checkpoints/ # all model trained in this project will be saved here as .pth files
├── deprecated/ # deprecated files
├── aria2c.exe # aria2c executable used for downloading the dataset on windows
├── data_prep_v2.py # data preparation script
├── download_dataset.py # dataset download script
├── requirements.txt # all dependencies for this project
├── README.md # this file
├── v4_CNN_framerate_test.ipynb # framerate test for v4_CNN
├── v4_CNN_MobileNetV2.ipynb
├── v4_CNN.ipynb
├── v4_CNN2.ipynb
├── v4_RNN.ipynb
├── v4_RNN_GRU.ipynb
```


## Downlaod the Dataset

- For Windows:

  1. Open a terminal/command prompt in the root folder of this project

  2. Run this command to download the full dataset (94.62GB):
  ```bash
  aria2c.exe --dir="data/" --seed-time=0 --continue=true "https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent"
  ```

  To download selected chunks (about 9GB each), add `--select-file=1` (for first chunk), `--select-file=2` (for second chunk), etc.:
  ```bash
  aria2c.exe --dir="data/" --select-file=1 --seed-time=0 --continue=true "https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent"
  ```

  > [!WARNING] 
  > Download speed will vary based on your internet connection and available seeders. Thus, it is best to download the few chunks than the full dataset.


  - NONE IMPRTANT PART (this is just what nerds do)

  ```bash
  .\aria2c.exe --dir="E:\SteerNet\data" --select-file=2 --seed-time=0 --file-allocation=falloc --max-connection-per-server=16 --split=16 --min-split-size=1M --max-concurrent-downloads=64 --max-overall-download-limit=0 --max-download-limit=0 --disable-ipv6=true --bt-max-peers=500 --bt-request-peer-speed-limit=0 --max-overall-upload-limit=1K --async-dns=true --summary-interval=1 --disk-cache=128M --enable-mmap=true --optimize-concurrent-downloads=true --bt-tracker="http://academictorrents.com:6969/announce,udp://tracker.opentrackr.org:1337/announce,udp://9.rarbg.com:2810/announce,udp://tracker.openbittorrent.com:6969/announce,udp://tracker.torrent.eu.org:451/announce,udp://exodus.desync.com:6969/announce,udp://tracker.torrent.eu.org:451/announce,udp://tracker.moeking.me:6969/announce,udp://tracker.opentrackr.org:1337/announce,udp://open.stealth.si:80/announce,udp://movies.zsw.ca:6969/announce" --bt-enable-lpd=true --enable-peer-exchange=true --follow-torrent=mem --continue=true --console-log-level=notice "https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent"
  ```
<br>

- For Mac:
  directly run the [download_dataset.py](download_dataset.py) to download the dataset



