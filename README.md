# SteerNet

## downlaod the dataset

- For Windows:


    1. Open a terminal/command prompt in the root folder of this project

    2. Run this command to download the full dataset (94.62GB):
    ```bash
    aria2c.exe --dir="data/comma2k19" --seed-time=0 --continue=true "https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent"
    ```

    To download selected chunks (about 9GB each), add `--select-file=1` (for first chunk), `--select-file=2` (for second chunk), etc.:
    ```bash
    aria2c.exe --dir="data/comma2k19" --select-file=1 --seed-time=0 --continue=true "https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent"
    ```

    > [!TIP] 
    > The download can be paused (Ctrl+C) and resumed by running the same command again.

    > [!WARNING] 
    > Download speed will vary based on your internet connection and available seeders. Thus, it is best to download the few chunks than the full dataset.


    - NONE IMPRTANT PART (this is just what nerds do)

    ```bash
    .\aria2c.exe --dir="E:\SteerNet\data\comma2k19" --select-file=2 --seed-time=0 --file-allocation=falloc --max-connection-per-server=16 --split=16 --min-split-size=1M --max-concurrent-downloads=64 --max-overall-download-limit=0 --max-download-limit=0 --disable-ipv6=true --bt-max-peers=500 --bt-request-peer-speed-limit=0 --max-overall-upload-limit=1K --async-dns=true --summary-interval=1 --disk-cache=128M --enable-mmap=true --optimize-concurrent-downloads=true --bt-tracker="http://academictorrents.com:6969/announce,udp://tracker.opentrackr.org:1337/announce,udp://9.rarbg.com:2810/announce,udp://tracker.openbittorrent.com:6969/announce,udp://tracker.torrent.eu.org:451/announce,udp://exodus.desync.com:6969/announce,udp://tracker.torrent.eu.org:451/announce,udp://tracker.moeking.me:6969/announce,udp://tracker.opentrackr.org:1337/announce,udp://open.stealth.si:80/announce,udp://movies.zsw.ca:6969/announce" --bt-enable-lpd=true --enable-peer-exchange=true --follow-torrent=mem --continue=true --console-log-level=notice "https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent"
    ```

- For Mac:
  directly run the [download_dataset.py](download_dataset.py) to download the dataset
