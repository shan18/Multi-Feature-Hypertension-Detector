# PhysioNet Hypertension Detector

## Installation

Before starting setting up the code. The WFDB Software Package needs to be installed on your system. Go to the [WFDB](https://archive.physionet.org/physiotools/wfdb.shtml) website to see the installation instructions.

## Dataset

Two datasets were used from the [PhysioNet](https://physionet.org/) database

- Smart Health for Assessing the Risk of Events via ECG Database 1.0.0 (shareedb)
- Normal Sinus Rhythm RR Interval Database (nsr2db)

To setup the dataset, follow the steps below

- Download the [shareedb](https://physionet.org/content/shareedb/1.0.0/) and the [nsr2db](https://physionet.org/content/nsr2db/1.0.0/) datasets.
- Extract the downloaded datasets and put the contents of the datasets inside the directories `data/files/shareedb` and `data/files/nsr2db` respectively.
- Run the command  
   `$ mv data/files/shareedb/info.txt data/files/shareedb_info.txt`
- Now extract the required data from the datasets  
   `$ python3 data/extract_records.py`
- Parse the extracted data to create the final dataset  
   `$ python3 data/parse_data.py`

After following the steps a file called `physiobank_dataset.json` will be created containing the final data.
