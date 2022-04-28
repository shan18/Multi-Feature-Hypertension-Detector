# Multi-Feature Hypertension Detector

Hypertension is a serious underlying condition that has been found to cause a number of diseases if left undetected. This project uses the Instantaneous Heart Rate (IHR) records of users along with their other biometric information such as gender and weight to predict if the user is suffering from hypertension.

This project proposes two models for this task

### Bi-LSTM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OJZMqy3eDAE1ToftCtRn4mEn_8N4t271?usp=sharing)

The IHR records are fed as input to a two-layer stacked bidirectional LSTM network whereas the biometric information is first passed through a fully connected layer and sent as the initial hidden state to both the LSTM layers.

### CNN-GRU

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Z1q6QcGiNbNq86z36J1q-PRlhwfne8Ne?usp=sharing)

The IHR records are first fed to a series of Conv-1D blocks followed by a max pooling layer. The extracted features are then sent to a two-layer stacked GRU network as input whereas the biometric information is first passed through a fully connected layer and sent as the initial hidden state to both the GRU layers.

## Installation

- Before starting setting up the code. The WFDB Software Package needs to be installed on your system. Go to the [WFDB](https://archive.physionet.org/physiotools/wfdb.shtml) website to see the installation instructions.
- Run the command below to install the required python dependencies  
   `$ pip install -r requirements.txt`

## Dataset

Three datasets were used from the [PhysioNet](https://physionet.org/) database

- Smart Health for Assessing the Risk of Events via ECG Database 1.0.0 (shareedb)
- MIT-BIH Normal Sinus Rhythm Database (nsrdb)
- Normal Sinus Rhythm RR Interval Database (nsr2db)

To setup the dataset, follow the steps below

- Download the [shareedb](https://physionet.org/content/shareedb/1.0.0/), [nsrdb](https://www.physionet.org/content/nsrdb/1.0.0/), and the [nsr2db](https://physionet.org/content/nsr2db/1.0.0/) datasets.
- Extract the downloaded datasets and put the contents of the datasets inside the directories `data/files/shareedb`, `data/files/nsrdb`, and `data/files/nsr2db` respectively.
- Now extract the required data from the datasets  
   `$ python3 data/extract_records.py`
- Parse the extracted data to create the final dataset  
   `$ python3 data/parse_data.py`

After following the steps a file called `physiobank_dataset.json` will be created containing the final data.

## Training and Testing

The code for training and testing the model is given in the form of both **python scripts** and **jupyter notebooks**.

### Python Scripts

To run the code via python scripts, check the `train.py` file.

### Jupyter Notebooks

To run the code via notebooks, checkout the `notebooks` directory or see the links to the _Google Colab_ files given above.

## References

The techniques used in this repository for detecting hypertension using a **multi-feature** model are an extension of the methods referenced in the publication ["Instantaneous Heart Rate-based Automated Monitoring of Hypertension using Machine Learning"](https://ieeexplore.ieee.org/document/9397126).
