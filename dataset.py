import json
import torch
import h5py
from torch.utils.data import Dataset, DataLoader


MIN_MAX_VALUES = {
    'ihr': [0.08, 256],
    'age': [21, 92],
    'bsa': [1.47, 2.3],
    'bmi': [18.37, 40.04],
    'sbp': [95, 200.0],
    'dbp': [30, 100.0]
}


class PhysioBankDataset(Dataset):

    def __init__(self, data_file, user_info_file):
        super().__init__()
        self.data = h5py.File(data_file, 'r')
        self.data_keys = list(self.data.keys())
        with open(user_info_file) as f:
            self.user_info = {x['record']: x for x in json.load(f)}

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, index):
        ihr = self.data[self.data_keys[index]][:]
        record = self.user_info[self.data[self.data_keys[index]].attrs['record']]

        return (
            (torch.FloatTensor(ihr) - MIN_MAX_VALUES['ihr'][0]) / (MIN_MAX_VALUES['ihr'][1] - MIN_MAX_VALUES['ihr'][0]),
            torch.FloatTensor([
                record['gender'],
                record['smoker'],
                record['vascular_event'],
                (record['age'] - MIN_MAX_VALUES['age'][0]) / (MIN_MAX_VALUES['age'][1] - MIN_MAX_VALUES['age'][0]),
                (record['bsa'] - MIN_MAX_VALUES['bsa'][0]) / (MIN_MAX_VALUES['bsa'][1] - MIN_MAX_VALUES['bsa'][0]),
                (record['bmi'] - MIN_MAX_VALUES['bmi'][0]) / (MIN_MAX_VALUES['bmi'][1] - MIN_MAX_VALUES['bmi'][0]),
                (record['sbp'] - MIN_MAX_VALUES['sbp'][0]) / (MIN_MAX_VALUES['sbp'][1] - MIN_MAX_VALUES['sbp'][0]),
                (record['dbp'] - MIN_MAX_VALUES['dbp'][0]) / (MIN_MAX_VALUES['dbp'][1] - MIN_MAX_VALUES['dbp'][0])
            ]),
            torch.FloatTensor([record['hypertensive']])
        )

    def loader(self, batch_size, num_workers, shuffle=True):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
