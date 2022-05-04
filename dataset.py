import json
import random
import numpy as np
import torch
import pandas as pd
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

        self.data_file = data_file
        with open(user_info_file) as f:
            self.user_info = {x['record']: x for x in json.load(f)}

        # Index samples
        self.samples = [
            (user_val['record'], num_seq)
            for user_val in self.user_info.values()
            for num_seq in range(user_val['num_seq'])
        ]
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _normalize(self, data, min_max_attr):
        return (data - MIN_MAX_VALUES[min_max_attr][0]) / (MIN_MAX_VALUES[min_max_attr][1] - MIN_MAX_VALUES[min_max_attr][0])

    def __getitem__(self, index):
        sample = self.samples[index]
        record = self.user_info[sample[0]]
        ihr = np.array(pd.read_hdf(self.data_file, key=sample[0], start=sample[1], stop=sample[1]+1))[0]

        return (
            self._normalize(torch.FloatTensor(ihr), 'ihr'),
            torch.FloatTensor([
                record['gender'],
                record['smoker'],
                record['vascular_event'],
                self._normalize(record['age'], 'age'),
                self._normalize(record['bsa'], 'bsa'),
                self._normalize(record['bmi'], 'bmi'),
                self._normalize(record['sbp'], 'sbp'),
                self._normalize(record['dbp'], 'dbp'),
            ]),
            torch.FloatTensor([record['hypertensive']])
        )

    def loader(self, batch_size, num_workers, shuffle=True):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def create_dataset(data_file, user_info_file, batch_size, num_workers):
    dataset = PhysioBankDataset(data_file, user_info_file)
    return dataset, dataset.loader(batch_size, num_workers)
