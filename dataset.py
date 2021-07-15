import json
import random
import torch
from torch.utils.data import Dataset, DataLoader


class PhysioBank:
    def __init__(
        self, path, train_batch_size=1, val_batch_size=1, test_batch_size=1,
        cuda=False, num_workers=1, train_split=0.7, val_split=0.15, mean=78.78, std=28.35
    ):
        """Initializes the dataset for loading."""

        self.path = path
        self.cuda = cuda
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.mean = mean
        self.std = std

        # Get data
        self._create_data(self._read_data())
    
    def _read_data(self):
        with open(self.path) as f:
            data = json.load(f)
        return data
    
    def _get_normalization(self, samples):
        self.transition, self.scale = {}, {}

        # IHR
        samples_ihr = [y for x in samples for y in x['ihr']]
        self.transition['ihr'] = min(samples_ihr)
        self.scale['ihr'] = max(samples_ihr) - self.transition['ihr']

        # Age
        samples_age = [x['age'] for x in samples]
        self.transition['age'] = min(samples_age)
        self.scale['age'] = max(samples_age) - self.transition['age']

    def _create_data(self, samples):
        random.shuffle(samples)

        # Calculate number of samples in each set
        train_limit = int(len(samples) * self.train_split)
        val_limit = int(len(samples) * self.val_split)

        # Distribute data
        self._get_normalization(samples[:train_limit])
        self.train_data = PhysioBankDataset(samples[:train_limit], self.transition, self.scale)
        self.val_data = PhysioBankDataset(samples[train_limit:train_limit + val_limit], self.transition, self.scale)
        self.test_data = PhysioBankDataset(samples[train_limit + val_limit:], self.transition, self.scale)

    def loader(self, type='train', shuffle=True):
        loader_args = { 'shuffle': shuffle }

        # If GPU exists
        if self.cuda:
            loader_args['num_workers'] = self.num_workers
            loader_args['pin_memory'] = True

        if type == 'train':
            loader_args['batch_size'] = self.train_batch_size
            return DataLoader(self.train_data, **loader_args)
        elif type == 'val':
            loader_args['batch_size'] = self.val_batch_size
            return DataLoader(self.val_data, **loader_args)
        else:
            loader_args['batch_size'] = self.test_batch_size
            return DataLoader(self.test_data, **loader_args)


class PhysioBankDataset(Dataset):
    def __init__(self, samples, transition, scale):
        """Initializes the dataset for loading."""
        super(PhysioBankDataset, self).__init__()
        self.samples = samples
        self.transition = transition
        self.scale = scale

    def __len__(self):
        """Returns length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        return (
            (
                (torch.FloatTensor(sample['ihr']) - self.transition['ihr']) / self.scale['ihr'],
                torch.FloatTensor([
                    sample['gender'],
                    (sample['age'] - self.transition['age']) / self.scale['age']
                ])
            ),
            torch.FloatTensor([sample['hypertensive']])
        )
