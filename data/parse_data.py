import os
import json
import argparse
import h5py
import random
import numpy as np


def read_file(path):
    with open(path) as f:
        if path.endswith('.json'):
            return json.load(f)
        elif path.endswith('.txt'):
            return [float(line.strip()) for line in f]


def split_data(dataset_file, split_ratio):
    print('Splitting data...')
    dataset = h5py.File(dataset_file, 'r')
    idxs = list(dataset.keys())
    random.shuffle(idxs)

    # Create train dataset
    print('Creating train dataset...')
    train_split_size = int(len(dataset) * split_ratio[0])
    train_dataset = h5py.File(dataset_file.replace('.hdf5', '_train.hdf5'), 'w')
    for idx in idxs[:train_split_size]:
        sub_data = train_dataset.create_dataset(idx, data=dataset[idx])
        sub_data.attrs['record'] = dataset[idx].attrs['record']
    train_dataset.close()

    # Create validation dataset
    print('Creating validation dataset...')
    val_split_size = int(len(dataset) * split_ratio[1])
    val_dataset = h5py.File(dataset_file.replace('.hdf5', '_val.hdf5'), 'w')
    for idx in idxs[train_split_size:train_split_size + val_split_size]:
        sub_data = val_dataset.create_dataset(idx, data=dataset[idx])
        sub_data.attrs['record'] = dataset[idx].attrs['record']
    val_dataset.close()

    # Create test dataset
    print('Creating test dataset...')
    test_dataset = h5py.File(dataset_file.replace('.hdf5', '_test.hdf5'), 'w')
    for idx in idxs[train_split_size + val_split_size:]:
        sub_data = test_dataset.create_dataset(idx, data=dataset[idx])
        sub_data.attrs['record'] = dataset[idx].attrs['record']
    test_dataset.close()

    dataset.close()
    print('Done.')


def create_data(user_info, ihr_dir, overlap_count, output_file):
    num_records = len(user_info)
    user_info = [{'ihr_dir': ihr_dir, 'overlap_count': overlap_count, **x} for x in user_info]
    output = h5py.File(output_file, 'w')

    print('Creating data...')
    print(f'\rProgress: 0/{num_records}', end='\r')
    dataset_idx = 0
    for count, uinfo in enumerate(user_info):
        ihr_values = read_file(os.path.join(ihr_dir, uinfo['record'] + '.txt'))
        for index in range(0, len(ihr_values), overlap_count):
            if len(ihr_values) - index > 4096:
                ihr_sequence = np.array(ihr_values[index:index + 4096])
                user_data = output.create_dataset(f'{dataset_idx}', data=ihr_sequence)
                user_data.attrs['record'] = uinfo['record']
                dataset_idx += 1
        print(f'\rProgress: {count + 1}/{num_records}', end='\r')
    print()

    output.close()


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--user_info', default=os.path.join(BASE_DIR, 'user_info.json'),
        help='Path to user info json'
    )
    parser.add_argument(
        '--ihr_dir', default=os.path.join(BASE_DIR, 'files', 'ihr_data'),
        help='Path to folder containing the IHR data'
    )
    parser.add_argument(
        '--overlap', default=128, type=int,
        help='How many sequences to overlap'
    )
    parser.add_argument(
        '--split_ratio', nargs='+', type=float, default=[0.8, 0.1, 0.1],
        help='Split of train, val, test'
    )
    parser.add_argument('--split', action='store_true', help='Split data')
    parser.add_argument(
        '--output', default=os.path.join(BASE_DIR, 'physiobank_dataset.hdf5'),
        help='Name of file in which dataset will be stored'
    )
    args = parser.parse_args()

    if args.split:
        split_data(args.output, args.split_ratio)
    else:
        create_data(
            read_file(args.user_info), args.ihr_dir,
            args.overlap, args.output
        )
