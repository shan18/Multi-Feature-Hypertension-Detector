import os
import json
import argparse
import random
import pandas as pd
import numpy as np
from multiprocessing import Pool


def read_file(path):
    with open(path) as f:
        if path.endswith('.json'):
            return json.load(f)
        elif path.endswith('.txt'):
            return [float(line.strip()) for line in f]


def write_file(path, data):
    with open(path, 'w') as f:
        if path.endswith('.json'):
            json.dump(data, f, indent=2)


def generate_sequences(uinfo):
    ihr_values = read_file(os.path.join(uinfo['ihr_dir'], uinfo['record'] + '.txt'))
    uihr = []
    for index in range(0, len(ihr_values), uinfo['overlap_count']):
        if len(ihr_values) - index > 4096:
            ihr_sequence = np.array(ihr_values[index:index + 4096])
            uihr.append(ihr_sequence)

    df = pd.DataFrame(uihr)
    return df, uinfo['record'], len(uihr)


def split_data(user_info_file, split_ratio):
    user_info = read_file(user_info_file)
    random.shuffle(user_info)
    train_split_idx = int(len(user_info) * split_ratio[0])
    val_split_idx = train_split_idx + int(len(user_info) * split_ratio[1])
    train_user_info = user_info[:train_split_idx]
    val_user_info = user_info[train_split_idx:val_split_idx]
    test_user_info = user_info[val_split_idx:]

    write_file(user_info_file.replace('.json', '_train.json'), train_user_info)
    write_file(user_info_file.replace('.json', '_val.json'), val_user_info)
    write_file(user_info_file.replace('.json', '_test.json'), test_user_info)


def create_data(user_info_file, ihr_dir, overlap_count, output_file, num_threads):
    user_info = read_file(user_info_file)
    num_records = len(user_info)
    user_info_mp = [{'ihr_dir': ihr_dir, 'overlap_count': overlap_count, **x} for x in user_info]
    user_seq_info = {}

    pool = Pool(num_threads)

    print('Creating data...')
    print(f'\rProgress: 0/{num_records}', end='\r')
    for count, (df_sequences, urecord, u_num_seq) in enumerate(pool.imap_unordered(generate_sequences, user_info_mp)):
        user_seq_info[urecord] = u_num_seq
        df_sequences.to_hdf(output_file, key=f'r{urecord}', mode='a')
        print(f'\rProgress: {count + 1}/{num_records}', end='\r')
    print()

    # Update user info
    for idx in range(num_records):
        user_info[idx]['num_seq'] = user_seq_info[user_info[idx]['record']]
        user_info[idx]['record'] = f'r{user_info[idx]["record"]}'
    user_info = [x for x in user_info if x['num_seq'] > 0]
    write_file(user_info_file, user_info)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--user_info', default=os.path.join(BASE_DIR, 'files', 'user_info.json'),
        help='Path to user info json'
    )
    parser.add_argument(
        '--ihr_dir', default=os.path.join(BASE_DIR, 'files', 'ihr_data'),
        help='Path to folder containing the IHR data'
    )
    parser.add_argument(
        '--overlap', default=1, type=int,
        help='How many sequences to overlap'
    )
    parser.add_argument('--num_threads', default=8, type=int, help='Number of threads')
    parser.add_argument(
        '--split_ratio', nargs='+', type=float, default=[0.8, 0.1, 0.1],
        help='Split of train, val, test'
    )
    parser.add_argument(
        '--output', default=os.path.join(BASE_DIR, 'files', 'physiobank_dataset.h5'),
        help='Name of file in which dataset will be stored'
    )
    args = parser.parse_args()

    create_data(
        args.user_info, args.ihr_dir,
        args.overlap, args.output, args.num_threads
    )

    split_data(args.user_info, args.split_ratio)
