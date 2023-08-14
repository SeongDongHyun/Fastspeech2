import os

import tgt
import natsort
import numpy as np
from tqdm import tqdm

def read_file_list(folder_path):
    path_list = []
    for (path, dir, files) in os.walk(folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext.lower() == '.textgrid':
                path_list.append(path + '/' + filename)

    return natsort.natsorted(path_list)

def get_alignment(tier, sampling_rate, hop_length):
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


def alignments(tg_path):
    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phone, duration, start, end = get_alignment(
        textgrid.get_tier_by_name("phones"),
        sampling_rate=24000,
        hop_length=256,
    )

    return duration

def write_file(filename, result_list):
    with open(filename, 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(result_list))

def main():
    read_folder_path = '/home/dhseong/work/data/MFA/alignmnets'
    file_list = read_file_list(read_folder_path)

    reuslt_list = []
    for filename in tqdm(file_list, total=len(file_list)):
        key = filename.split('/')[-1][:-9]
        duration_list = alignments(filename)
        duration_list.append(0)
        duration = ' '.join(map(str, duration_list))
        reuslt_list.append(f'{key} {duration}')
    
    print(f'key: {key}')
    print(f'duration: {duration}')

    write_file_name = '/home/dhseong/work/data/MFA/durations'
    write_file(write_file_name, reuslt_list)


if __name__ == "__main__":
    main()
