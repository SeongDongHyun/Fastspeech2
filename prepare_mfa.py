import re
import os
import yaml
import argparse

from jamo import h2j
from glob import glob
from tqdm import tqdm
# from g2pk import G2p


def mfa_align(config):
    text_file = config["path"]["transcript_path"]

    # Transcript File Read
    with open(text_file, 'r', encoding='utf-8') as fr:
        lines = [line.strip() for line in fr.readlines()]

    # [File_dir|text|preprocess_text|eng]
    filters = '([.,!?])"'
    for line in lines:
        temp = line.split("|")

        file_dir, text = temp[0], temp[3]

        # Filer로 특수문자 제거
        text = re.sub(re.compile(filters), '', text)

        # [:-3] -> .wav delete
        file_lab = file_dir[:-3] + 'lab'
        file_dir = os.path.join(config["path"]["raw_path"], file_lab)

        os.makedirs(os.path.join(config["path"]["raw_path"], file_lab[0]), exist_ok=True)
        with open(file_dir, 'w', encoding='utf-8') as fw:
            fw.write(text)

    file_list = sorted(glob(os.path.join(config["path"]["raw_path"], '**/*.lab')))

    phoneme_dict = {}
    # g2p = G2p()
    for file_name in tqdm(file_list):
        sentence = open(file_name, 'r', encoding='utf-8').readline()
        word_list = sentence.split(' ')
        phoneme_list = h2j(sentence).split(' ')
        # jamo = h2j(sentence).split(' ')
        for i, s in enumerate(phoneme_list):
            if s not in phoneme_dict:
                phoneme_dict[s] = ' '.join(phoneme_list[i])

    dict_name = os.path.join(config["path"]["raw_path"], 'korean_dict.txt')
    with open(dict_name, 'w', encoding='utf-8') as f:
        for key in phoneme_dict.keys():
            content = '{}\t{}\n'.format(key, phoneme_dict[key])
            f.write(content)


def mfa_train(config):

    dataset = config["path"]["corpus_path"]

    dict_txt = os.path.join(config["path"]["raw_path"], 'korean_dict.txt')
    print("MFA train_g2p Start")
    os.system(f'mfa train_g2p {dict_txt} korean.zip')
    print("MFA train_g2p Finish")

    g2p_txt = os.path.join(config["path"]["raw_path"], 'korean.txt')
    print("MFA G2P Start!")
    os.system(f'mfa g2p korean.zip {dataset} {g2p_txt}')
    print("MFA G2P Finish!")

    # MFA Train
    text_gird_folder = os.path.join(config["path"]["preprocessed_path"], 'TextGrid')
    os.makedirs(text_gird_folder, exist_ok=True)
    print('MFA Train Start!')
    os.system(f'mfa train {dataset} {g2p_txt} {text_gird_folder}')
    print('MFA Train Finish!')


def main(config):

    # Make .lab file
    print('MFA Align Start!')
    mfa_align(config)
    print('MFA Align Finish!')

    # MFA Train
    mfa_train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    config_dir = os.path.join("./config", args.dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    main(preprocess_config)
