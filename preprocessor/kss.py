import os
import re

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text

def prepare_align(config):
    '''
        in_dir: Input wave path
        out_dir: Save foler preprocessed data
        wav_tag: Wave file tag
        txt_dir: txt dir name
        wav_dir: wav dir name
        sampling_rate: Set sampling rate (default=22050)
        max_wav_value: max wav value (default=32768.0)
        cleaners: Text preprocessing Select
    '''

    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    # kss/transcript.v.1.4.txt
    with open(os.path.join(in_dir, "transcript.v.1.4.txt"), encoding="utf-8") as fr:
        lines = [lines.strip() for lines in fr.readlines()]

    for line in tqdm(lines, total=len(lines)):
        parts = line.strip().split("|")
        base_name = parts[0][:-4]
        spk_folder = base_name.split("/")[0]
        text = parts[3]

        # Special Character Extract with filters
        filters = '([.,!?])"'
        text = re.sub(re.compile(filters), '', text)
        text = _clean_text(text, cleaners)

        wav_path = os.path.join(in_dir, "wav", "{}.wav".format(base_name))
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, spk_folder), exist_ok=True)
            wav, _ = librosa.load(wav_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)
