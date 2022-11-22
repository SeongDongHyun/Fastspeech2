import os
import re

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
from g2pk import G2p
from jamo import h2j

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

    # Read multi_kr/metadata.csv
    with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as fr:
        lines = [lines.strip() for lines in fr.readlines()]

    g2p = G2p()
    for line in tqdm(lines, total=len(lines)):
        parts = line.strip().split("|")
        base_name = parts[0]
        spk_folder = base_name[:3]
        text = parts[2]

        # Special Character Extract with filters
        filters = '([.,!?])"'
        text = re.sub(re.compile(filters), '', text)
        text = _clean_text(text, cleaners)
        text = h2j(g2p(text))

        wav_path = os.path.join(in_dir, "wavs", spk_folder, "{}.wav".format(base_name))
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, spk_folder), exist_ok=True)
            wav, _ = librosa.load(wav_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, spk_folder, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, spk_folder, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)
