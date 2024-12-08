from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pyaudio
import matplotlib.pylab as plt
import matplotlib
import torchaudio
import io
import numpy as np
import torch
import random
import librosa


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# torchaudio.set_audio_backend("soundfile")
set_seed(1)
torch.set_num_threads(1)

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad:v4.0',
                                  model='silero_vad',
                                  force_reload=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


# Define constants
DATASET_PATH = "0_large-corpus"
PROTOCOL_PATH = "0_large-corpus/protocol.txt"

SAMPLE_RATE = 16000
CUT_SIZE = 40000  # 2.5 seconds


DESTINATION_PATH = f"0_large-corpus/trim"

if not os.path.exists(DESTINATION_PATH):
    os.makedirs(DESTINATION_PATH)

THRESHOLD = 0.8  # Threshold for VAD to determine speech


def process_line(line):
    line = line.strip().split()
    file = line[0]
    file_path = os.path.join(DATASET_PATH, file)
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.squeeze(0)

    # Convert to numpy to trim
    waveform = waveform.numpy()
    waveform = librosa.effects.trim(waveform)[0]
    waveform = torch.from_numpy(waveform)

    if sample_rate != SAMPLE_RATE:
        print(f"File {file} sample rate is not {SAMPLE_RATE}")
        print('loading with librosa')
        waveform, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        waveform = librosa.effects.trim(waveform)[0]
        # convert to tensor
        waveform = torch.from_numpy(waveform)

    duration = len(waveform) / SAMPLE_RATE
    number_of_cuts = len(waveform) // CUT_SIZE
    residual = len(waveform) % CUT_SIZE

    results = []
    # ___ is used to separate the original file name and the cut number
    for i in range(number_of_cuts):
        cut_waveform = waveform[i * CUT_SIZE: (i + 1) * CUT_SIZE]
        new_confidence = vad_model(cut_waveform, SAMPLE_RATE).item()
        suffix = "" if new_confidence > THRESHOLD else "_no_speech"

        ori_filename, file_extension = file.split(".")

        file_name = f"{ori_filename}___{i}{suffix}.{file_extension}"

        file_path = os.path.join(DESTINATION_PATH, file_name)

        # Pre-process the cut_waveform to remove silence
        cut_waveform = librosa.effects.trim(cut_waveform.numpy())[0]
        cut_waveform = torch.from_numpy(cut_waveform)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        torchaudio.save(os.path.join(DESTINATION_PATH, file_name),
                        cut_waveform.unsqueeze(0), SAMPLE_RATE)

    # Process residual part if necessary
    if residual > 0 and len(waveform[-residual:]) >= 512:
        process_residual_part(waveform, residual,
                              number_of_cuts, line, results)

    return results


def process_residual_part(waveform, residual, count, line, results):
    file_line = line[0]

    # get extension of the file
    ori_filename, file_extension = file_line.split(".")

    cut_waveform = waveform[-residual:]
    new_confidence = vad_model(cut_waveform, SAMPLE_RATE).item()
    suffix = "" if new_confidence > THRESHOLD else "_no_speech"
    file_name = f"{ori_filename}___{count}_residual{suffix}.{file_extension}"
    file_path = os.path.join(DESTINATION_PATH, file_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    torchaudio.save(file_path,
                    cut_waveform.unsqueeze(0), SAMPLE_RATE)


# Reading the lines in advance to prevent file access issues in threads
with open(PROTOCOL_PATH) as file:
    lines = file.readlines()

executor = ProcessPoolExecutor(max_workers=40)
futures = [executor.submit(process_line, line) for line in lines]


for future in tqdm(as_completed(futures), total=len(futures)):
    result = future.result()
