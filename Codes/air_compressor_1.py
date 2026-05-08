from xml.etree import ElementTree as ET
import requests
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math


# =========================
# CONSTANTS
# =========================
SAMPLE = "sample"
CURRENT = "current"
SAMP_RATE = 48000
CHUNK = 2048

AGENT = "http://192.168.1.4:5001/"
N = 23
N_FFT = 2024
N_MELS = 128


# =========================
# GET MTCONNECT DATA
# =========================
response = requests.get(AGENT + CURRENT + "?path=//DataItem[@id=%27sensor1%27]")
root = ET.fromstring(response.content)

MTCONNECT_STR = root.tag.split("}")[0] + "}"

header = root.find("./" + MTCONNECT_STR + "Header")
header_attribs = header.attrib

nextSeq = int(header_attribs["nextSequence"])
firstSeq = int(header_attribs["firstSequence"])
lastSeq = int(header_attribs["lastSequence"])


# timestamp
for sample in root.iter(MTCONNECT_STR + "DisplacementTimeSeries"):
    timestamp = sample.get("timestamp")


print(f"timestamp: {timestamp}, firstSeq={firstSeq}, nextSeq={nextSeq}, lastSeq={lastSeq}")


# =========================
# GET SIGNAL WINDOW
# =========================
response = requests.get(
    AGENT + SAMPLE +
    "?from=" + str(int(lastSeq - N)) +
    "&count=" + str(N) +
    "&path=//DataItem[@id=%27sensor1%27]"
)

root = ET.fromstring(response.content)

signal_array = []

for sample in root.iter(MTCONNECT_STR + "DisplacementTimeSeries"):
    chunk = np.fromstring(sample.text, dtype=np.int16, sep=' ') / (2 ** 15)
    signal_array = np.append(signal_array, chunk)

signal = np.array(signal_array, dtype=np.float32)


# =========================
# FEATURE EXTRACTION
# =========================
signal_rms = np.sqrt(np.mean(signal ** 2))

sound_level = 20 * math.log10(signal_rms / 9.9963e-7) - 28.87

M_signal = librosa.feature.melspectrogram(
    y=signal,
    sr=SAMP_RATE,
    n_fft=N_FFT,
    hop_length=int(N_FFT / 4),
    win_length=N_FFT,
    window='hann',
    n_mels=N_MELS
)


# =========================
# ML LABEL
# =========================
def get_label(rms):
    if rms <= 0.008:
        return "OFF"
    elif rms <= 0.2:
        return "REST"
    else:
        return "RUNNING"

label = get_label(signal_rms)


print(f"SPL: {sound_level:.2f} dB | RMS: {signal_rms:.5f} | LABEL: {label}")


# =========================
# VISUALIZATION
# =========================
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

librosa.display.waveshow(signal, sr=SAMP_RATE, ax=ax[0], color="blue")
ax[0].text(0.1, 0.7, f"{sound_level:.2f} dB\n{label}")

img = librosa.display.specshow(
    librosa.power_to_db(2 * abs(M_signal) / N_FFT, ref=1),
    ax=ax[1],
    x_axis='time',
    y_axis='mel',
    sr=SAMP_RATE,
    vmin=-80,
    vmax=0
)

ax[0].set(title="Time Domain", ylim=[-1, 1])
ax[1].set(title="Mel Spectrogram")

fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

plt.show()


print("Signal duration:", len(signal) / SAMP_RATE, "sec")