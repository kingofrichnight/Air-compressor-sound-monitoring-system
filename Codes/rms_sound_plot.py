import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Main folder
base_folder = r"C:\Users\hades\Downloads\segmented_per_subfolder\260302"

states = ["off", "rest", "running"]

frame_length = 2048
hop_length = 512

plt.figure(figsize=(14, 5))

for state in states:
    folder_path = os.path.join(base_folder, state)

    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    wav_files.sort()

    print(f"{state}: {len(wav_files)} files")

    combined_signal = np.array([])
    sample_rate = None

    for file in wav_files:
        file_path = os.path.join(folder_path, file)

        signal, sr = librosa.load(file_path, sr=None)

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"Sample rate mismatch in {file}")

        combined_signal = np.concatenate((combined_signal, signal))

    rms = librosa.feature.rms(
        y=combined_signal,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    time = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sample_rate,
        hop_length=hop_length
    )

    plt.plot(time, rms, label=state)

plt.title("RMS Comparison: OFF vs REST vs RUNNING")
plt.xlabel("Time (seconds)")
plt.ylabel("RMS Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()