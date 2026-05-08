import os
import random
import numpy as np
import soundfile as sf

# =========================================================
# SETTINGS
# =========================================================
input_root_folder = r"C:\Users\hades\Box\air_compressor\sound"
output_root_folder = r"C:\Users\hades\Downloads\segmented_per_subfolder"

segment_duration = 2.0
overlap_ratio = 0.0
min_segment_samples = 1000

OFF_THRESHOLD = 0.008
RUNNING_THRESHOLD = 0.20
# between OFF_THRESHOLD and RUNNING_THRESHOLD => REST

target_per_class = 200
normalize_before_save = False
random_seed = 42

random.seed(random_seed)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def compute_rms(signal):
    signal = signal.astype(np.float64)
    return np.sqrt(np.mean(signal ** 2))

def classify_segment(rms_value, off_th=OFF_THRESHOLD, run_th=RUNNING_THRESHOLD):
    if rms_value < off_th:
        return "off"
    elif rms_value >= run_th:
        return "running"
    else:
        return "rest"

def normalize_audio(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal

def convert_to_mono(audio):
    if len(audio.shape) == 1:
        return audio
    return np.mean(audio, axis=1)

def save_segment(segment, sr, save_path):
    sf.write(save_path, segment, sr)

def get_wav_files_in_folder(folder_path):
    wav_files = []
    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isfile(full_path) and file_name.lower().endswith(".wav"):
            wav_files.append(full_path)
    return sorted(wav_files)

# =========================================================
# COLLECT CANDIDATES FROM ONE SUBFOLDER
# =========================================================
def collect_candidates_from_subfolder(subfolder_path):
    wav_files = get_wav_files_in_folder(subfolder_path)

    candidates = {
        "off": [],
        "rest": [],
        "running": []
    }

    if not wav_files:
        print(f"  No WAV files found in: {subfolder_path}")
        return candidates

    print(f"  Found {len(wav_files)} WAV files")

    for file_idx, file_path in enumerate(wav_files, start=1):
        print(f"    [FILE {file_idx}/{len(wav_files)}] {os.path.basename(file_path)}")

        try:
            audio, sr = sf.read(file_path)
        except Exception as e:
            print(f"      Could not read file: {e}")
            continue

        audio = convert_to_mono(audio)
        total_samples = len(audio)

        segment_samples = int(segment_duration * sr)
        if segment_samples <= 0:
            continue

        step_samples = int(segment_samples * (1 - overlap_ratio))
        if step_samples <= 0:
            print("      Invalid overlap_ratio")
            continue

        if total_samples < segment_samples:
            print("      File too short, skipped")
            continue

        for start in range(0, total_samples - segment_samples + 1, step_samples):
            end = start + segment_samples
            segment = audio[start:end]

            if len(segment) < min_segment_samples:
                continue

            rms_value = compute_rms(segment)
            label = classify_segment(rms_value)

            candidates[label].append({
                "file_path": file_path,
                "start": start,
                "end": end,
                "rms": rms_value,
                "sr": sr
            })

    return candidates

# =========================================================
# SAVE EXACTLY N PER CLASS FOR ONE SUBFOLDER
# =========================================================
def save_selected_segments(subfolder_name, candidates):
    base_output = os.path.join(output_root_folder, subfolder_name)
    off_folder = os.path.join(base_output, "off")
    rest_folder = os.path.join(base_output, "rest")
    running_folder = os.path.join(base_output, "running")

    os.makedirs(off_folder, exist_ok=True)
    os.makedirs(rest_folder, exist_ok=True)
    os.makedirs(running_folder, exist_ok=True)

    summary = {}

    for label in ["off", "rest", "running"]:
        total_found = len(candidates[label])

        if total_found < target_per_class:
            print(f"  WARNING: {subfolder_name} has only {total_found} '{label}' segments")
            selected = candidates[label]
        else:
            selected = random.sample(candidates[label], target_per_class)

        summary[label] = len(selected)

        print(f"  Saving {len(selected)} '{label}' segments")

        for idx, item in enumerate(selected, start=1):
            file_path = item["file_path"]
            start = item["start"]
            end = item["end"]
            rms_value = item["rms"]

            try:
                audio, sr = sf.read(file_path)
            except Exception as e:
                print(f"    Could not re-read file: {file_path}")
                print(f"    Error: {e}")
                continue

            audio = convert_to_mono(audio)
            segment = audio[start:end]

            if normalize_before_save:
                segment = normalize_audio(segment)

            file_base = os.path.splitext(os.path.basename(file_path))[0]

            output_name = (
                f"{subfolder_name}_{label}_{idx:03d}_"
                f"{file_base}_s{start}_e{end}_rms_{rms_value:.6f}.wav"
            )

            if label == "off":
                save_path = os.path.join(off_folder, output_name)
            elif label == "rest":
                save_path = os.path.join(rest_folder, output_name)
            else:
                save_path = os.path.join(running_folder, output_name)

            save_segment(segment, sr, save_path)

    return summary

# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.exists(input_root_folder):
        print("Input root folder not found:")
        print(input_root_folder)
        return

    subfolders = [
        os.path.join(input_root_folder, name)
        for name in os.listdir(input_root_folder)
        if os.path.isdir(os.path.join(input_root_folder, name))
    ]

    subfolders = sorted(subfolders)

    if not subfolders:
        print("No subfolders found.")
        return

    print(f"Found {len(subfolders)} subfolders.\n")

    overall_report = []

    for subfolder_path in subfolders:
        subfolder_name = os.path.basename(subfolder_path)
        print("====================================================")
        print(f"Processing subfolder: {subfolder_name}")
        print("====================================================")

        candidates = collect_candidates_from_subfolder(subfolder_path)

        print(f"  Candidate OFF     : {len(candidates['off'])}")
        print(f"  Candidate REST    : {len(candidates['rest'])}")
        print(f"  Candidate RUNNING : {len(candidates['running'])}")

        summary = save_selected_segments(subfolder_name, candidates)

        overall_report.append({
            "subfolder": subfolder_name,
            "off": summary["off"],
            "rest": summary["rest"],
            "running": summary["running"]
        })

        print()

    print("====================================================")
    print("FINAL REPORT")
    print("====================================================")
    for item in overall_report:
        print(
            f"{item['subfolder']} -> "
            f"OFF: {item['off']}, "
            f"REST: {item['rest']}, "
            f"RUNNING: {item['running']}"
        )

    print("\nDone.")

if __name__ == "__main__":
    main()