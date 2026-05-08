from xml.etree import ElementTree as ET
import requests
import numpy as np
import librosa
import math
import tensorflow as tf
import time
from data_item import Event, Sample
from mtconnect_adapter import Adapter
import sys

# ==========================
# GLOBAL CONSTANTS
# ==========================
SAMPLE = "sample"
CURRENT = "current"

SAMP_RATE = 48000
N = 23
N_FFT = 2048
N_MELS = 128

# This is your sound-stream MTConnect agent
AGENT = "http://127.0.0.1:5000/"

# Your model file
MODEL_FILE = "air_compressor_cnn.h5"

# Class order based on your previous model output
CLASS_NAMES = ["OFF", "REST", "RUNNING"]

print("Loading model...")
model_keras = tf.keras.models.load_model(MODEL_FILE, compile=False)
print("Model loaded successfully.")

def get_sound_signal(response):
    root = ET.fromstring(response.content)
    MTCONNECT_STR = root.tag.split("}")[0] + "}"

    array = []

    for sample in root.iter(MTCONNECT_STR + "DisplacementTimeSeries"):
        if sample.text is not None:
            chunk = np.fromstring(sample.text, dtype=np.int16, sep=" ") / (2 ** 15)
            array = np.append(array, chunk)

    return np.array(array, dtype=np.float32)


def get_sound_level(signal):
    rms = np.sqrt(np.mean(signal ** 2))

    if rms <= 0:
        return 0

    sound_level = 20 * math.log10(rms / 9.9963e-7) - 28.87
    return sound_level


def get_rms(signal):
    return float(np.sqrt(np.mean(signal ** 2)))


def get_rms_state(rms):
    if rms > 0.25:
        return "RUNNING"
    elif rms > 0.05:
        return "REST"
    else:
        return "OFF"


def feature_extraction(x):
    M = librosa.feature.melspectrogram(
        y=x,
        sr=SAMP_RATE,
        n_fft=N_FFT,
        hop_length=int(N_FFT / 4),
        win_length=N_FFT,
        window="hann",
        n_mels=N_MELS
    )

    X = 2 * abs(M) / N_FFT

    # Make sure model input is 128 x 93
    if X.shape[1] < 93:
        pad_width = 93 - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")
    elif X.shape[1] > 93:
        X = X[:, :93]

    S = np.reshape(X, (1, X.shape[0], X.shape[1]))

    return S


class CurrentParsing(object):
    def __init__(self, response):
        root = ET.fromstring(response.content)
        MTCONNECT_STR = root.tag.split("}")[0] + "}"

        header = root.find("./" + MTCONNECT_STR + "Header")

        if header is None:
            print("ERROR: Header not found.")
            print(response.text[:1000])
            raise Exception("Header not found")

        header_attribs = header.attrib

        self.nextSeq = int(header_attribs["nextSequence"])
        self.firstSeq = int(header_attribs["firstSequence"])
        self.lastSeq = int(header_attribs["lastSequence"])

        self.timestamp = "N/A"

        for sample in root.iter(MTCONNECT_STR + "DisplacementTimeSeries"):
            self.timestamp = sample.get("timestamp")

class AirCompressorAdapter(object):

    def __init__(self, host, port):
        self.adapter = Adapter((host, port))

        # SAMPLE
        self.sound_level = Sample("spl")
        self.adapter.add_data_item(self.sound_level)
        
        #rms
        self.rms_value = Sample("rms")
        self.adapter.add_data_item(self.rms_value)

        # EVENTS
        self.execution = Event("e1")
        self.adapter.add_data_item(self.execution)

        self.compressor_state = Event("vs1")
        self.adapter.add_data_item(self.compressor_state)

        self.avail = Event("avail")
        self.adapter.add_data_item(self.avail)

        # START ADAPTER
        self.adapter.start()

        self.adapter.begin_gather()
        self.avail.set_value("AVAILABLE")
        self.execution.set_value("READY")
        self.compressor_state.set_value("UNKNOWN")
        self.sound_level.set_value(0)
        self.adapter.complete_gather()

        self.adapter_stream()
        
    def adapter_stream(self):
        while True:
            try:
                # Get latest sequence from sound stream
                Current = CurrentParsing(
                    requests.get(
                        AGENT + CURRENT + "?path=//DataItem[@id=%27sensor0%27]",
                        timeout=10
                    )
                )

                lastSeq = Current.lastSeq
                startSeq = max(Current.firstSeq, lastSeq - N)

                # Get 23 chunks of sound
                response = requests.get(
                    AGENT + SAMPLE +
                    "?from=" + str(startSeq) +
                    "&count=" + str(N) +
                    "&path=//DataItem[@id=%27sensor0%27]",
                    timeout=10
                )

                x = get_sound_signal(response)

                if len(x) == 0:
                    print("No sound data received.")
                    time.sleep(2)
                    continue

                # RMS
                rms = get_rms(x)
                rms_state = get_rms_state(rms)

                # Model prediction
                X = feature_extraction(x)
                yhat = model_keras.predict(X, verbose=0)

                Y = int(np.argmax(yhat))
                confidence = float(np.max(yhat))

                model_state = CLASS_NAMES[Y]
                # Final decision: use RMS as safety rule
                
             #   if rms > 0.35:
              #         final_state = "RUNNING"
              #          
              #  elif rms >= 0.08:
              #        final_statestate = "REST"
                        
             #   else:
               #     final_state = "OFF"
                final_state = model_state

                # Air compressor state logic
                if final_state == "OFF":
                    execution_value = "OFF"
                    compressor_value = "OFF"
                elif final_state == "RUNNING":
                    execution_value = "ON"
                    compressor_value = "RUNNING"
                elif final_state == "REST":
                    execution_value = "ON"
                    compressor_value = "REST"
                else:
                    execution_value = "UNKNOWN"
                    compressor_value = "UNKNOWN"

                sound_pressure = round(get_sound_level(x), 2)
                
                # Send values to MTConnect
                self.adapter.begin_gather()
                self.execution.set_value(execution_value)
                self.compressor_state.set_value(compressor_value)
                self.sound_level.set_value(sound_pressure)
                self.rms_value.set_value(round(rms, 6))
                self.adapter.complete_gather()
                

                print("--------------------------------------")
                print(f"Timestamp        : {Current.timestamp}")
                print(f"Execution        : {execution_value}")
                print(f"Compressor State : {compressor_value}")
                print(f"Sound Level      : {sound_pressure} dB SPL")
                print(f"RMS              : {rms:.6f}")
                print(f"RMS State        : {rms_state}")
                print(f"Model Output     : {yhat}")
                print(f"Confidence       : {confidence:.4f}")
                print(f"Final State      : {final_state}")
                print("--------------------------------------")

                time.sleep(2)

            except KeyboardInterrupt:
                print("Stopping adapter...")
                self.adapter.stop()
                sys.exit()

            except Exception as e:
                print("Error:")
                print(e)
                time.sleep(2)


if __name__ == "__main__":
    print("Starting Air Compressor Adapter...")
    AirCompressorAdapter("127.0.0.1", 7890)
