from xml.etree import ElementTree as ET
import requests
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import time

## == GLOBAL CONSTANT ==
SAMPLE = "sample"
CURRENT = "current"
SAMP_RATE = int(48000)
CHUNK = int(2 ** 11)
AGENT = "http://192.168.1.4:5001/"

N = 23
N_FFT = 2048
N_MELS = 128


def get_sound_signal(response):
    root = ET.fromstring(response.content)
    MTCONNECT_STR = root.tag.split("}")[0] + "}"

    array = []

    for sample in root.iter(MTCONNECT_STR + "DisplacementTimeSeries"):
        chunk = np.fromstring(sample.text, dtype=np.int16, sep=' ') / (2 ** 15)
        array = np.append(array, chunk)

    return np.array(array, dtype=np.float32)


def get_sound_level(signal):
    signal_rms = np.sqrt(np.mean(signal ** 2))
    sound_level = 20 * math.log10(signal_rms / 9.9963e-7) - 28.87
    return sound_level


def feature_extraction(x):
    M = librosa.feature.melspectrogram(
        y=x,
        sr=SAMP_RATE,
        n_fft=N_FFT,
        hop_length=int(N_FFT / 4),
        win_length=N_FFT,
        window='hann',
        n_mels=N_MELS
    )

    X = 2 * abs(M) / N_FFT
    return np.reshape(X, (1, -1, X.shape[1]))


class CurrentParsing(object):
    def __init__(self, response):
        root = ET.fromstring(response.content)
        MTCONNECT_STR = root.tag.split("}")[0] + "}"

        header = root.find("./" + MTCONNECT_STR + "Header")
        header_attribs = header.attrib

        self.nextSeq = int(header_attribs["nextSequence"])
        self.firstSeq = int(header_attribs["firstSequence"])
        self.lastSeq = int(header_attribs["lastSequence"])

        for sample in root.iter(MTCONNECT_STR + "DisplacementTimeSeries"):
            self.timestamp = sample.get('timestamp')


if __name__ == "__main__":

    model_file = '20230309_212154_Prelab10_CNN_model1.h5'

    model_keras = tf.keras.models.load_model(model_file, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)

    # FIX (must stay outside loop)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    while True:

        Current = CurrentParsing(
            requests.get(AGENT + CURRENT + "?path=//DataItem[@id=%27sensor1%27]")
        )

        lastSeq = Current.lastSeq

        print('Last Sequence:', lastSeq)
        print('Timestamp:', Current.timestamp)

        x = get_sound_signal(
            requests.get(
                AGENT + SAMPLE +
                "?from=" + str(int(lastSeq - N)) +
                "&count=" + str(N) +
                "&path=//DataItem[@id=%27sensor1%27]"
            )
        )

        print('Sound level:', get_sound_level(x), 'dB')

        X = feature_extraction(x)

        interpreter.set_tensor(input_details['index'], X)
        interpreter.invoke()

        yhat = interpreter.get_tensor(output_details['index'])
        Y = yhat.argmax()

        print('Model output:', yhat)
        print('Prediction inference index:', Y)

        # =========================
        # YOUR BLOCK (UNCHANGED)
        # =========================
        if Y == 0:
            prediction_label = "OFF"

        elif Y == 1:
            prediction_label = "REST"

        elif Y == 2:
            prediction_label = "RUNNING"

        else:
            prediction_label = "UNKNOWN"
        # complete your algorithm to take appropriate prediction_label string accord

        print('The air compressor is now {}.\n'.format(prediction_label))

        time.sleep(1)