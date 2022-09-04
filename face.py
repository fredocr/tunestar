from deepface import DeepFace
import numpy as np



VALUES = {"sad": np.array([0.225, 0.333, 0.149]),
          "happy": [1, 0.735, 0.772],
          "disgust": [0.051, 0.773, 0.274],
          "angry": [0.122, 0.83, 0.604],
          "fear": [0.073, 0.84, 0.293],
          "surprise": [0.784, 0.855, 0.539],
          "neutral": [0.5, 0.5, 0.5]}
VAD_max = [8.475, 7.27, 7.44]


def get_VAD(filename):
    fa = DeepFace.analyze(img_path=filename, actions=['emotion'])

    print(fa['dominant_emotion'])
    emotions = fa['emotion']

    result = np.array([0, 0, 0], dtype=np.float64)

    for k, v in emotions.items():
        result[0] += v * VALUES[k][0]
        result[1] += v * VALUES[k][1]
        result[2] += v * VALUES[k][2]

    r = []
    for i in range(0, 3):
        r.append(result[i] * VAD_max[i] / 100)

    return r

filename = "../data/fear3.jpg"
print(get_VAD(filename))