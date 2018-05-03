import argparse
import math
import numpy as np

from collections import OrderedDict
from scipy import stats
from scipy import signal


def filter(data, fs):

    #third order median filter
    total_acc_x = signal.medfilt(data[:,0], 3)
    total_acc_y = signal.medfilt(data[:,1], 3)
    total_acc_z = signal.medfilt(data[:,2], 3)
    data[:, 0] = total_acc_x
    data[:, 1] = total_acc_y
    data[:, 2] = total_acc_z
    return data


def angle_between_vectors(a, b):
    dot = np.dot(a, b)
    cp = np.cross(a, b)
    cp_norm = np.sqrt(np.sum(cp * cp))
    angle = math.atan2(cp_norm, dot)
    return angle


def get_feature_vector(data):
    feature_functions = [
                       # 1. 
                       np.mean,
                       # 2. 
                       np.amax,
                       # 3. 
                       np.amin,
                       # 4. 
                       np.std,
                       # 5. energy
                       lambda d: np.sum(d**2)/d.shape[0],
                       # 6.
                       stats.kurtosis,
                       # 7.
                       stats.skew,
                       # 8. rms
                       lambda d: np.sqrt(np.mean(np.square(d))),
                       # 9. rss
                       lambda d: np.sqrt(np.sum(np.square(d))),
                       # 10. area
                       np.sum,
                       # 11. abs area
                       lambda d: np.sum(np.abs(d)),
                       # 12. abs mean
                       lambda d: np.mean(np.abs(d)),
                       # 13. range
                       lambda d: np.amax(d)-np.amin(d),
                       # 14. quartiles
                       lambda d: np.percentile(d, 25),
                       # 15. quartiles
                       lambda d: np.percentile(d, 50),
                       # 16. quartiles
                       lambda d: np.percentile(d, 75),
                       # 17. mad
                       lambda d: np.median(np.abs(d - np.median(d)))]

    features = [f(data) for f in feature_functions]

    return features
    #return np.array(features)


def extract_features(window):

    features = []
    heart_rate = window[:, -1]
    window_no_hr = window[:, :-1]
    for column in window_no_hr.T:
        features.extend(get_feature_vector(column))

    # acc
    # 17 * 3 = 51
    # gyro
    # 17 * 3 = 51

    # total = 102

    ##angle - 3
    x = window[:, 0]
    y = window[:, 1]
    z = window[:, 2]

    # 51 + 3

    vector = np.array([np.mean(x), np.mean(y), np.mean(z)])
    angle_wrt_xaxis = angle_between_vectors(vector, np.array([1, 0, 0]))
    angle_wrt_yaxis = angle_between_vectors(vector, np.array([0, 1, 0]))
    angle_wrt_zaxis = angle_between_vectors(vector, np.array([0, 0, 1]))
    features.extend([angle_wrt_xaxis, angle_wrt_yaxis, angle_wrt_zaxis])

    ## magnitude - std - 1
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    features.append(np.std(magnitude))

    # (17*3) + (17*3) + 3 + 1 + 1 (hr) = 107
    # + y label = 108

    features.append(heart_rate[0])

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", metavar='file', type=str, nargs='+', help="file containing acc data")
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument("-w", help="window size (whole seconds)", type=float, default=1)
    parser.add_argument("--overlap", help="overlap (percent, i.e. 0, 0.5, 0.8)", type=float, default=0.5)
    parser.add_argument("-d", "--delimiter", type=str, help="delimiter used in file, default is , (csv)", default = ',')
    args = parser.parse_args()

    window_size_sec = args.w
    overlap = args.overlap
    input_files = args.input_files
    output_dir = args.output_dir.strip('/') + '/'
    delimiter = args.delimiter

    FREQ_RATE = 24.0
    
    window_size = int(window_size_sec * FREQ_RATE)
    step = int(window_size * (1.-overlap))

    for fname in input_files:
        short_name = fname.split('/')[-1]
        print 'processing ', short_name
        condition_emotion = np.genfromtxt(fname, skip_header=1, delimiter=delimiter, usecols=(0,1))
        emotions = map(int, condition_emotion[:,1].tolist())

        data = np.genfromtxt(fname, skip_header=1, delimiter=delimiter, usecols=range(2, 9))

        # get emotions from second column
        emotion_ids = list(OrderedDict.fromkeys(emotions))
        emo_0 = emotions.index(emotion_ids[0])
        emo_1 = emotions.index(emotion_ids[1])
        emo_2 = emotions.index(emotion_ids[2])
        frames = [(emo_0, emo_1), (emo_1, emo_2), (emo_2, len(emotions))]

        features = []

        for (fstart, fend), label in zip(frames, emotion_ids):

            # filter data within start-end time, except heart rate
            data[fstart:fend,:-1] = filter(data[fstart:fend,:-1], FREQ_RATE)
            # extract consecutive windows
            i = fstart
            while i+window_size < fend:
                window = data[i:i+window_size]

                f_vector = extract_features(window)
                f_vector.append(label)
                features.append(f_vector)
                i += step


        features = np.array(features)

        filename = 'features_{}'.format(short_name)
        print '\tSaving file {}...'.format(filename)
        np.savetxt(output_dir + filename, features, fmt='%f', delimiter=',')
        print '\tfeatures: ', features.shape


if __name__ == "__main__":
    main()

