import argparse
import math
import numpy as np

from collections import OrderedDict
from scipy import stats
from scipy import signal


def main():
    '''
    Run as:
    python3 build_input_tensor.py csv_files output_folder
    Example:
    python3 build_input_tensor.py walking_data/m*.csv tensor_data_folder
    python3 build_input_tensor.py -w 2 walking_data/m*.csv tensor_data_2secwindow
    '''
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
        print('processing ', short_name)
        condition_emotion = np.genfromtxt(fname, skip_header=1, delimiter=delimiter, usecols=(0,1))
        emotions = list(map(int, condition_emotion[:,1].tolist()))

        data = np.genfromtxt(fname, skip_header=1, delimiter=delimiter, usecols=range(2, 9))

        # get emotions from second column
        emotion_ids = list(OrderedDict.fromkeys(emotions))
        emo_0 = emotions.index(emotion_ids[0])
        emo_1 = emotions.index(emotion_ids[1])
        emo_2 = emotions.index(emotion_ids[2])
        frames = [(emo_0, emo_1), (emo_1, emo_2), (emo_2, len(emotions))]

        features = []
        y = []

        for (fstart, fend), label in zip(frames, emotion_ids):

            # extract consecutive windows
            i = fstart
            while i+window_size < fend:
                window = data[i:i+window_size]

                features.append(window)
                y.append(label)
                i += step


        features = np.array(features)
        y = np.array(y)

        n = short_name.rstrip('.csv')
        filename = f'{n}_x'
        print(f'\tSaving file {filename}...')
        np.save(output_dir + filename, features)
        filename = f'{n}_y.csv'
        np.savetxt(output_dir + filename, y, fmt='%f', delimiter=',')

        print('\tfeatures: ', features.shape)


if __name__ == "__main__":
    main()

