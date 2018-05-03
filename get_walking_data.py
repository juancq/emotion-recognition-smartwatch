import numpy as np
import argparse
import sys
import glob


def main():
    '''
    Run as:
    python get_walking_data.py user_study_encoding.csv input_directory output_directory

    Example:
    python get_walking_data.py user_study_encoding.csv raw_data walking_data

    Takes a csv file of start-stop end times for each participant,
    and generates a csv file with only the walking times.
    The output file is formatted as follows:
    condition,emotion,walking_data

    where condition is one of mo, mu, and mw
    and emotions are sad, neutral, happy
    '''

    EMOTIONS = {'s': -1, 'n': 0, 'h': 1}
    R_EMOTIONS = {-1:'s', 0:'n', 1:'h'}
    CONDITIONS = {'mo': 0, 'mu': 1, 'mw': 2}

    parser = argparse.ArgumentParser(description="frame extraction")
    parser.add_argument("encoding", type=str, help="file containing participant list")
    parser.add_argument("input_dir", type=str, help="directory containing participant data")
    parser.add_argument('output_dir', type=str, help='output directory')

    args = parser.parse_args()
    input_dir = args.input_dir.strip('/') + '/'
    output_dir = args.output_dir.strip('/') + '/'

    # get participant id and condition code
    encoding = np.genfromtxt(args.encoding, delimiter=',', dtype=str, skip_header=1)
    participants = [p.lower() for p in encoding[:,0]]
    conditions = [p.lower() for p in encoding[:,1]]

    for i, pid in enumerate(participants):

        condition, emotions = conditions[i].split('-')
        
        # find file matching ew id
        fname = glob.glob(input_dir + pid + '_*')
        if len(fname) > 1:
            print '^^^^ more than one file matched %s ' % pid
            return
        fname = fname[0]
        short_name = fname.split('/')[-1]
        print 'processing ', short_name

        # read data from ew file
        time = np.genfromtxt(fname, delimiter=',', dtype=str, usecols=(0), skip_header=2, skip_footer=1)
        time = [':'.join([frag.zfill(2) for frag in t.split(':')[:-1]]) for t in time.tolist()]

        #acc + gyro + heart
        data = np.genfromtxt(fname, delimiter=',', usecols=(1,2,3,7,8,9,10), skip_header=2, skip_footer=1)

        # get start-stop times
        start_stop_time = [t.replace('.', ':') for t in encoding[i, -6:].tolist()]

        indexes = []
        for time_ in start_stop_time:
            if time_ in time:
                index = time.index(time_)
                indexes.append(index)
            else:
                print 'invalid index ', short_name, start_stop_time
                break

        # check for valid indexes
        if len(indexes) != 6:
            print 'missing an index ', short_name
            continue

        invalid = False
        for i in range(len(indexes)-1):
            if indexes[i] >= indexes[i+1]:
                invalid = True
                break

        if invalid:
            print 'invalid index ', indexes
            continue
            
        # compile the rows between each start-stop 
        document = []

        id_ = 0
        emotion_col = []
        #for all sets of walking time
        for set_id, i in enumerate(range(0, len(indexes), 2)):
            start, end = indexes[i], indexes[i+1]
            document.extend(data[start:end])
            # compile emo ids as list
            emo_code = EMOTIONS.get(emotions[set_id])
            emotion_col.extend([emo_code] * (end-start))

        # make condition column, same for all
        condition_col = [CONDITIONS[condition]] * len(document)

        # add frame ids as first column
        document = np.column_stack((condition_col, emotion_col, document))
            
        np.savetxt('{}{}_{}'.format(output_dir, condition, short_name), 
                        document, delimiter=',', 
                        header='condition,emotion,data', fmt='%s')

if __name__ == "__main__":
    main()
