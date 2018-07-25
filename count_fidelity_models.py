import argparse
import yaml
import numpy as np


def main():
    '''
    Count how many of the personal models have high fidelity results (>=80%), 
    average fidelity (>=70% and <80%), and poor fidelity (<70%).
    '''
    parser = argparse.ArgumentParser("plot accelerometer data")
    parser.add_argument("-mo", type=str, help="movie yaml")
    parser.add_argument("-mu", type=str, help="music yaml")
    parser.add_argument("-mw", type=str, help="music+walk yaml")

    # set variables based on arguments passed
    args = parser.parse_args()
    movie = yaml.load(open(args.mo))
    music = yaml.load(open(args.mu))
    music_walk = yaml.load(open(args.mw))

    labels = ['rf', 'logit']

    base = 'acc'
    data = []
    for key in labels:
        data.extend(movie[key][base])
        data.extend(music[key][base])
        data.extend(music_walk[key][base])

    data = np.array(data)
    total =  data.shape[0]

    print 'total personal models = {}'.format(total)

    upper = 0.80
    lower = 0.70
    competent = len(np.where(data>upper)[0])
    average = len(np.where((data<=upper) & (data >= lower))[0])
    poor = len(np.where(data<lower)[0])
    print 'greater than 80% = {}'.format(competent)
    print '70-80% = {}'.format(average)
    print '<70% = {}'.format(poor)
    print 'check total =  {}'.format(competent + average + poor)


if __name__ == "__main__":
    main()
