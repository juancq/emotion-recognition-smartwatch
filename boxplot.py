import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt


def main():
    '''
    Creates a line plot of accelerometer data vs time.
    '''
    parser = argparse.ArgumentParser("plot accelerometer data")
    parser.add_argument("-mo", type=str, help="movie yaml")
    parser.add_argument("-mu", type=str, help="music yaml")
    parser.add_argument("-mw", type=str, help="music+walk yaml")
    parser.add_argument("-o", "--output_file", type=str, help="file name for saving the generated plot.")
    parser.add_argument("-r", "--dpi", type=int, help="resolution of image", default = 300)
    parser.add_argument("-lw", "--line_width", type=float, help="line_width", default = 1.0)
    parser.add_argument("--title", type=str, help="plot title")

    # set variables based on arguments passed
    args = parser.parse_args()
    linewidth = args.line_width
    dpi = args.dpi
    output_file = args.output_file
    main_title = args.title

    movie = yaml.load(open(args.mo))
    music = yaml.load(open(args.mu))
    music_walk = yaml.load(open(args.mw))

    plt.style.use('seaborn-whitegrid')
    plt.figure(dpi=dpi)
    plt.rcParams.update({'font.size': 12})

    labels = ['baseline', 'rf', 'logit']

    data = []
    data.append([movie[key]['acc'] for key in labels])
    data.append([music[key]['acc'] for key in labels])
    data.append([music_walk[key]['acc'] for key in labels])

    titles = ['Movie', 'Music', 'Music while walking']
    labels = ['Baseline', 'RF', 'Logistic']
    colors = ['darkseagreen', 'plum', 'sandybrown']

    share = None
    for i, (group, title) in enumerate(zip(data, titles)):
        if share:
            share = plt.subplot(1, 3, i+1, sharey=share)
        else:
            share = plt.subplot(1, 3, i+1)

        #sub_fig, ax = share
        share.xaxis.grid(False)
        start, end = share.get_xlim()
        step = 0.05
        share.yaxis.set_ticks(np.arange(start, end+step, step))

        plt.title(title, fontsize=12)
        if i == 0:
            plt.ylabel('Accuracy')

        bp = plt.boxplot(group, labels=labels, patch_artist=True)
        for j, box in enumerate(bp['boxes']):
            # change outline color
            box.set( color='#7570b3', linewidth=1.2)
            # change fill color
            #box.set(facecolor='#1b9e77')
            box.set(facecolor=colors[j])

    
    # for h vs s
    #plt.suptitle('Distribution of Accuracies of Personal Models Per Condition\nHappy vs Sad', fontsize=14)
    #plt.ylim(0.45, 1)

    # for s-n-h
    plt.suptitle('Distribution of Accuracies of Personal Models Per Condition\nHappy - Neutral - Sad', fontsize=14)
    plt.ylim(0.3, 1)

    # for emotion cv
    #plt.suptitle('Emotion Cross-Validation of Happy vs Sad', fontsize=14)
    #plt.ylim(0, 1)


    plt.subplots_adjust(wspace=1.)
    plt.tight_layout()
    plt.subplots_adjust(top=0.87, wspace=0.5)

    if output_file:
        plt.savefig(output_file + '.png', bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    main()
