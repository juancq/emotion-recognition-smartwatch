import argparse
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as plticker

def aggregate_impor(impor):
    new_impor = []
    for user_impor in impor:
        # folds x features
        user_impor = np.array(user_impor)
        # 1 x features
        sum_impor = np.mean(user_impor, axis=0)
        sum_impor = sum_impor / np.max(sum_impor)
        # user x features
        new_impor.append(sum_impor)

    medians = np.median(np.array(new_impor), axis=0)

    indices = range(len(medians))
    # transpose: features x user
    indices = sorted(indices, key=lambda i: medians[i], reverse=True)
    return np.array(new_impor).T[indices], indices


def make_legend(colors):
    """Make a legend for the acc, gyro, and heart sensor data"""

    acc_patch = mpatches.Patch(label='Acc', 
                            facecolor=colors[0], edgecolor='black')
    gyro_patch = mpatches.Patch(label='Gyro', 
                            facecolor=colors[1], edgecolor='black')
    heart_patch = mpatches.Patch(label='Heart', 
                            facecolor=colors[2], edgecolor='black',
                            hatch=r'\\')

    plt.rcParams["legend.fontsize"] = 11.5

    l = plt.legend(handles=[heart_patch, acc_patch, gyro_patch], 
                    loc=1, frameon=False,
                    ncol=3, borderaxespad=0.2, 
                    columnspacing=0.7, handletextpad=0.3, 
                    fancybox=True, framealpha=0.3,
                    bbox_to_anchor=(0., 1.07, 1., .102))
    l.get_frame().set_facecolor('white')
    return l


def main():
    ''' Creates feature importances plot divided by condition.  '''

    parser = argparse.ArgumentParser("plot accelerometer data")
    parser.add_argument("-mo", type=str, help="movie yaml")
    parser.add_argument("-mu", type=str, help="music yaml")
    parser.add_argument("-mw", type=str, help="music+walk yaml")
    parser.add_argument("-o", "--output_file", type=str, help="file name for saving the generated plot.")
    parser.add_argument("-r", "--dpi", type=int, help="resolution of image", default = 300)

    # set variables based on arguments passed
    args = parser.parse_args()
    dpi = args.dpi
    output_file = args.output_file

    plt.style.use('seaborn-whitegrid')
    plt.figure(dpi=dpi)
    plt.rcParams.update({'font.size': 10})

    movie = yaml.load(open(args.mo))
    music = yaml.load(open(args.mu))
    music_walk = yaml.load(open(args.mw))
    data = [movie['rf'], music['rf'], music_walk['rf']]

    # colors for different sensors in boxplots
    colors=['lightblue', 'white', 'thistle']
    # titles for subplots
    titles = ['Movie', 'Music', 'Music while walking']

    # labels for features 
    labels_unsorted = [s.strip() for s in open('feature_list').readlines()]

    plt.figure(figsize=(9,9))

    share = None
    for i, (group, title) in enumerate(zip(data, titles)):
        if share:
            share = plt.subplot(3, 1, i+1, sharex=share)
        else:
            share = plt.subplot(3, 1, i+1)

        l1 = make_legend(colors)
        plt.gca().add_artist(l1)

        feature_impor = group

        feature_impor_all, indices = aggregate_impor(feature_impor)
        labels = [labels_unsorted[s] for s in indices]

        limit = 30
        bp = plt.boxplot(feature_impor_all[:limit].T, patch_artist=True, 
                    labels=labels[:limit], sym='+')

        # change color of median line
        for median in bp['medians']:
            median.set(color='red', linewidth=1.8)

        # set color of boxplot based on sensor type
        for j, box in enumerate(bp['boxes']):
            if 'acc_' in labels[j] or 'mag' in labels[j]:
                box.set(facecolor=colors[0])
            elif 'gyro_' in labels[j]:
                box.set(facecolor=colors[1])
            elif 'heart' in labels[j]:
                box.set(facecolor=colors[2])
                box.set(hatch = r'\\')

        short_labels = [re.sub(r'^(acc_|gyro_)', '', l) for l in labels[:limit]]
        plt.xticks(range(1,limit+1), short_labels, fontsize=9, rotation=70)
        plt.ylabel('Feature importance')
        # for max =1
        plt.ylim(0.0, 1.01)
        share.xaxis.grid(False)
        share.tick_params(labelright=True)
        plt.title(title, fontsize=12)


    plt.suptitle('Feature Importances per Condition', fontsize=13)

    plt.subplots_adjust(wspace=1.)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_file + '.png', bbox_inches='tight')


if __name__ == "__main__":
    main()
