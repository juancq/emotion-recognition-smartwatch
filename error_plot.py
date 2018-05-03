import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as plticker


def main():
    '''
    Creates plot showing user lift for each personal model.
    '''
    parser = argparse.ArgumentParser("plot accelerometer data")
    parser.add_argument("-mo", type=str, help="movie yaml")
    parser.add_argument("-mu", type=str, help="music yaml")
    parser.add_argument("-mw", type=str, help="music+walk yaml")
    parser.add_argument("-o", "--output_file", type=str, help="file name for saving the generated plot.")
    parser.add_argument("-r", "--dpi", type=int, help="resolution of image", default = 300)
    parser.add_argument("-lw", "--line_width", type=float, help="line_width", default = 1.0)

    # set variables based on arguments passed
    args = parser.parse_args()
    linewidth = args.line_width
    dpi = args.dpi
    output_file = args.output_file

    movie = yaml.load(open(args.mo))
    music = yaml.load(open(args.mu))
    music_walk = yaml.load(open(args.mw))

    plt.style.use('seaborn-whitegrid')
    plt.figure(dpi=dpi)

    labels = ['baseline', 'rf', 'logit']

    data = []
    data = [movie, music, music_walk]
    titles = ['Movie', 'Music', 'Music while walking']

    colors=['darkseagreen', 'plum', 'sandybrown']

    start = 2
    for i, (group, title) in enumerate(zip(data, titles)):

        baseline = np.array(group['baseline']['acc'])
        logit = np.array(group['logit']['acc'])
        rf = np.array(group['rf']['acc'])

        logit_lift = logit - baseline
        rf_lift = rf - baseline

        end = start + len(logit_lift)
        xaxis_range = range(start,end)
        plt.plot(xaxis_range, logit_lift, linestyle=' ', 
                    markerfacecolor=colors[i], marker='^', alpha=0.7,
                    markersize=13, label="Logit lift")
        plt.plot(xaxis_range, rf_lift, linestyle=' ', 
                    markerfacecolor=colors[i], marker='*', alpha=0.7,
                    markersize=13, label="RF lift")
        start = end + 10

    ax = plt.axes()
    ax.xaxis.grid(False)
    plt.xlim(0, 70)
    plt.ylim(-0.05, 0.57)
    plt.xticks((9, 34, 59), titles)
    plt.ylabel('User Lift')

    loc = plticker.MultipleLocator(base=0.05) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)

    ax_line = plt.axhline(0, color='black', lw=3, label='No Improvement')
    plt.title('User Lift of Personal Models Per Condition')

    star = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                          markersize=12, label='RF',
                          markerfacecolor='none', lw=4)
    triangle = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                          markersize=12, label='Logistic', 
                          markerfacecolor='none', lw=4)

    plt.legend(handles=[star, triangle, ax_line], loc=2, frameon=True, fontsize=12)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file + '.png', bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    main()
