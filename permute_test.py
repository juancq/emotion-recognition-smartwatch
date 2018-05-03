import argparse
import yaml
import numpy as np

from permute.core import one_sample

def permute(results):
    baseline = np.array(results['baseline']['acc'])
    logit = np.array(results['logit']['acc'])
    rf = np.array(results['rf']['acc'])

    baseline_f1 = np.array(results['baseline']['f1'])
    logit_f1 = np.array(results['logit']['f1'])
    rf_f1 = np.array(results['rf']['f1'])

    baseline_roc = np.array(results['baseline']['roc_auc'])
    logit_roc = np.array(results['logit']['roc_auc'])
    rf_roc = np.array(results['rf']['roc_auc'])

    logit_lift = logit - baseline

    print('model\t\tAUC\t\tF1\t\tmean\t\tdiff\tp-value')
    print('baseline\t{roc:.3f} ({roc_std:.3f})\t{f1:.3f} ({f1_std:.3f})\t{acc:.3f} ({acc_std:.3f})'.format(
                    roc=baseline_roc.mean(), roc_std=baseline_roc.std(),
                    f1=baseline_f1.mean(), f1_std=baseline_f1.std(),
                    acc=baseline.mean(), acc_std=baseline.std()))

    (p, diff_means) = one_sample(logit_lift, stat='mean')
    print('Logit\t\t{roc:.3f} ({roc_std:.3f})\t{f1:.3f} ({f1_std:.3f})\t{acc:.3f} ({acc_std:.3f})\t{diff:.3f}\t{p:.3f}'.format(
            roc=logit_roc.mean(), roc_std=logit_roc.std(),
            f1=logit_f1.mean(), f1_std=logit_f1.std(),
            acc=logit.mean(), acc_std=logit.std(),
            diff=diff_means, p=p))
    
    rf_lift = rf - baseline
    (p, diff_means) = one_sample(rf_lift, stat='mean')
    print('RF\t\t{roc:.3f} ({roc_std:.3f})\t{f1:.3f} ({f1_std:.3f})\t{acc:.3f} ({acc_std:.3f})\t{diff:.3f}\t{p:.3f}'.format(
            roc=rf_roc.mean(), roc_std=rf_roc.std(),
            f1=rf_f1.mean(), f1_std=rf_f1.std(),
            acc=rf.mean(), acc_std=rf.std(),
            diff=diff_means, p=p))


def main():
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("-mo", type=str, help="movie yaml")
    parser.add_argument("-mu", type=str, help="music yaml")
    parser.add_argument("-mw", type=str, help="music+walk yaml")
    args = parser.parse_args()

    if args.mo:
        movie_results = yaml.load(open(args.mo))
        print '^' * 20
        print 'Movie'
        permute(movie_results)

    if args.mu:
        music_results = yaml.load(open(args.mu))
        print '^' * 20
        print 'Music'
        permute(music_results)

    if args.mw:
        music_walk_results = yaml.load(open(args.mw))
        print '^' * 20
        print 'Music + walking'
        permute(music_walk_results)


if __name__ == "__main__":
    main()
