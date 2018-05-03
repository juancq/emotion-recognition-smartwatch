import argparse
import math
import yaml
import numpy as np
from collections import defaultdict

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from permute.core import one_sample


def main():
    '''
    Computes cross-validation by holding out a contiguous block of windows from a single emotion.
    '''
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("-mu", metavar='mu', type=str, nargs='+', help="file containing music features, input to model", default=[])
    parser.add_argument("-mw", metavar='mw', type=str, nargs='+', help="file containing music+walking features, input to model", default=[])
    parser.add_argument("-mo", metavar='mo', type=str, nargs='+', help="file containing movie features, input to model", default=[])
    parser.add_argument("-e", "--estimators", help="number of estimators for meta-classifiers", type=int, default=100)
    parser.add_argument("-o", "--output_file", help="output with pickle results", type=str)
    args = parser.parse_args()
    output_file = args.output_file
    N_ESTIMATORS = args.estimators

    def process_condition(fnames, condition):

        print 'condition', condition

        results = {'labels':[], 'baseline': defaultdict(list),
                    'logit': defaultdict(list), 
                    'rf': defaultdict(list)}

        folds = 10

        for fname in fnames:
            print 'classifying: %s' % fname
            label = fname.split('/')[-1]

            data = np.loadtxt(fname, delimiter=',')

            # delete neutral to see if we can distinguish between
            # happy/sad
            data = np.delete(data, np.where(data[:,-1]==0), axis=0)

            group_a = np.where(data[:,-1]==1)
            group_b = np.where(data[:,-1]==-1)

            a_folds = math.floor(folds/2.)
            b_folds = folds - a_folds

            split_groups = []
            split_groups.extend(np.array_split(group_a[0], a_folds))
            split_groups.extend(np.array_split(group_b[0], b_folds))

            k_folds = []

            for i in range(folds):
                test = split_groups[i]
                train = np.concatenate((split_groups[:i] + split_groups[i+1:]))
                k_folds.append((train, test))

            x_data = data[:,:-1]
            y_data = data[:,-1]

            # scaled
            x_data = preprocessing.scale(x_data)

            models = [
                    ('baseline', DummyClassifier(strategy = 'most_frequent')),
                    #('logit', linear_model.LogisticRegressionCV(Cs=20, cv=10)),
                    ('logit', linear_model.LogisticRegression()),
                    ('rf', RandomForestClassifier(n_estimators = N_ESTIMATORS)),
                    ]

            results['labels'].append(label)

            # roc_auc generates error because test includes single class
            for key, clf in models:
                scores = {'f1':[], 'acc':[]}

                for (train, test) in k_folds:
                    x_train, x_test = x_data[train], x_data[test]
                    y_train, y_test = y_data[train], y_data[test]

                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
                    _f1 = metrics.f1_score(y_test, y_pred, average='weighted')
                    _acc = metrics.accuracy_score(y_test, y_pred)
                    y_proba = clf.predict_proba(x_test)
                    scores['f1'].append(_f1)
                    scores['acc'].append(_acc)


                results[key]['f1'].append(np.mean(scores['f1']))
                results[key]['acc'].append(np.mean(scores['acc']))

        yaml.dump(results, open(condition+'_lift_scores_'+output_file+'.yaml', 'w'))

    # end of function
    #---------
    if args.mu:
        process_condition(args.mu, 'mu')
    if args.mw:
        process_condition(args.mw, 'mw')
    if args.mo:
        process_condition(args.mo, 'mo')

if __name__ == "__main__":
    main()
