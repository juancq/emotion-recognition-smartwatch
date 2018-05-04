import argparse
import yaml
import numpy as np
from collections import defaultdict

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


def main():
    '''
    Compute leave-one-user-out cross-validation.
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
        user_ids = []
        # these store the CV results for each personal model
        scores = {key:defaultdict(list) for key in ['baseline', 'logit', 'rf']}

        user_data = []
        user_index = []
        start = 0
        for i,fname in enumerate(fnames):
            user_i = np.loadtxt(fname, delimiter=',')
            # delete neutral to see if we can distinguish between
            # happy/sad
            user_i = np.delete(user_i, np.where(user_i[:,-1]==0), axis=0)

            user_data.append(user_i)
            end = start + user_i.shape[0]
            user_index.append((start, end))
            start = end

        data = np.concatenate(user_data)
        x_data = data[:,:-1]
        y_data = data[:,-1]

        num_users = len(fnames)

        for i in range(num_users):
            print 'validating user ', i
            start, end = user_index[i]
            x_test = x_data[start:end]
            y_test = y_data[start:end]

            train_index = range(0, start) + range(end, data.shape[0])
            x_train = x_data[train_index]
            y_train = y_data[train_index]

            models = [
                    ('baseline', DummyClassifier(strategy = 'most_frequent')),
                    #('logit', linear_model.LogisticRegressionCV(Cs=20, cv=10)),
                    ('logit', linear_model.LogisticRegression()),
                    ('rf', RandomForestClassifier(n_estimators = N_ESTIMATORS)),
                    ]

            for key, clf in models:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                _f1 = metrics.f1_score(y_test, y_pred, average='weighted')
                _acc = metrics.accuracy_score(y_test, y_pred)
                y_proba = clf.predict_proba(x_test)
                _roc_auc = metrics.roc_auc_score(y_test, y_proba[:, 1])

                scores[key]['f1'].append(_f1)
                scores[key]['acc'].append(_acc)
                scores[key]['roc_auc'].append(_roc_auc)


        print 'model\t\tacc\tf1\troc_auc'
        for key, scores in scores.iteritems():
            print '{}:\t\t{:.3f} ({:.3f})\t{:.3f} ({:.3f})\t{:.3f} ({:.3f})'.format(
                    key, np.mean(scores['acc']), np.std(scores['acc']),
                    np.mean(scores['f1']), np.std(scores['f1']),
                    np.mean(scores['roc_auc']), np.std(scores['roc_auc']))

        results = {'labels': user_ids, 'scores': scores}

        yaml.dump(results, open(condition+'_louo_lift_scores_'+output_file+'.yaml', 'w'))

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
