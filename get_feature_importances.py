import argparse
import yaml
import numpy as np

from collections import defaultdict
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from permute.core import one_sample

SEED = 1
np.random.seed(SEED)

def main():
    '''
    Train a model and then compile feature importances.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-mu", metavar='mu', type=str, nargs='+', help="file containing music features, input to model", default=[])
    parser.add_argument("-mw", metavar='mw', type=str, nargs='+', help="file containing music+walking features, input to model", default=[])
    parser.add_argument("-mo", metavar='mo', type=str, nargs='+', help="file containing movie features, input to model", default=[])
    parser.add_argument("-e", "--estimators", help="number of estimators for meta-classifiers", type=int, default=100)
    parser.add_argument("-o", "--output_file", help="output with pickle results", type=str)
    parser.add_argument("--neutral", action='store_true', help="classify happy-sad-neutral")
    args = parser.parse_args()
    output_file = args.output_file
    N_ESTIMATORS = args.estimators
    neutral = args.neutral


    def process_condition(fnames, condition):

        print 'condition', condition

        results = defaultdict(list)

        for fname in fnames:
            print 'classifying: %s' % fname
            label = fname.split('/')[-1]

            data = np.loadtxt(fname, delimiter=',')

            #acc only
            #data = np.hstack([data[:,:51], data[:,-1].reshape(data.shape[0], 1)])
            # acc features + heart rate + y label
            #data = np.hstack([data[:,:51], data[:,-2:]])

            print data.shape

            if not neutral:
                # delete neutral to see if we can distinguish between
                # happy/sad
                data = np.delete(data, np.where(data[:,-1]==0), axis=0)

            np.random.shuffle(data)

            x_data = data[:,:-1]
            y_data = data[:,-1]

            # scaled
            x_data = preprocessing.scale(x_data)

            models = [
                    #('logit', linear_model.LogisticRegressionCV(Cs=20, cv=10)),
                    ('rf', RandomForestClassifier(n_estimators = N_ESTIMATORS)),
                    ]
                    
            results['labels'].append(label)
            repeats = 2
            folds = 10
            rskf = RepeatedStratifiedKFold(n_splits=folds, 
                                        n_repeats=repeats,
                                        random_state=SEED)

            for key, clf in models:
                feature_importances = []
                for train,test in rskf.split(x_data, y_data):
                    x_train, x_test = x_data[train], x_data[test]
                    y_train, y_test = y_data[train], y_data[test]
                    clf.fit(x_train, y_train)
                    if key == 'rf':
                        feature_importances.append(clf.feature_importances_)
                    else:
                        feature_importances.append(clf.coef_)

                results[key].append(feature_importances)


        yaml.dump(results, open(condition+'_feature_import_'+output_file+'.yaml', 'w'))
    # end of function
    #---------

    process_condition(args.mu, 'mu')
    process_condition(args.mw, 'mw')
    process_condition(args.mo, 'mo')


if __name__ == "__main__":
    main()
