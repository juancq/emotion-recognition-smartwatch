# Emotion-recognition using smart watch accelerometer data

This is the data and the source code used in the paper "Emotion-recognition using smart watch accelerometer data: preliminary findings." If you use our data set, please cite the following paper:

Juan C. Quiroz, Min Hooi Yong, and Elena Geangu. 2017. Emotion-recognition using smart watch accelerometer data: preliminary findings. In Proceedings of the 2017 ACM International Joint Conference on Pervasive and Ubiquitous Computing and Proceedings of the 2017 ACM International Symposium on Wearable Computers (UbiComp '17). ACM, New York, NY, USA, 805-812. DOI: https://doi.org/10.1145/3123024.3125614

## Commands used to generate results from our data set

Extract the accelerometer data from the recorded walking times.
```bash
python get_walking_data.py user_study_encoding.csv raw_data/ walking_data/
```

Extract features from sliding windows.
```bash
python extract_windows.py walking_data/m* features/
```

Compute classification accuracies of happy vs sad.
```bash
python user_lift.py -mo features/features_mo_ew* -mu features/features_mu_ew* -mw features/features_mw_ew* -o acc_f1
```

Compute classification accuracies of happy vs sad vs neutral.
```bash
python2 user_lift.py -mo features/features_mo_ew* -mu features/features_mu_ew* -mw features/features_mw_ew* -o neutral --neutral
```

Run permutation test to determine if accuracies are higher than baseline.
```bash
python permute_test.py -mo mo_lift_scores_log.yaml -mu mu_lift_scores_log.yaml -mw mw_lift_scores_log.yaml
```

Generate plot of feature importances.
```bash
python feature_importance_plot.py -mo mo_feature_import_acc_f1.yaml -mu mu_feature_import_acc_f1.yaml -mw mw_feature_import_acc_f1.yaml
```
