# emotion-recognition-walking-acc


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
python fi_plot.py -mo mo_feature_import_acc_f1.yaml
```
