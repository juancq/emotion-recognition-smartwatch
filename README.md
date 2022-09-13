# Emotion-recognition using smart watch sensor data

This is the data and the source code used in the paper ["Emotion Recognition Using Smart Watch Sensor Data: Mixed-Design Study"](http://doi.org/10.2196/10153) published in [JMIR Mental Health](https://mental.jmir.org/). If you use our code or dataset, cite the following paper: 

```
Quiroz JC, Geangu E, Yong MH
Emotion Recognition Using Smart Watch Sensor Data: Mixed-Design Study
JMIR Ment Health 2018;5(3):e10153
URL: https://mental.jmir.org/2018/3/e10153
DOI: 10.2196/10153
PMID: 30089610
```

Our preliminary results were published in:
```
Juan C. Quiroz, Min Hooi Yong, and Elena Geangu. 2017. 
Emotion-recognition using smart watch accelerometer data: preliminary findings. 
In Proceedings of the 2017 ACM International Joint Conference on Pervasive and 
Ubiquitous Computing and Proceedings of the 2017 ACM International Symposium 
on Wearable Computers (UbiComp '17). ACM, New York, NY, USA, 805-812. 
DOI: https://doi.org/10.1145/3123024.3125614
```

## Dataset
Data was collected from 50 participants. Coding details are available in user_study_encoding.csv. 



## Replicating Study Results

### Install requirements

The code was written in python2.7. To install the requiements run:
```bash
pip install -r requirements.txt
```

### Commands used to generate results from our data set

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
