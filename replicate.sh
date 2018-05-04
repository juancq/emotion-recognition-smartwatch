#----------------------------------------#
# CLASSIFICATION
#----------------------------------------#
# happy vs sad
python2 user_lift.py -mo features/features_mo_ew* -mu features/features_mu_ew* -mw features/features_mw_ew* -o acc

# happy vs sad vs neutral
python2 user_lift.py -mo features/features_mo_ew* -mu features/features_mu_ew* -mw features/features_mw_ew* -o neutral --neutral

# hold out one user
python2 lift_leave_one_user_out.py -mo features/features_mo_ew* -mu features/features_mu_ew* -mw features/features_mw_ew* -o loov_user

# hold out emotion
python2 lift_holdout_emotion.py -mo features/features_mo_ew* -mu features/features_mu_ew* -mw features/features_mw_ew* -o holdout


#----------------------------------------#
# BOXPLOTS 
#----------------------------------------#
python2 boxplot.py -mo mo_lift_scores_acc.yaml -mu mu_lift_scores_acc.yaml -mw mw_lift_scores_acc.yaml -o fig2

# improvement over baseline
python2 error_plot.py -mo mo_lift_scores_acc.yaml -mu mu_lift_scores_acc.yaml -mw mw_lift_scores_acc.yaml -o fig3

python2 boxplot.py -mo mo_lift_scores_neutral.yaml -mu mu_lift_scores_neutral.yaml -mw mw_lift_scores_neutral.yaml -o fig4

python2 boxplot.py -mo mo_lift_scores_holdout.yaml -mu mu_lift_scores_holdout.yaml -mw mw_lift_scores_holdout.yaml -o fig5

#----------------------------------------#
# PERMUTATION TEST
#----------------------------------------#

python2 permute_test.py -mo mo_lift_scores_loov_user.yaml -mu mu_lift_scores_loov_user.yaml -mw mw_lift_scores_loov_user.yaml

# all features
python2 permute_test.py -mo mo_lift_scores_acc.yaml -mu mu_lift_scores_acc.yaml -mw mw_lift_scores_acc.yaml

# neutral
python2 permute_test.py -mo mo_lift_scores_neutral.yaml -mu mu_lift_scores_neutral.yaml -mw mw_lift_scores_neutral.yaml

# emotion holdout
python2 permute_test.py -mo mo_lift_scores_holdout.yaml -mu mu_lift_scores_holdout.yaml -mw mw_lift_scores_holdout.yaml
