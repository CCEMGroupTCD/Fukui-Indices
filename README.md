# Fukui Index Prediction and Selectivity Classification
This repo is for a project together with Manting and her experimental collaborator. We are predicting the Fukui indices via regression and classifying whether a certain atomic site is the one with the highest Fukui index/selectivity or not. 
Main aspects are:
    - SOAP features
        - RF classifier works best with full SOAP features
        - PCA does not improve performance
    - Predict classifier probability:
        - Because we predict individual atomic sites, we are predicting the probability of this site to be the one with the highest Fukui index/selectivity using a classifier and the predict_proba() method. That way, we can later calculate the score of the model by comparing if the predicted site is the one with the highest Fukui index/selectivity. This is implemented in this code in the function accuracy_of_highest_probability_in_group().