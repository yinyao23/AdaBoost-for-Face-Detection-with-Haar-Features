# AdaBoost-for-Face-Detection-with-Haar-Features 

Run this project in the following 4 steps.

## Add the "dataset" folder under the main directory "Adaboost"

## Pre-process: Run the "extract_haar_features.py" in "pre-processing" folder to generate 3 json files. 

Detail: the haar features are defined as the same as the original paper "Rapid Object Detection using a Boosted Cascade of Simple Features". The 3 json files are "haar_feature_coordinates.json", "testset_haar_features.json" and "trainset_haar_features.json", which will be utilized in the next training process.

## Run the "train.py" to obtain the 10 decision stump classifiers. 

Detail: In each training round of AdaBoost, a simple classifier (haar feature) is selected by optimizing the ERM for decision stumps. 

The "classifiers.json" file will be generated to record the coordinates of the simple classifiers. And the "results" folder will be generated to visualize the extracted haar features on a test image.

## Run the "evaluate.py" to evaluate the combined classifier on the test dataset and draw the ROC curve.

Detail: The combined classifiers with top [1, 3, 5, 10] features will be evaluated by their ROC curve.

