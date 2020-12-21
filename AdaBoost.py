import numpy as np
import json
import time
import cv2

def AdaBoost(trainset, T = 10):
  def ERM_for_decision_stump(samples):
    # Parma samples: [[[x1, ...], y, d], ...]
    d = len(samples[0][0])
    m = len(samples)

    w = [sample[2] for sample in samples]
    print('{0}:{1}'.format(sum(w), w))

    min_F = 1; min_j = 0; min_theta = 0
    max_F = 0; max_j = 0; max_theta = 0
    for j in range(d):
      sorted_samples = sorted(samples, key=lambda sample: sample[0][j])
      last_sample_x = sorted_samples[-1][0].copy()
      last_sample_x[j] += 1
      sorted_samples.append([last_sample_x, 0, 0])
      # print(sorted_samples)
      F = sum([s[2] for s in samples if s[1] == 1])
      if F < min_F:
        min_F = F
        min_j = j
        min_theta = sorted_samples[0][0][j] - 1
      if F > max_F:
        max_F = F
        max_j = j
        max_theta = sorted_samples[0][0][j] - 1
      for i in range(m):
        F = F - sorted_samples[i][1] * sorted_samples[i][2]
        if F < min_F and sorted_samples[i][0][j] != sorted_samples[i + 1][0][j]:
          # print(F, j)
          min_F = F
          min_j = j
          min_theta = 1 / 2 * (sorted_samples[i][0][j] + sorted_samples[i + 1][0][j])
        if F > max_F and sorted_samples[i][0][j] != sorted_samples[i + 1][0][j]:
          max_F = F
          max_j = j
          max_theta = 1 / 2 * (sorted_samples[i][0][j] + sorted_samples[i + 1][0][j])
    polarization = 1
    if 1-max_F < min_F:
      min_F = 1-max_F
      min_j = max_j
      min_theta = max_theta
      polarization = -1
    print(min_j, min_theta, min_F, polarization)
    return min_j, min_theta, min_F, polarization

  # Initialize the weight distribution
  train_l = len([sample for sample in trainset if sample[1] == 1])
  train_m = len([sample for sample in trainset if sample[1] == -1])
  for idx in range(len(trainset)):
    if trainset[idx][1] == 1:
      trainset[idx].append(1 / (2 * train_l))
    else:
      trainset[idx].append(1 / (2 * train_m))

  classifiers = []
  for t in range(T):
    start = time.time()
    # Normalize the weight distribution
    norm = sum([sample[2] for sample in trainset])
    for idx in range(len(trainset)):
      trainset[idx][2] /= norm

    # Use ERM for decision stump to obtain the classifier
    min_j, min_theta, error, polarization = ERM_for_decision_stump(trainset)

    # Update the weight distribution
    beta = error / (1 - error)
    for idx in range(len(trainset)):
      if ((trainset[idx][0][min_j] - min_theta) * trainset[idx][1]) * polarization < 0:
        trainset[idx][2] *= beta
    classifiers.append([min_j, min_theta, beta, polarization])
    print("Running for the {0} round: {1}/s".format(t, time.time() - start))
  return classifiers

def obtain_coordinates_for_classifiers(classifiers, visualization = True):
  with open('pre-processing/haar_feature_coordinates.json', 'r') as file:
    coordinates = json.load(file)
  feature_type_idx = [0]
  for feature_type in ['A', 'B', 'C']:
    feature_type_idx.append(len(coordinates[feature_type]) + feature_type_idx[-1])

  feature_coordinates = []
  features = [cla[0] for cla in classifiers]
  for idx in features:
    if idx >= feature_type_idx[3]:
      feature_coordinates.append(coordinates['D'][idx - feature_type_idx[3]])
      print("Type D feature: {0}".format(coordinates['D'][idx - feature_type_idx[3]]))
    elif idx >= feature_type_idx[2]:
      feature_coordinates.append(coordinates['C'][idx - feature_type_idx[2]])
      print("Type C feature: {0}".format(coordinates['C'][idx - feature_type_idx[2]]))
    elif idx >= feature_type_idx[1]:
      feature_coordinates.append(coordinates['B'][idx - feature_type_idx[1]])
      print("Type B feature: {0}".format(coordinates['B'][idx - feature_type_idx[1]]))
    else:
      feature_coordinates.append(coordinates['A'][idx - feature_type_idx[0]])
      print("Type A feature: {0}".format(coordinates['A'][idx - feature_type_idx[0]]))

  if visualization:
    idx = 0
    for (x1, y1, x4, y4) in feature_coordinates:
      img = cv2.imread('VJ_dataset/trainset/faces/face00482.png', 0)
      cv2.rectangle(img, (x1, y1), (x4, y4), (255, 0, 0), 1)
      cv2.imshow('img', img)
      cv2.imwrite('results/' + str(idx) + '.jpg', img)
      cv2.waitKey(0)
      idx += 1
  return feature_coordinates

if __name__ == '__main__':
  '''
  1. Use AdaBoost to obtain the final classifier
  '''
  # with open('pre-processing/trainset_haar_features.json', 'r') as file:
  #   trainset = json.load(file)['train']
  # classifiers = AdaBoost(trainset)
  # with open('classifiers.json', 'w') as file:
  #   json.dump({'classifiers': classifiers}, file)

  '''
  2. Analyze the selected features 
  '''
  with open('classifiers.json', 'r') as file:
    classifiers = json.load(file)['classifiers']
  for cla in classifiers:
    print(cla)
  feature_coordinates = obtain_coordinates_for_classifiers(classifiers, visualization=True)





