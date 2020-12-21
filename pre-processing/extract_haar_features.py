import json
import cv2
import os
import numpy as np

def define_haar_features(W, H):
  '''
  :param W: image width
  :param H: image height
  :return: extracted key coordinates of all Haar features (4 types)
        {'A': [[x1, y1, x4, y4], ...], ...}
  '''
  features = {'A': [], 'B': [], 'C': [], 'D': []}
  for x1 in range(W-1):
    for y1 in range(H-1):
      for x4 in range(x1+1, W):
        for y4 in range(y1+1, H):
          if (x4-x1) % 2 == 1:
            features['A'].append([x1, y1, x4, y4])
          if (y4-y1) % 2 == 1:
            features['B'].append([x1, y1, x4, y4])
          if (x4-x1) % 3 == 2:
            features['C'].append([x1, y1, x4, y4])
          if (x4-x1) % 2 == 1 and (y4-y1) % 2 == 1:
            features['D'].append([x1, y1, x4, y4])
  print("The number of type A Haar features is {0}".format(len(features['A'])))
  print("The number of type B Haar features is {0}".format(len(features['B'])))
  print("The number of type C Haar features is {0}".format(len(features['C'])))
  print("The number of type D Haar features is {0}".format(len(features['D'])))
  print("The total number of Haar features is {0}".format(len(features['A']) + len(features['B'])
                                                          + len(features['C']) + len(features['D'])))
  return features

def extract_haar_features_for_dataset(dir, features):
  def extract_haar_features_for_img(img, features):
    '''
    :param img: image with W*H pixels
    :param features: extracted key coordinates of all Haar features
    :return: a list of feature values for image
    '''
    # compute the sum of pixels for all rectangles (0, 0, x, y)
    w = img.shape[1]; h = img.shape[0]
    rec_sum = np.zeros((w+1, h+1)) # index range: 0-19
    for x in range(1, w+1):
      for y in range(1, h+1):
        rec_sum[x][y] = rec_sum[x-1][y] + rec_sum[x][y-1] - rec_sum[x-1][y-1] + img[y-1, x-1]

    # compute all the feature values for image
    img_feature = []
    # print(img)
    def compute_rec_sum(x1, y1, x4, y4):
      return rec_sum[x4+1][y4+1] + rec_sum[x1][y1] - rec_sum[x4+1][y1] - rec_sum[x1][y4+1]
    for x1, y1, x4, y4 in features['A']:
      midx = x1 + int((x4 - x1) / 2)
      left_rec = compute_rec_sum(x1, y1, midx, y4)
      right_rec = compute_rec_sum(midx+1, y1, x4, y4)
      img_feature.append(left_rec - right_rec)
    for x1, y1, x4, y4 in features['B']:
      midy = y1 + int((y4 - y1) / 2)
      bottom_rec = compute_rec_sum(x1, y1, x4, midy)
      top_rec = compute_rec_sum(x1, midy+1, x4, y4)
      img_feature.append(bottom_rec - top_rec)
    for x1, y1, x4, y4 in features['C']:
      midx1 = x1 + int((x4 - x1) / 3)
      midx2 = x4 - int((x4 - x1) / 3)
      left_rec = compute_rec_sum(x1, y1, midx1, y4)
      middle_rec = compute_rec_sum(midx1+1, y1, midx2-1, y4)
      right_rec = compute_rec_sum(midx2, y1, x4, y4)
      img_feature.append(left_rec + right_rec - middle_rec)
    for x1, y1, x4, y4 in features['D']:
      midx = x1 + int((x4 - x1) / 2)
      midy = y1 + int((y4 - y1) / 2)
      left_bottom_rec = compute_rec_sum(x1, y1, midx, midy)
      right_bottom_rec = compute_rec_sum(midx+1, y1, x4, midy)
      left_top_rec = compute_rec_sum(x1, midy+1, midx, y4)
      right_top_rec = compute_rec_sum(midx+1, midy+1, x4, y4)
      img_feature.append(left_top_rec + right_bottom_rec - left_bottom_rec - right_top_rec)
      # print(x1, y1, x4, y4, img_feature[-1])
    return img_feature

  dataset = []
  for face_img in os.listdir(dir + 'faces'):
    img = cv2.imread(dir + 'faces/' + face_img, 0)
    dataset.append([extract_haar_features_for_img(img, features), 1])
  print("The number of positive samples:", len(dataset))
  for nonface_img in os.listdir(dir + 'non-faces'):
    img = cv2.imread(dir + 'non-faces/' + nonface_img, 0)
    dataset.append([extract_haar_features_for_img(img, features), -1])
  print("The total number of samples:", len(dataset))
  return dataset

if __name__ == '__main__':
  features = define_haar_features(19, 19)
  with open('haar_feature_coordinates.json', 'w') as file:
    json.dump(features, file)

  # trainset_features = extract_haar_features_for_dataset('../VJ_dataset/trainset/', features)
  # with open('trainset_haar_features.json', 'w') as file:
  #   json.dump({'train': trainset_features}, file)
  testset_features = extract_haar_features_for_dataset('../VJ_dataset/testset/', features)
  with open('testset_haar_features.json', 'w') as file:
    json.dump({'test': testset_features}, file)

