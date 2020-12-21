import json
import math
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import numpy as np

def evaluate(testset, classifiers):
  for t in range(len(classifiers)):
    classifiers[t][2] = math.log(1 / classifiers[t][2])
  print("Coefficients for classifiers: {0}".format([cla[2] for cla in classifiers]))

  predictions = []
  for sample in testset:
    pres = []
    for h in classifiers:
      if h[3] == 1:
        pres.append(int(sample[0][h[0]] <= h[1]) * h[2])
      else:
        pres.append(int(sample[0][h[0]] > h[1]) * h[2])
    predictions.append(pres)
  print(np.array(predictions).shape, predictions)

  for t in [1, 3, 5, 10]:
    baseline = sum([h[2] for h in classifiers[:t]]) / 2
    results = [sum(pres[:t]) - baseline for pres in predictions]
    targets = [sample[1] for sample in testset]

    res = [int(r >= 0) for r in results]
    acc = [int(res[i] == targets[i]) for i in range(len(res)) if res[i] == 1]
    print(np.mean(np.array(acc)), acc)
    print("The accuracy of the final predictor is: {0}".format(np.mean(np.array(acc))))

    fpr, tpr, _ = roc_curve(targets, results)
    # print(fpr, tpr)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
  with open('pre-processing/trainset_haar_features.json', 'r') as file:
    testset = json.load(file)['train']
  # with open('pre-processing/testset_haar_features.json', 'r') as file:
  #   testset = json.load(file)['test']
  with open('classifiers.json', 'r') as file:
    classifiers = json.load(file)['classifiers']
  evaluate(testset, classifiers)