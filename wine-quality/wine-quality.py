# more detail at https://github.com/booleanhunter/ML-supervised-learning/blob/master/game-of-wines/visuals.py
# and at https://medium.freecodecamp.org/using-machine-learning-to-predict-the-quality-of-wines-9e2e13d7480d

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from time import time
plt.rcParams.update({'font.size': 8})

def train_predict_evaluate(learner, sample_size, X_train, y_train, X_test, y_test):
    results = {}

    start = time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    results['train_time'] = end - start

    start = time()
    predictions_train = learner.predict(X_train[:300])
    predictions_test = learner.predict(X_test)
    end = time()

    results['pred_time'] = end - start

    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5, average='micro')
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5, average='micro')

    print("{} trained on {} samples".format(learner.__class__.__name__, sample_size))

    return results

def visualize_classification_performance(results):
    """
    Visualization code to display results of various learners.

    inputs:
      - results: a list of dictionaries of the statistic results from 'train_predict_evaluate()'
    """

    # Create figure
    sns.set()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(2, 3, figsize = (12,8.5))
    # print("VERSION:")
    # print(matplotlib.__version__)
    # Constants
    bar_width = 0.3
    colors = ["#e55547", "#4e6e8e", "#2ecc71"]

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):

                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size", fontsize=8)
                ax[j//3, j%3].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)", fontsize=8)
    ax[0, 1].set_ylabel("Accuracy Score", fontsize=8)
    ax[0, 2].set_ylabel("F-score", fontsize=8)
    ax[1, 0].set_ylabel("Time (in seconds)", fontsize=8)
    ax[1, 1].set_ylabel("Accuracy Score", fontsize=8)
    ax[1, 2].set_ylabel("F-score", fontsize=8)

    # Add titles
    ax[0, 0].set_title("Model Training", fontsize=8)
    ax[0, 1].set_title("Accuracy Score on Training Subset", fontsize=8)
    ax[0, 2].set_title("F-score on Training Subset", fontsize=8)
    ax[1, 0].set_title("Model Predicting", fontsize=8)
    ax[1, 1].set_title("Accuracy Score on Testing Set", fontsize=8)
    ax[1, 2].set_title("F-score on Testing Set", fontsize=8)

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = 1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = 1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = 1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = 1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')

    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    #plt.tight_layout(pad=1, w_pad=2, h_pad=5.0)
    plt.savefig('./classifiers-comparison.png', dpi=250)
    #plt.show()
    plt.clf()
    plt.close()

def feature_plot(importances, X_train, y_train):

    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:11]]
    values = importances[indices][:11]

    sns.set()
    sns.set_style("whitegrid")

    # Creat the plot
    fig = plt.figure(figsize = (12,5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(11), values, width = 0.2, align="center", label = "Feature Weight")
    # plt.bar(np.arange(11) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
    #       label = "Cumulative Feature Weight")
    plt.xticks(np.arange(11), columns)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)

    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.savefig('./feature-importances.png', dpi=250)
    #plt.show()
    plt.clf()
    plt.close()

####################################################################################################################################################################################################
############################################################################################## MAIN ################################################################################################
####################################################################################################################################################################################################

data = pd.read_csv('./input-wine-quality/winequality_red.csv')

#Defining the splits for categories. 1–4 will be poor quality, 5–6 will be average, 7–10 will be great
bins = [1,4,6,10]
#0 for low quality, 1 for average, 2 for great quality
quality_labels=[0,1,2]
data['quality_categorical'] = pd.cut(data['quality'], bins=bins, labels=quality_labels, include_lowest=True)
#Displays the first 2 columns
data.head(n=2)
# Split the data into features and target label
quality_raw = data['quality_categorical']
features_raw = data.drop(['quality', 'quality_categorical'], axis = 1)

# Import train_test_split
from sklearn.model_selection import train_test_split
# Import any three supervised learning classification models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#Import metrics
from sklearn.metrics import fbeta_score, accuracy_score

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_raw,
 quality_raw,
 test_size = 0.2,
 random_state = 0)
# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(max_depth=None, random_state=None)
clf_C = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=None)
# Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100
# HINT: samples_1 is 1% of samples_100
samples_100 = len(y_train)
samples_10 = int(len(y_train)*10/100)
samples_1 = int(len(y_train)*1/100)
# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict_evaluate(clf, samples, X_train, y_train, X_test, y_test)
#print(results)
# Run metrics visualization for the three supervised learning models chosen
visualize_classification_performance(results)

# Import a supervised learning model that has 'feature_importances_'
model = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=None)
# Train the supervised model on the training set using .fit(X_train, y_train)
model = model.fit(X_train, y_train)
# Extract the feature importances using .feature_importances_
importances = model.feature_importances_
print(X_train.columns)
print(importances)
# Plot
feature_plot(importances, X_train, y_train)
