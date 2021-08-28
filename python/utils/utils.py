from __future__ import absolute_import
from collections import Counter
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


# https://github.com/scikit-optimize/scikit-optimize/issues/674
class TQDMCallback(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()


def get_f1(x, labels=None):
    if labels is None:
        raise ValueError
    return precision_recall_fscore_support(x[:, 0], x[:, 1], labels=labels, average='macro', zero_division=0)[2]


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority/count) for cls, count in counter.items()}


# https://towardsdatascience.com/introduction-to-topic-modeling-using-scikit-learn-4c3f3290f5b9
def get_model_topics(model, vectorizer, topics, n_top_words):
    word_dict = {}
    feature_names = vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        word_dict[topics[topic_idx]] = top_features
    return pd.DataFrame(word_dict)


# https://towardsdatascience.com/introduction-to-topic-modeling-using-scikit-learn-4c3f3290f5b9
def get_inference(model, vectorizer, topics, text, threshold):
    v_text = vectorizer.transform([text])
    score = model.transform(v_text)
    labels = set()
    for i in range(len(score[0])):
        if score[0][i] > threshold:
            labels.add(topics[i])
    if not labels:
        return 'None', -1, set()
    return topics[np.argmax(score)], score, labels


def unique_str(tmp, remove_nan=False):
    tmp = np.array(list(set(tmp)))
    if remove_nan:
        tmp = [x for x in tmp if str(x) != 'nan']
    return tmp


def remove_short_words(x, n=1):
    return re.sub(r'\b\w{1,n}\b', '', x)


def to_categories(x):
    x = np.array(x)
    x[x == "N"] = "0"
    x[x == "Y"] = "1"
    x = x.astype(int)
    return x


