import re
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow_addons.metrics import F1Score
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

sys.path.append(os.path.abspath("../utils/"))
from utils import get_class_weights, get_model_topics, get_inference, unique_str, to_categories
from losses import categorical_focal_loss
from metrics import f1_m, precision_m, recall_m, fbeta_score_macro


# PARAMS
n_min_note_length = 30
n_features = 2000
# n_components = 4
n_top_words = 50
n_iter = 30  # 30
n_min_words = 5
n_max_words = 20
n_remove_samples_end = 2
stop_words = ["pasient", "pasienter", "pasienten"]  # None

# set seed for session
n_seed = 42
np.random.seed(n_seed)


raw_data_path = "../../data/2020_03_04_Uttrekk_kateter_fra_2015_uten_id.csv"
annotated_data_path = "../../data/AE_annotated_Labeled_Hel_and_Manual_20210604.csv"
save_model_path = "../../output/models/"
history_path = "../../output/history/"
datasets_path = "../../output/datasets/"

data_raw = pd.read_csv(raw_data_path)
annotated_raw = pd.read_csv(annotated_data_path)

# remove nan-initialized samples from annotated data
annotated_raw = annotated_raw[:-n_remove_samples_end]

print(data_raw.head())
print(annotated_raw.head())

# extract relevant train data
data_raw_title = data_raw["Tittel"]
data_raw_notes = data_raw["Hendelsesbeskrivelse"]

# extract relevant test data
annotated_raw_notes = annotated_raw["content"]
ids = annotated_raw["filename"]

# extract relevant GT from test data (manually labelled, at note-level)
infections = annotated_raw["manuelt_merket_Infeksjonsrelatert"]
device_failures = annotated_raw["manuelt_merket_Feil_pa_enheten"]
fallens = annotated_raw["manuelt_merket_falt"]
mistake_unit = annotated_raw["Feil_pa_enheten"]
catheters = annotated_raw["Kateterrelatert"]
sepsis = annotated_raw["Sepsis"]
pvks = annotated_raw["PVK_relatert"]

infections = to_categories(infections)
device_failures = to_categories(device_failures)
fallens = to_categories(fallens)
mistake_unit = to_categories(mistake_unit)
catheters = to_categories(catheters)
sepsis = to_categories(sepsis)
pvks = to_categories(pvks)

print(np.histogram(infections, 2))
print(np.histogram(device_failures, 2))
print(np.histogram(fallens, 2))
print(np.histogram(mistake_unit, 2))
print(np.histogram(catheters, 2))
print(np.histogram(sepsis, 2))
print(np.histogram(sepsis, 2))

unique_ids = unique_str(ids, remove_nan=True)

for id_ in unique_ids:
    tmp = annotated_raw[ids == id_]["manuelt_merket_Infeksjonsrelatert"]
    print(tmp)

exit()

unique_annotated_raw_notes = unique_str(annotated_raw_notes, remove_nan=True)

# preprocess: 1) remove or substitute/impute sensitive data based on removing unique words/elements in notes

# produce shuffled corpus for training models
corpus = np.asarray(data_raw_notes)
np.random.shuffle(corpus)

print(corpus)
print(len(corpus))
new = []
for x in corpus:
    # print(x)
    print(x)
    # if len(x) < 10:
    #   print(x)
    #   exit()
    if isinstance(x, str):
        if (x != "Se vedlegg") and (len(x) > n_min_note_length):
            new.append(x)
    else:
        print(np.isnan(x))
    print("\n")
    print("#" * 50)

corpus = new.copy()
del new

# remove redundant topics (I'm lazy, hopefully my topic analysis model will handle this)
removes = ["\n", "Hele_Notater"]
for i, c in enumerate(corpus):
    #print(i, c)
    for r in removes:
        c = c.replace(r, " ")
    c.strip()
    corpus[i] = c
    #print(i, c)
    #print()

#print(corpus)
#exit()

# topics of relevance for Latent Direchlet Allocation (LDA)
lda_topics = ["woman.birth", "admission", "person.fallen", "disease.infection", "device.failure", "misc", "redundant", "redness"]  # , "redundant", "Misc"]

n_components = len(lda_topics)

pattern = r'\b\w{' + re.escape(str(n_min_words)) + ',' + re.escape(str(n_max_words)) + r'}\b'
print(pattern)

# tokenizers
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=stop_words, ngram_range=(1, 2),
                                lowercase=True, strip_accents="unicode", analyzer="word",
                                token_pattern=pattern)
tf_out = tf_vectorizer.fit_transform(corpus)

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=stop_words, ngram_range=(1, 2),
                                   lowercase=True, strip_accents="unicode", analyzer="word",
                                   token_pattern=pattern)
tfidf_out = tfidf_vectorizer.fit_transform(corpus)  # r'\b\w{1,n}\b'

# LDA
print("Performing LDA: ")
lda_model = LatentDirichletAllocation(n_components=n_components, random_state=n_seed, verbose=1,
                                      n_jobs=-1, max_iter=n_iter).fit(tf_out)


'''
### EVAL ON TRAIN
print("\nEVAL ON TRAIN SET: ")
preds = {key: [] for key in lda_topics}
for text in corpus:
    #print("\nCurrent line: ")
    #print(text)
    topic, score, _ = get_inference(lda_model, tf_vectorizer, lda_topics, text, 0)
    #print(topic, score)
    preds[topic].append(text)

print(get_model_topics(lda_model, tf_vectorizer, lda_topics, n_top_words))
'''

#'''
### EVAL ON TEST
print("\nEVAL ON TEST SET: ")
# Display some predictions/results to interpret model performance
preds = {key: [] for key in lda_topics}
for text in unique_annotated_raw_notes:
    #print("\nCurrent line: ")
    #print(text)
    topic, score, _ = get_inference(lda_model, tf_vectorizer, lda_topics, text, 0)
    #print(topic, score)
    preds[topic].append(text)
print(get_model_topics(lda_model, tf_vectorizer, lda_topics, n_top_words))
#'''

'''
for key in list(preds.keys()):
    print("Current key:", key)
    print(preds[key])
    print("\n" * 2)
'''

## plot word clouds for the different groups
# https://www.datacamp.com/community/tutorials/wordcloud-python
# https://stackoverflow.com/questions/60790721/topic-modeling-run-lda-in-sklearn-how-to-compute-the-wordcloud
# https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud

# define vocabulary to get words names
vocab = tf_vectorizer.get_feature_names()

words = {}
freqs = {}
n_top_words = 20
n_plot_horz = 4

for topic, component in enumerate(lda_model.components_):

    # need [::-1] to sort the array in descending order
    indices = np.argsort(component)[::-1][:n_top_words]

    # store the words most relevant to the topic
    words[topic] = {vocab[i]: component[i] for i in indices}

# lower max_font_size, change the maximum number of word and lighten the background:
fig, ax = plt.subplots(int(np.ceil(n_components / n_plot_horz)), n_plot_horz)
plt.tight_layout()
for i, (key, value) in enumerate(words.items()):
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(value)
    ax[int(i / n_plot_horz), int(i % n_plot_horz)].imshow(wordcloud, interpolation="bilinear")
    ax[int(i / n_plot_horz), int(i % n_plot_horz)].axis("off")
plt.show()


## evaluate the unsupervised LDA model for classification based on a selection of cases






