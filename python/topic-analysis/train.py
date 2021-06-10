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
import configparser

print(sys.path)
sys.path.append(os.path.abspath(os.getcwd() + "/utils/"))
# sys.path.append(os.path.abspath(os.getcwd() + "../"))
print(sys.path)
from utils import get_class_weights, get_model_topics, get_inference, unique_str, to_categories
from losses import categorical_focal_loss
from metrics import f1_m, precision_m, recall_m, fbeta_score_macro

config = configparser.ConfigParser()
config.read(sys.argv[1])

print(sys.argv[1])
# exit()

# PARAMS
'''
n_min_note_length = 30
n_features = 2000
# n_components = 4
n_top_words = 50
n_iter = 30  # 30
n_min_words = 5
n_max_words = 20
n_remove_samples_end = 2
stop_words = ["pasient", "pasienter", "pasienten"]  # None

n_wc_top_words = 20
n_wc_plot_horz = 4

alg = "randomized"
lower_flag = True

n_jobs = -1
verbose = 1

tol = 0.0

# set seed for session
n_seed = 42
np.random.seed(n_seed)
'''

n_seed = int(config["Analysis"]["n_seed"])

# print(os.listdir("."))

raw_data_path = os.getcwd() + "/../data/2020_03_04_Uttrekk_kateter_fra_2015_uten_id.csv"
annotated_data_path = os.getcwd() + "/../data/AE_annotated_Labeled_Hel_and_Manual_20210604.csv"
save_model_path = os.getcwd() + "/../output/models/"
history_path = os.getcwd() + "/../output/history/"
datasets_path = os.getcwd() + "/../output/datasets/"

data_raw = pd.read_csv(raw_data_path)
annotated_raw = pd.read_csv(annotated_data_path)

# remove nan-initialized samples from annotated data
annotated_raw = annotated_raw[:-int(config["Preprocessing"]["n_remove_samples_end"])]

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

#for id_ in unique_ids:
#    tmp = annotated_raw[ids == id_]["manuelt_merket_Infeksjonsrelatert"]
#    print(tmp)

#exit()

unique_annotated_raw_notes = unique_str(annotated_raw_notes, remove_nan=True)

# preprocess: 1) remove or substitute/impute sensitive data based on removing unique words/elements in notes

# produce shuffled corpus for training models
corpus = np.asarray(data_raw_notes)
np.random.shuffle(corpus)

#print(corpus)
#print(len(corpus))
new = []
for x in corpus:
    if isinstance(x, str):
        if (x != "Se vedlegg") and (len(x) > int(config["Preprocessing"]["n_min_note_length"])):
            new.append(x)
    else:
        print(np.isnan(x))

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
#lda_topics = ["woman.birth", "admission", "person.fallen", "disease.infection", "device.failure", "misc", "redundant", "redness"]  # , "redundant", "Misc"]
lda_topics = list(range(int(config["Topic analysis"]["n_components"])))
n_components = len(lda_topics)

pattern = r'\b\w{' + re.escape(str(int(config["Preprocessing"]["n_min_words"]))) + ',' + re.escape(str(int(config["Preprocessing"]["n_max_words"]))) + r'}\b'
print(pattern)

# tokenizers
token = config["Tokenizer"]["token"]
if token == "count":
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=int(config["Tokenizer"]["n_features"]), stop_words=tuple(config["Tokenizer"]["stop_words"]), ngram_range=tuple([int(x) for x in config["Tokenizer"]["n_grams"].split(",")]),
                                    lowercase=eval(config["Tokenizer"]["lower_flag"]), strip_accents="unicode", analyzer="word",
                                    token_pattern=pattern)
elif token == "tfidf":
    tf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=int(config["Tokenizer"]["n_features"]), stop_words=tuple(config["Tokenizer"]["stop_words"]), ngram_range=tuple([int(x) for x in config["Tokenizer"]["n_grams"].split(",")]),
                                   lowercase=eval(config["Tokenizer"]["lower_flag"]), strip_accents="unicode", analyzer="word",
                                   token_pattern=pattern)
else:
    print("Unknown tokenizer was defined.")
    exit()
tf_out = tf_vectorizer.fit_transform(corpus)


# LDA
method = config["Topic analysis"]["method"]
if method == "LDA":
    print("Performing LDA: ")
    model = LatentDirichletAllocation(n_components=n_components, random_state=n_seed, verbose=eval(config["LDA"]["verbose"]),
                                          n_jobs=int(config["LDA"]["n_jobs"]), max_iter=int(config["Topic analysis"]["n_iter"])).fit(tf_out)
elif method == "LSA":
    model = TruncatedSVD(n_components=n_components, random_state=eval(config["Analysis"]["n_seed"]), algorithm=config["SVD"]["alg"])
else:
    print("Unknown topic analysis method was defined.")
    exit()


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
    topic, score, _ = get_inference(model, tf_vectorizer, lda_topics, text, 0)
    #print(topic, score)
    preds[topic].append(text)
print(get_model_topics(model, tf_vectorizer, lda_topics, int(config["Word cloud"]["n_top_words"])))
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

for topic, component in enumerate(model.components_):

    # need [::-1] to sort the array in descending order
    indices = np.argsort(component)[::-1][:int(config["Word cloud"]["n_top_words"])]

    # store the words most relevant to the topic
    words[topic] = {vocab[i]: component[i] for i in indices}

# lower max_font_size, change the maximum number of word and lighten the background:
n_plot_horz = int(config["Word cloud"]["n_wc_plot_horz"])
fig, ax = plt.subplots(int(np.ceil(n_components / n_plot_horz)), n_plot_horz)
plt.tight_layout()
for i, (key, value) in enumerate(words.items()):
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(value)
    ax[int(i / n_plot_horz), int(i % n_plot_horz)].imshow(wordcloud, interpolation="bilinear")
    ax[int(i / n_plot_horz), int(i % n_plot_horz)].axis("off")
plt.show()


## evaluate the unsupervised LDA model for classification based on a selection of cases






