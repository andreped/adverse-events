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
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow_addons.metrics import F1Score
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import configparser
import scipy
from skopt import BayesSearchCV
import pickle

print(sys.path)
sys.path.append(os.path.abspath(os.getcwd() + "/utils/"))
# sys.path.append(os.path.abspath(os.getcwd() + "../"))
print(sys.path)
from utils import get_class_weights, get_model_topics, get_inference, unique_str, to_categories
from losses import categorical_focal_loss
from metrics import f1_m, precision_m, recall_m, fbeta_score_macro
from stats import BCa_interval_macro_metric
from datetime import datetime


# today's date and time
today = datetime.now()
name = today.strftime("%d%m") + today.strftime("%Y")[2:] + "_" + today.strftime("%H%M%S") + "_topic-analysis"

# parse config file (.ini)
config = configparser.ConfigParser()
config.read(sys.argv[1])

print(sys.argv[1])
# exit()

n_seed = int(config["Analysis"]["n_seed"])
np.random.seed(n_seed)

raw_data_path = os.getcwd() + "/../data/2020_03_04_Uttrekk_kateter_fra_2015_uten_id.csv"
#annotated_data_path = os.getcwd() + "/../data/AE_annotated_Labeled_Hel_and_Manual_20210604.csv"
annotated_data_path = os.getcwd() + "/../data/AE_annotated_labeled_Hel_and_Manual_merged_category_20210617_115324.csv"
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

print("\nraw data keys: ")
print(list(data_raw.keys()))
print("\nannotated data keys: ")
print(list(annotated_raw.keys()))
print()

# exit()

# define searrch space (relevant for Bayesian optimization)
params = dict()



# extract relevant GT from test data (manually labelled, at note-level)
infections = annotated_raw["manuelt_merket_Infeksjonsrelatert"]
device_failures = annotated_raw["manuelt_merket_Feil_pa_enheten"]
fallens = annotated_raw["manuelt_merket_falt"]
mistake_unit = annotated_raw["Feil_pa_enheten"]
catheters = annotated_raw["Kateterrelatert"]
sepsis = annotated_raw["Sepsis"]
pvks = annotated_raw["PVK_relatert"]

infections_merged = annotated_raw["merge_Infeksjonsrelatert"]

infections = to_categories(infections)
device_failures = to_categories(device_failures)
fallens = to_categories(fallens)
mistake_unit = to_categories(mistake_unit)
catheters = to_categories(catheters)
sepsis = to_categories(sepsis)
pvks = to_categories(pvks)

infections_merged = to_categories(infections_merged)
device_failures_merged = np.logical_or(device_failures, mistake_unit).astype(int)

unique_ids = unique_str(ids, remove_nan=True)
unique_annotated_raw_notes = unique_str(annotated_raw_notes, remove_nan=True)

unique_infections = []
unique_fallens = []
unique_device_failures = []
unique_mistake_unit = []
unique_catheters = []
unique_sepsis = []
unique_pvks = []
unique_infections_merged = []
unique_device_failures_merged = []
for note in unique_annotated_raw_notes:
    tmp = np.array(annotated_raw_notes) == note
    tmp2 = infections[tmp]
    unique_infections.append(scipy.stats.mode(tmp2)[0][0])
    tmp2 = fallens[tmp]
    unique_fallens.append(scipy.stats.mode(tmp2)[0][0])
    tmp2 = device_failures[tmp]
    unique_device_failures.append(scipy.stats.mode(tmp2)[0][0])
    tmp2 = mistake_unit[tmp]
    unique_mistake_unit.append(scipy.stats.mode(tmp2)[0][0])
    tmp2 = catheters[tmp]
    unique_catheters.append(scipy.stats.mode(tmp2)[0][0])
    tmp2 = sepsis[tmp]
    unique_sepsis.append(scipy.stats.mode(tmp2)[0][0])
    tmp2 = pvks[tmp]
    unique_pvks.append(scipy.stats.mode(tmp2)[0][0])
    tmp2 = infections_merged[tmp]
    unique_infections_merged.append(scipy.stats.mode(tmp2)[0][0])
    tmp2 = device_failures_merged[tmp]
    unique_device_failures_merged.append(scipy.stats.mode(tmp2)[0][0])

print("Counts for the tasks {infections, fallens, device_failures, mistake_unit, catheters, sepsis, pvks, infec_m, devic_m}:")
print(np.unique(unique_infections), np.bincount(unique_infections))
print(np.unique(unique_fallens), np.bincount(unique_fallens))
print(np.unique(unique_device_failures), np.bincount(unique_device_failures))
print(np.unique(unique_mistake_unit), np.bincount(unique_mistake_unit))
print(np.unique(unique_catheters), np.bincount(unique_catheters))
print(np.unique(unique_sepsis), np.bincount(unique_sepsis))
print(np.unique(unique_pvks), np.bincount(unique_pvks))
print(np.unique(unique_infections_merged), np.bincount(unique_infections_merged))
print(np.unique(unique_device_failures_merged), np.bincount(unique_device_failures_merged))

# exit()

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
removes = ["\n", "Hele_Notater"]  # , "pasienter", "pasienten", "pasient"]
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
tf_model = tf_vectorizer.fit(corpus)
tf_out = tf_model.transform(corpus)

# tf_out = tf_vectorizer.fit_transform(corpus)


# save 


# LDA
method = config["Topic analysis"]["method"]
if method == "LDA":
    print("Performing LDA: ")
    model = LatentDirichletAllocation(n_components=n_components, random_state=n_seed, verbose=eval(config["LDA"]["verbose"]),
                                          n_jobs=int(config["LDA"]["n_jobs"]), max_iter=int(config["Topic analysis"]["n_iter"])).fit(tf_out)
elif method == "LSA":
    model = TruncatedSVD(n_components=n_components, random_state=eval(config["Analysis"]["n_seed"]), algorithm=config["LSA"]["alg"]).fit(tf_out)
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
    topic, score, _ = get_inference(model, tf_vectorizer, lda_topics, text, 0)
    #print(topic, score)
    preds[topic].append(text)

# print(get_model_topics(model, tf_vectorizer, lda_topics, int(config["Word cloud"]["n_top_words"])))
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
#'''

print(get_model_topics(model, tf_vectorizer, lda_topics, int(config["Word cloud"]["n_top_words"])))

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

if eval(config["Word cloud"]["perform"]):
    print("Producing word clouds...")

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
#print(infections)

preds = {key: [] for key in lda_topics}
topic_preds = []
for text in unique_annotated_raw_notes:
    text = text.replace("Hele_Notater", " ")
    text = text.replace("\n", " ")
    topic, score, _ = get_inference(model, tf_vectorizer, lda_topics, text, 0)
    #preds[topic].append(text)
    topic_preds.append(topic)
    
#print(len(unique_annotated_raw_notes))

#exit()

#print(topic_preds)
#print(len(unique_annotated_raw_notes))

out = []
for top, inf in zip(topic_preds, unique_infections):
    out.append([top, inf])

print()
print("Predicted topic vs infection topics:")
print(out)

out_fallens = []
for top, fal in zip(topic_preds, unique_fallens):
    out_fallens.append([top, fal])

print()
print("Predicted topic vs fallen topics:")
print(out_fallens)

out_device = []
for top, dev in zip(topic_preds, unique_device_failures):
    out_device.append([top, dev])

print()
print("Predicted topic vs device_failure topics:")
print(out_device)

print()
print(len(unique_infections), len(unique_fallens), len(unique_device_failures))
print(len(out_device))
#print()
#print(infections)

## check which topic that best represents the different classes
# start with infection
topic_preds = np.array(topic_preds)

nb_components = int(config["Topic analysis"]["n_components"])

accs = {"infections": [], "fallens": [], "device_failures": [], "pvks": [], "catheters": []}
origs = {key: [] for key in list(accs.keys())}
for top in range(nb_components):
    tmp = np.array(topic_preds == top).astype(int)
    #ret = classification_report(unique_fallens, tmp, labels=[0, 1])
    ret = precision_recall_fscore_support(unique_infections, tmp, labels=[0, 1], average='macro')
    accs["infections"].append(ret[2])
    origs["infections"].append([unique_infections, tmp])

    ret = precision_recall_fscore_support(unique_fallens, tmp, labels=[0, 1], average='macro')
    accs["fallens"].append(ret[2])
    origs["fallens"].append([unique_fallens, tmp])

    ret = precision_recall_fscore_support(unique_device_failures, tmp, labels=[0, 1], average='macro')
    accs["device_failures"].append(ret[2])
    origs["device_failures"].append([unique_device_failures, tmp])

    ret = precision_recall_fscore_support(unique_pvks, tmp, labels=[0, 1], average='macro')
    accs["pvks"].append(ret[2])

    ret = precision_recall_fscore_support(unique_catheters, tmp, labels=[0, 1], average='macro')
    accs["catheters"].append(ret[2])
    #iaccs.append(np.sum(unique_fallens == tmp))

keys_ = list(accs.keys())

print(accs)
print("Highest f1-score for the individual tasks " + str(keys_) + ": ")
print([(key, max(accs[key])) for key in keys_])


def some_func(x, labels=[0, 1]):
    ret = precision_recall_fscore_support(x[:, 0], x[:, 1], labels=labels, average='macro', zero_division=0)
    return ret[2]


print("---")
res = {key: [] for key in keys_}
for key in keys_:
    tmp = accs[key]
    orig_ = origs[key]
    max_ = max(tmp)
    
    if len(orig_) == 0:
        continue
    new = orig_[np.argmax(tmp)]
    print()
    print("key: ", key)
    # print(orig_)
    # print(max_)
    # print(new)

    # acc_vals = np.array(new[0] == new[1]).astype(int)
    
    tmp = np.stack([new[0], new[1]], axis=1)
    ci, theta_boot_mean = BCa_interval_macro_metric(tmp, func=some_func, B=10000)

    #print(theta_boot_mean)
    print(max_, ci)
    #res[key] = (max_, "with CI: ", max_ - 1.96 * )

#bs = IIDBootstrap()
#ci = bs.conf_int(sharpe_ratio, 1000, method='bca')

# finally, save trained models on disk to be easily deployable elsewhere
if eval(config["Export"]["save_flag"]):
    pickle.dump(tf_model, save_model_path + "model_topic_embedding_" + name + ".pk")
    pickle.dump(model, save_model_path + "model_topic_classifier" + name + ".pk")

# load model
# model = pickle.load(save_model_path + "model_topic_" + name + ".pkl")

print("\nFinished!")


