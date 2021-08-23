from __future__ import absolute_import
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
from skopt import BayesSearchCV, gp_minimize, dump, load
from skopt.utils import use_named_args
from skopt.space import Integer
from skopt.callbacks import CheckpointSaver
import jsonpickle

# print(sys.path)
sys.path.append(os.path.realpath("./utils/"))
# sys.path.append(os.path.abspath(os.getcwd() + "../"))

# sys.path.append(os.path.join(".", os.path.dirname(__file__), "..", "utils"))

sys.path.append(".")

print()
print(sys.path)
# from .. import utils
from utils import get_class_weights, get_model_topics, get_inference, unique_str, to_categories
from losses import categorical_focal_loss
from metrics import f1_m, precision_m, recall_m, fbeta_score_macro
from stats import BCa_interval_macro_metric
from datetime import datetime

# mute all sklearn warnings
import warnings
warnings.filterwarnings('ignore')


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
figures_path = os.getcwd() + "/../output/figures/"

data_raw = pd.read_csv(raw_data_path)
annotated_raw = pd.read_csv(annotated_data_path)

# remove nan-initialized samples from annotated data
annotated_raw = annotated_raw[:-int(config["Preprocessing"]["n_remove_samples_end"])]

print(data_raw.head())
print(annotated_raw.head())

#exit()
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
search_space = list()
search_space.append(Integer(2, 100, name='K'))
search_space.append(Integer(1000, 10000, name='N'))
search_space.append(Integer(1, 10, name='n_min'))
search_space.append(Integer(11, 50, name='n_max'))
search_space.append(Integer(15, 50, name='l_min'))


# extract relevant GT from test data (manually labelled, at note-level)
infections = annotated_raw["manuelt_merket_Infeksjonsrelatert"]
device_failures = annotated_raw["manuelt_merket_Feil_pa_enheten"]
fallens = annotated_raw["manuelt_merket_falt"]
mistake_unit = annotated_raw["Feil_pa_enheten"]
catheters = annotated_raw["Kateterrelatert"]
sepsis = annotated_raw["Sepsis"]
pvks = annotated_raw["PVK_relatert"]

infections_merged = annotated_raw["merge_Infeksjonsrelatert"]
category_Infeksjon = annotated_raw["category_Infeksjon"]
category_Enhet = annotated_raw["category_Enhet"]

infections = to_categories(infections)
device_failures = to_categories(device_failures)
fallens = to_categories(fallens)
mistake_unit = to_categories(mistake_unit)
catheters = to_categories(catheters)
sepsis = to_categories(sepsis)
pvks = to_categories(pvks)

infections_merged = to_categories(infections_merged)
device_failures_merged = np.logical_or(device_failures, mistake_unit).astype(int)
category_Infeksjon = to_categories(category_Infeksjon)
category_Enhet = to_categories(category_Enhet)

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
unique_category_Infeksjon = []
unique_category_Enhet = []
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
    tmp2 = category_Infeksjon[tmp]
    unique_category_Infeksjon.append(scipy.stats.mode(tmp2)[0][0])
    tmp2 = category_Enhet[tmp]
    unique_category_Enhet.append(scipy.stats.mode(tmp2)[0][0])

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
print(np.unique(unique_category_Infeksjon), np.bincount(unique_category_Infeksjon))
print(np.unique(unique_category_Enhet), np.bincount(unique_category_Enhet))

# exit()

# preprocess: 1) remove or substitute/impute sensitive data based on removing unique words/elements in notes

# produce shuffled corpus for training models
corpus = np.asarray(data_raw_notes)
np.random.shuffle(corpus)

corpus_orig = corpus.copy()
del corpus

perform = config["Bayes"]["eval"]  # config["Analysis"]["perform"]

# define evaluation function to be used in Bayesian hyperparameter optimization
@use_named_args(search_space)
def evaluate_model(**params):

    n_components = params["K"]
    n_min_note_length = params["l_min"]
    n_min_words = params["n_min"]
    n_max_words = params["n_max"]
    n_features = params["N"]

    new = []
    for x in corpus_orig:
        if isinstance(x, str):
            if (x != "Se vedlegg") and (len(x) > n_min_note_length):  # int(config["Preprocessing"]["n_min_note_length"])):
                new.append(x)
        else:
            pass
            # print(np.isnan(x))
    corpus = new.copy()
    del new

    # remove redundant topics (I'm lazy, hopefully my topic analysis model will handle this)
    removes = ["\n", "Hele_Notater"]  # , "pasienter", "pasienten", "pasient"]
    for i, c in enumerate(corpus):
        for r in removes:
            c = c.replace(r, " ")
        c.strip()
        corpus[i] = c

    # pattern for tokenizer
    pattern = r'\b\w{' + re.escape(str(n_min_words)) + ',' + re.escape(str(n_max_words)) + r'}\b'

    # tokenizers
    token = config["Tokenizer"]["token"]
    if token == "count":
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words=tuple(config["Tokenizer"]["stop_words"]), ngram_range=tuple([int(x) for x in config["Tokenizer"]["n_grams"].split(",")]),
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

    lda_topics = list(range(n_components))

    # LDA
    method = config["Topic analysis"]["method"]
    if method == "LDA":
        #print("Performing LDA: ")
        model = LatentDirichletAllocation(n_components=n_components, random_state=n_seed, verbose=eval(config["LDA"]["verbose"]),
                                              n_jobs=int(config["LDA"]["n_jobs"]), max_iter=int(config["Topic analysis"]["n_iter"])).fit(tf_out)
    elif method == "LSA":
        model = TruncatedSVD(n_components=n_components, random_state=eval(config["Analysis"]["n_seed"]), algorithm=config["LSA"]["alg"]).fit(tf_out)
    else:
        #print("Unknown topic analysis method was defined.")
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
    # print("\nEVAL ON TEST SET: ")
    # Display some predictions/results to interpret model performance
    preds = {key: [] for key in lda_topics}
    for text in unique_annotated_raw_notes:
        #print("\nCurrent line: ")
        #print(text)
        topic, score, _ = get_inference(model, tf_vectorizer, lda_topics, text, 0)
        #print(topic, score)
        preds[topic].append(text)
    #'''

    # print(get_model_topics(model, tf_vectorizer, lda_topics, int(config["Word cloud"]["n_top_words"])))

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
        # print("Producing word clouds...")

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

    '''
    out = []
    for top, inf in zip(topic_preds, unique_infections):
        out.append([top, inf])

    out_fallens = []
    for top, fal in zip(topic_preds, unique_fallens):
        out_fallens.append([top, fal])

    out_device = []
    for top, dev in zip(topic_preds, unique_device_failures):
        out_device.append([top, dev])
    '''

    #print()
    #print(infections)

    ## check which topic that best represents the different classes
    # start with infection
    topic_preds = np.array(topic_preds)

    # nb_components = int(config["Topic analysis"]["n_components"])

    accs = {"infections": [], "fallens": [], "device_failures": [], "pvks": [], "catheters": [], "infections_merged": [], "category_Infeksjon": [], "category_Enhet": []}
    origs = {key: [] for key in list(accs.keys())}
    for top in range(n_components):
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
        origs["pvks"].append([unique_pvks, tmp])

        ret = precision_recall_fscore_support(unique_catheters, tmp, labels=[0, 1], average='macro')
        accs["catheters"].append(ret[2])
        origs["catheters"].append([unique_catheters, tmp])
        #iaccs.append(np.sum(unique_fallens == tmp))

        ret = precision_recall_fscore_support(unique_infections_merged, tmp, labels=[0, 1], average='macro')
        accs["infections_merged"].append(ret[2])
        origs["infections_merged"].append([unique_infections_merged, tmp])

        ret = precision_recall_fscore_support(unique_category_Infeksjon, tmp, labels=[0, 1], average='macro')
        accs["category_Infeksjon"].append(ret[2])
        origs["category_Infeksjon"].append([unique_category_Infeksjon, tmp])

        ret = precision_recall_fscore_support(unique_category_Enhet, tmp, labels=[0, 1], average='macro')
        accs["category_Enhet"].append(ret[2])
        origs["category_Enhet"].append([unique_category_Enhet, tmp])

    keys_ = list(accs.keys())

    # print(accs)
    # print("Highest f1-score for the individual tasks " + str(keys_) + ": ")
    # print([(key, max(accs[key])) for key in keys_])


    def some_func(x, labels=[0, 1]):
        ret = precision_recall_fscore_support(x[:, 0], x[:, 1], labels=labels, average='macro', zero_division=0)
        return ret[2]


    # print("---")
    res = {key: [] for key in keys_}
    for key in keys_:
        tmp = accs[key]
        orig_ = origs[key]
        max_ = max(tmp)

        if len(orig_) == 0:
            continue
        new = orig_[np.argmax(tmp)]
        # print()
        # print("key: ", key)
        # print(orig_)
        # print(max_)
        # print(new)

        if perform:
            # acc_vals = np.array(new[0] == new[1]).astype(int)

            tmp = np.stack([new[0], new[1]], axis=1)
            ci, theta_boot_mean = BCa_interval_macro_metric(tmp, func=some_func, B=10000)
            res[key] = [max_, theta_boot_mean, ci]
        else:
            #print(theta_boot_mean)
            # print(max_, ci)
            res[key] = max_
            #res[key] = (max_, "with CI: ", max_ - 1.96 * )

    if perform:
        return res[curr_task]
    else:
        #bs = IIDBootstrap()
        #ci = bs.conf_int(sharpe_ratio, 1000, method='bca')

        value = res[curr_task]

        #print()
        #print("Current iter: ")
        params_ = params.copy()
        params_["F1"] = value
        str_ = ""
        for key in list(params_.keys()):
            str_ += key + ": " + str(params_[key]) + ", "
        str_ = str_[:-2]
        print()
        print(str_)
        # print()
        # print("Current F1: ", value)

        return 1 - value



# PARAMS for Bayesian optimization (more details above function)
curr_task = config["Bayes"]["curr_task"]  # "infections", "fallens", "device_failures", "pvks", "catheters", "infections_merged"
n_calls = int(config["Bayes"]["n_calls"])
gp_verbose = eval(config["Bayes"]["gp_verbose"])
n_init_pts = int(config["Bayes"]["n_init_pts"])

print("\nINI PARAMS: ")
print(curr_task)
print(n_calls)
print(gp_verbose)
print(n_init_pts)

# only load and print resilts. If True, then does not perform Bayesian hyperparamter optimization, but rather loads file
curr_eval = config["Bayes"]["eval"]
if curr_eval:
    print("#"*20)
    print("\nEVAL\n")
    result = load(history_path + curr_eval)
    print(result)
    print()
    print(result.func_vals)

    f1s = result.func_vals
    f1s = f1s[:800]

    f1s = 1 - result.func_vals

    top_ = max(f1s)
    pos_ = np.argwhere(f1s == top_)

    # now run prediction using data to generate CIs and whatnot
    # evaluate_model()

    print(search_space)
    print()
    names = [x._name for x in search_space]

    best = result.x

    print(names, best)

    optimal_params = {key: int(c) for key, c in zip(names, best)}

    print("Optimal params: ")
    print(optimal_params)
    ret = evaluate_model(best)

    print("final result: ")
    print(ret)


    exit()

    fig, ax = plt.subplots(1, 1)
    ax.plot(f1s)
    ax.scatter(pos_, [top_ for x in range(len(pos_))], c="r", marker="x")
    ax.grid("on")
    ax.legend(["Best F1: " + str(np.round(top_, 4))], loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_path + "figure_f1history_" + curr_eval.split("history_")[-1].split(".pkl")[0] + ".png", dpi=300)

    if eval(config["Bayes"]["show_figure"]):
        plt.show()

    print("Finished eval!")
    exit()

search_space_names = [x.name for x in search_space]

# callback for saving results for each iteration (overwrites)
curr_path = history_path + "history_bayes_" + name + "_" + curr_task + "_" + str(search_space_names).replace("'", "").replace(" ", "") + ".pkl"
checkpoint_saver = CheckpointSaver(curr_path, compress=9)
print("Checkpoint save path:", curr_path)

# perform optimization
print("\n\n\nPerforming Bayesian optimization...\n\n\n")
result = gp_minimize(
    evaluate_model,
    search_space,
    n_calls=n_calls,
    n_initial_points=n_init_pts,
    verbose=gp_verbose,
    random_state=n_seed,
    callback=[checkpoint_saver],  # , tqdm_skopt(total=n_calls, desc="Gaussian Process")],
)

# save final result (DEPRECATED! If CheckpointSaver is used, the model will be saved for each iteration. Hence, the model has already been saved.
# dump(result, history_path + "history_bayes_" + name + "_" + curr_task + "_" + str(search_space_names).replace("'", "").replace(" ", "") + ".pkl")

# from skopt import load
curr_path = history_path + "history_bayes_" + name + "_" + curr_task + "_" + str(search_space_names).replace("'", "").replace(" ", "") + ".pkl"
res_loaded = load(curr_path)
print("History save path:", curr_path)

print("\nResults from task: " + curr_task + "\n")
print(result)
print()
print(res_loaded)

print("\nFinished!")

