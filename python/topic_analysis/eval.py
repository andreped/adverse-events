from __future__ import absolute_import
from skopt import load
import os
import sys

# sys.path.append(os.path.realpath("./utils/"))
# sys.path.append(os.path.abspath(os.getcwd() + "../"))
# sys.path.append(os.path.join(".", os.path.dirname(__file__), "..", "utils"))
# sys.path.append(".")

# sys.path.append(os.path.realpath("./utils/"))
# sys.path.append(os.path.realpath("./topic_analysis/"))
# sys.path.append(".")

# print(sys.path)

from train_bayes import evaluate_model
#from utils import get_class_weights, get_model_topics, get_inference, unique_str, to_categories
#from losses import categorical_focal_loss
#from metrics import f1_m, precision_m, recall_m, fbeta_score_macro
#from stats import BCa_interval_macro_metric
#from datetime import datetime

# paths
raw_data_path = os.getcwd() + "/../data/2020_03_04_Uttrekk_kateter_fra_2015_uten_id.csv"
annotated_data_path = os.getcwd() + "/../data/AE_annotated_labeled_Hel_and_Manual_merged_category_20210617_115324.csv"
save_model_path = os.getcwd() + "/../output/models/"
history_path = os.getcwd() + "/../output/history/"
datasets_path = os.getcwd() + "/../output/datasets/"

curr_path = history_path + "history_bayes_010721_215118_topic-analysis_fallens_[K,N,n_min,n_max,l_min].pkl"

res = load(curr_path)

print(res)


