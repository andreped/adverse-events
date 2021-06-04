import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
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
# import importlib

sys.path.append(os.path.abspath("../utils/"))
from utils import get_class_weights
from losses import categorical_focal_loss
from metrics import f1_m, precision_m, recall_m, fbeta_score_macro


negative_data_path = "../../data/AE_data/EQS_files/"
positive_data_path = "../../data/AE_data/2020_03_04_Uttrekk_kateter_fra_2015_uten_id.csv"
save_model_path = "../../output/models/"
history_path = "../../output/history/"
datasets_path = "../../output/datasets/"

data = pd.read_csv(positive_data_path)
# Preview the first 5 lines of the loaded data
print(data.head())
print(list(data.keys()))

X_orig = data["Hendelsesbeskrivelse"]
Y_orig = data['Klassifisering av alvorlighetsgrad']

print(X_orig)
print(Y_orig)

print(Y_orig.unique())

hist = Y_orig.hist()

# histogram
hist_ = Y_orig.value_counts()
print("hist:\n\n ", hist_)
names = list(hist_.keys())

# work with numpy arrays (because it is way easier)
X_orig = np.asarray(X_orig)
Y_orig = np.asarray(Y_orig)

# remove NaN from X
filter1_ = ~pd.isnull(X_orig)
X_orig = X_orig[filter1_]
Y_orig = Y_orig[filter1_]

# remove redundant classes
filter_ = (Y_orig != "Ukjent") & (Y_orig != "Mindre alvorlig konsekvens") & (Y_orig != "Betydelig skade") & (Y_orig != "Uventet dødsfall") & (~pd.isnull(Y_orig))
X_orig = X_orig[filter_]
Y_orig = Y_orig[filter_]

new_names = list(set(Y_orig))

print(Y_orig)

# categorical to numerical
Y = LabelEncoder().fit_transform(Y_orig)

# convert to numpy arrays for simplicity
X = np.asarray(X_orig)

### Preprocess

token_m = "counter"  # {"hashing", "counter", "tfidf", "keras"}

# extract features using simple word count
if token_m == "hashing":
    vectorizer = HashingVectorizer(n_features=5000, lowercase=True, strip_accents="unicode", analyzer="word")  # CountVectorizer(max_features=5000) #HashingVectorizer(n_features=5000) # #TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)
elif token_m == "counter":
    vectorizer = CountVectorizer(max_features=10000, min_df=1, ngram_range=(1, 1), lowercase=True, strip_accents="unicode", analyzer="word")
    X = vectorizer.fit_transform(X)
    X = X.toarray()
elif token_m == "tfidf":
    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(X)
    X = X.toarray()
elif token_m == "keras":
    pass
else:
    print("Invalid tokenizer was chosen. Please choose one of these alternatives: '{hashing, counter, tfid, keras}'")
    exit()

nb_classes = len(np.unique(Y))
nb_feats = X.shape[1]

# calculate class weights
class_weights = compute_class_weight("balanced", np.unique(Y), Y)
Y_before = Y.copy()

print(class_weights)
print(np.histogram(Y_before))

class_weights = {i: w for i, w in enumerate(class_weights)}

#class_weights = get_class_weights(Y)

#class_weight = {0: 10, 1: 1, 2: 1, 3: 5, 4: 10}

# one-hot encode GT
print(np.unique(Y), Y.dtype)
Y = np.eye(nb_classes)[Y]

print(Y)

# shuffle data
order_ = np.asarray(range(len(Y)))
np.random.shuffle(order_)
X = X[order_]
Y = Y[order_]

# split data into train/val/test
N = len(Y)
val1 = 0.6
val2 = 0.8

X_train = X[:int(val1 * N)]
Y_train = Y[:int(val1 * N)]

X_val = X[int(val1 * N):int(val2 * N)]
Y_val = Y[int(val1 * N):int(val2 * N)]

X_test = X[int(val1 * N):]
Y_test = Y[int(val1 * N):]

print("Size of each set: ")
print(len(Y_train), len(Y_val), len(Y_test))

### create model

# model
inputs = Input(shape=(nb_feats,))
x = Dense(30)(inputs)  # x
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Activation("relu")(x)
#x = Dense(10, activation="relu")(inputs)
#x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
#x = Activation("relu")(x)
x = Dense(nb_classes, activation="softmax")(x)
model = Model(inputs=inputs, outputs=x)

# keras deep embedding instead of preprocessing?
'''
embedding_dim = 50
vocab_size =

model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_dim,
                    input_length=maxlen))
model.add(GlobalMaxPool1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='softmax'))
'''

print(model.summary())

# compile
model.compile(
    loss="categorical_crossentropy",
    #loss=categorical_focal_loss(gamma=3.0, alpha=0.25),
    optimizer=Adam(lr=1e-4),
    weighted_metrics=["acc", f1_m, precision_m, recall_m],
    metrics=[F1Score(nb_classes, average="macro")],
)

# saving best model
mcp_save = ModelCheckpoint(
    save_model_path + 'curr_model.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='auto',
    verbose=1,
)

# early stopping
early = EarlyStopping(
    monitor='val_loss',
    patience=20,
    verbose=1,
)

# fit
model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=100,
    batch_size=64,
    class_weight=class_weights,
    verbose=1,
    callbacks=[early, mcp_save]
)

# evaluate model
print("--- TRAIN ---")
print(classification_report(np.argmax(Y_train, axis=-1), np.argmax(model.predict(X_train), axis=-1), target_names=new_names))
print("--- VAL ---")
print(classification_report(np.argmax(Y_val, axis=-1), np.argmax(model.predict(X_val), axis=-1), target_names=new_names))
print("--- TEST ---")
print(classification_report(np.argmax(Y_test, axis=-1), np.argmax(model.predict(X_test), axis=-1), target_names=new_names))






