
# adverse-events
[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)

This repository contains code for building machine learning models for extracting information from adverse events notes.

### Usage:

1) Clone repo:
```
git clone https://github.com/andreped/adverse-events.git
cd adverse-events
```

2) Create virtual environment and intall dependencies:
```
virtualenv -ppython3 venv
source venv/bin/activate
pip install -r /path/to/requirements.txt
```

3) Create the project structure as defined [below](https://github.com/andreped/adverse-events#project-structure):

4) Run scripts for training and evaluations different NLP models:
```
python3 main.py misc/default-params.ini
```
Different parameters relevant for the analysis, building of models, evaluation, plotting results, and similar, may be modified in the .ini-file.

### Project structure

```
+-- {adverse-events}/
|   +-- python/
|   |   +-- multi-class/
|   |   |   +-- train.py
|   |   |   +-- [...]
|   |   +-- topic-analysis/
|   |   |   +-- train.py
|   |   |   +-- [...]
|   |   +-- utils/
|   |   |   +-- losses.py
|   |   |   +-- [...]
|   |   +-- [...]
|   +-- data/
|   |   +-- AE_data/
|   |   |   +-- EQS_files/
|   |   |   |   +-- 1.txt
|   |   |   |   +-- 2.txt
|   |   |   |   +-- [...]
|   |   +-- file_with_all_notes.csv
|   |   +-- file_with_annotated_notes.csv
|   |   +-- [...]
|   +-- output/
|   |   +-- history/
|   |   |   +--- history_some_run_name1.txt
|   |   |   +--- history_some_run_name2.txt
|   |   |   +--- [...]
|   |   +-- models/
|   |   |   +--- model_some_run_name1.h5
|   |   |   +--- model_some_run_name2.h5
|   |   |   +--- [...]
|   |   +-- datasets/
|   |   |   +--- preprocessed_dataset1.h5
|   |   |   +--- preprocessed_dataset2.h5
|   |   |   +--- [...]
```

------

Made with :heart: and Python
