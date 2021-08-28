# adverse-events

[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)

This repository contains the code used for a manuscript submitted to the [IEEE International Conference on Bioinformatics and Biomedicine (BIBM) 2021](http://ieeebibm.org/BIBM2021/):
```
@ARTICLE{,
AUTHOR = {Yan, Melissa Y. and Høvik, Lise Husby and Pedersen, André and Gustad, Lise Tuset and Nytrø, Øystein},
TITLE = {Preliminary Processing and Analysis of an Adverse Event Dataset for Detecting Sepsis-Related Events},
YEAR = {2021}}
```

### Usage:

1) Clone repo:
```
git clone https://github.com/andreped/adverse-events.git
```

2) Create virtual environment, activate it, and install dependencies:
```
cd adverse-events/python
virtualenv -ppython3 venv
source venv/bin/activate
pip install -r /path/to/requirements.txt
```

3) Create the project structure as defined [below](https://github.com/andreped/adverse-events#project-structure):

4) Run scripts for training and evaluating different classifier models:
```
python3 main.py misc/default-params.ini
```
Different parameters relevant for the analysis, building of models, evaluation, plotting results, and similar, may be modified in the INI-file.

### Project structure

    ├── adverse-events
        ├── python
        │   ├── multi-class
        │   ├── topic-analysis
        │   ├── utils
        │   └── ...
        ├── data
        │   ├── EQS_files
        │   ├── file-with-all-notes.csv
        │   └── file_with_annotated_notes.csv
        └── output
            ├── history
            ├── models
            └── figures

------
Made with :heart: and Python
