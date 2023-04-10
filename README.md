# adverse-events

[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)

This repository contains the source code related to the manuscript _"Preliminary Processing and Analysis of an Adverse Event Dataset for Detecting Sepsis-Related Events"_, presented at the [IEEE International Conference on Bioinformatics and Biomedicine (IEEE BIBM 2021)](http://ieeebibm.org/BIBM2021/).

A PDF of the published paper can be accessed [here](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2979827/B579_9996.pdf?sequence=2&isAllowed=y). See [here](https://github.com/andreped/adverse-events/releases/tag/v1.0) to download the exact version of the source code used in the publication (v1.0).

### Usage:

1) Clone repo:
```
git clone https://github.com/andreped/adverse-events.git
```

2) Create virtual environment, activate it, and install dependencies:
```
cd adverse-events/python
virtualenv -ppython3 venv --clear
source venv/bin/activate
pip install -r /path/to/requirements.txt
```

3) Create the project structure as defined [below](https://github.com/andreped/adverse-events#project-structure).

4) Run scripts for training and evaluating different classifier models:
```
python main.py misc/default-params.ini
```
Different parameters relevant for the analysis, building of models, evaluation, plotting results, and similar, may be modified in the INI-file.

### Project structure

    └── adverse-events
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

### Acknowledgements

If you use parts of the source code in your research, please, cite this publication:

```
@INPROCEEDINGS{yan2021sepsis,
    author={Yan, Melissa Y. and Høvik, Lise Husby and Pedersen, André and Gustad, Lise Tuset and Nytrø, Øystein},
    booktitle={2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
    title={Preliminary Processing and Analysis of an Adverse Event Dataset for Detecting Sepsis-Related Events},
    year={2021},
    pages={1605-1610},
    doi={10.1109/BIBM52615.2021.9669410}
}
```
