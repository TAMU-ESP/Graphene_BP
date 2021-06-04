# Graphene_BP
The main file to run is called BP_BioZ_v0.py

## Installation of Outside Packages

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages.

```bash
pip install numpy 
pip install pandas
pip install hjson
pip install scikit_learn
pip install tensorflow
pip install xgboost
```

## Requirements

Extracted through [pypreqs](https://pypi.org/project/pipreqs/)

```python
xgboost==1.4.2
numpy==1.20.3
pandas==1.2.4
hjson==3.0.2
tensorflow==2.3.0
scikit_learn==0.24.2
```

## Configuration
There are several parameters to be configured before running the BP estimation from Bio-Z. If you want to regenerate the original results you can keep the training-related parameters untouched.

To configure the parameters run:
```bash
python configure.py
```
This will walk you through setting the parameters. To use default values for each parameter, showed in the brackets, press "Enter" without entering any character.

example:
```bash
Use average features instead of beat-to-beat? (true/false)
features_mean [true]:
```

## Usage
To run BP estimation from Bio-Z:
```bash
python BP_BioZ_v0.py
```
The results including the BP estimation for each subject and the overal error metrics will be stored in folder: _'./Data/predictions/'_
