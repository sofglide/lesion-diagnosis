# Lesion Diagnosis

This repo is for experimenting image classification techniques on the classical HAM10000
problem. The dataset is made of 10015 dermatoscopic images of pigmented skin lesions and
a csv file containing images metadata including the diagnostic category.

## Objective
The purpose of this work is to have a flexible code for experimenting various techniques
and find their strength and weakness. It is also aimed to have a didactic spirit so that
a reader starting in machine learning can understand, reproduce and modify this code easily.

Please keep in mind that this repo is a work in progress where I experiment ideas on my free time. 

## Data analysis
The dermatoscopic images are in RGB mode and have a resolution of (600, 450).

The image metadata are 'lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization'.

For now, we are only interested in image classifier, so we will ignore the additional features:
'dx_type', 'age', 'sex', 'localization'.

The label column is 'dx' which stands for diagnostic.

The column 'image_id' maps the row to the image file.

An important column is 'lesion_id'. We can see that there are duplicates in this column
(up to 6 duplicates for some 'lesion_id'). This means that some images may refer to the same lesion
and we need to handle these duplicates carefully, in particular, we should not spread images of the
same lesion over the training, validation, and test sets.

There are 7 classes 'nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df' and there is a strong imbalance
in the dataset.
![data distribution](images/data_distribution.png?raw=true "Data distribution")

Therefore, care should be taken when splitting the dataset and when choosing the loss function
and the metrics.

More details about the dataset exploration can be found in the notebook `data_exploration.ipynb`.


## Current experiment


## Repo initialization
After cloning the repo, create the virtual environment with `make env-create` then activate it
with `source .venv/bin/activate`.

To download the data run `make download-data`. The data is downloaded from Kaggle using the public
API. For this, you need to create a Kaggle account and a token which needs to be stored in
`$HOME/.kaggle/kaggle.json`.

Details about how to create a token can be found [here](https://www.kaggle.com/docs/api).

## Repo structure
The repo structure is as follows:
```
├── data                # data folder
├── experiments         # experiment logging
├── images              # documentation imags
├── Makefile            # makefile
├── notebooks           # notebooks directory
├── ray_results         # ray-tune log directory
├── README.md           # documentation
├── requirements.in     # requirements file
├── requirements.txt    # pip-tools generated requirements from requirements.ini
├── src                 # source code
├── tests               #  placeholder for tests
└── tune_config         # ray-tune experiments config
```

The structure of `notebooks` directory is as follows:
```
notebooks
├── data_exploration.ipynb          # data exploration notebook
├── experiment_analysis.ipynb       # single experiment exploration notebook
├── tune_analysis.ipynb             # tune experiement exploration notebook
├── tune_cnn.ipynb                  # colab notebook for simple cnn trials
├── tune-densenet.ipynb             # colab notebook for densenet trials
├── tune_hybrid.ipynb               # colab notebook for multi-models trials
└── tune_resnet.ipynb               # colab notebook for resnet trials
```

The directory `tune_config` contains ray-tune configuration files
```
tune_config
├── tune_cnn.json
├── tune_densenet.json
├── tune_hybrid.json
└── tune_resnet.json
```


Finally the source code is structured as follows:
```
src
├── config
│   ├── config.ini              # global configuration file
│   └── __init__.py             # global configuration parser
├── data_processing             # tools for loading data
│   ├── class_mapping.py
│   ├── data_loading.py
│   ├── data_splitting.py
│   ├── ham10000.py
│   ├── image_processing.py
│   ├── image_transforms.py     # data transforming and augmentation
│   ├── __init__.py
│   └── metadata_loading.py
├── experiment_analysis         # tools for results analysis in notebooks
│   ├── experiment.py
│   ├── __init__.py
│   └── parsing.py
├── experimentation             # tools for running model fitting
│   ├── __init__.py
│   ├── single_experiment.py    # module for running a single experiment
│   ├── tune_config_parsing.py
│   └── tune_experiment.py      # module for running a tune job
├── main.py                     # entry point script for running any task
├── networks
│   ├── densenet.py		# feature extraction / transfer learning model with densenet121
│   ├── hybrid.py		# feature extraction / transfer learning using multiple models simultaneously
│   ├── __init__.py
│   ├── model_selection.py
│   ├── resnet.py               # feature extraction / transfer learning model with resnet34
│   └── simple_cnn.py           # simple cnn configurable model: 3 conv layers, 3 fc layers
├── rendering                   # image display
│   ├── images.py
│   ├── __init__.py
│   └── sampling.py
├── saving                      # checkpoint generation during training
│   ├── checkpoints.py
│   ├── data_params.py
│   ├── __init__.py
│   ├── plotting.py
│   └── predictions.py
├── training                    # model training
│   ├── __init__.py
│   ├── logging.py
│   ├── metrics
│   ├── model_training.py
│   ├── model_validation.py
│   ├── optimization.py
│   └── training_manager.py
├── utils                       # utilies
│   ├── computing_device.py
│   ├── experiment_setup.py
│   ├── __init__.py
│   └── logging.py
└── version.py
```

The `main.py` script is easy to understand. Calling it without arguments will redirect you to a verbose
help and calling each of the 3 commands will display the command help and parameters
```shell
python src/main.py
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  download-data      Download and prepare data
  single-experiment  Run a single experiment
  tune-experiment    Run a ray tune driven set of experiments

```
## Duplicate lesion IDs
Spreading images with the same lesion id between the training, validation, and test sets will lead
to data leakage and to unreliable model evaluation as the model will be evaluated on some already
seen lesions (even though the image is different).

The simplest way to deal with this problem, is to random select 1 image from each lesion id and
remove the other images.

There is no really better way to deal with this problem. Splitting the dataset using the lesion
id instead of the image id, will lead to an imbalance in the lesion representation because multiple
images of the same lesion will be present in exactly one of the training, validation, and test sets
and this will impact the evaluation metric.

When keeping only 1 image per lesion id, the dataset size will be reduced from 10015 to 7470.

At the same time, the class imbalance is made slightly stronger (probably because the duplicates
occur less in the strongly represented classes).

Before removing the lesion id duplicates the data distribution is:
```
nv       66.9 %
mel      11.1 %
bkl      11.0 %
bcc       5.1 %
akiec     3.3 %
vasc      1.4 %
df        1.1 %
```
while after removing the lesion id duplicates, the data distribution is:
```
nv       72.3 %
bkl       9.7 %
mel       8.2 %
bcc       4.4 %
akiec     3.1 %
vasc      1.3 %
df        1.0 %
```


##  Dataset size imbalance
The number of images after removing the lesion id duplicates is 7470.
This could be small with respect to the variability which may easily
lead to overfitting. In order to reduce that, some data augmentation has been introduced in
`image_transoforms.py` like random rotation, flipping. More transforms could be added, ideally after
careful examination of image samples in order to make sure these transforms are not creating unrealistic
data.

The label distribution is strongly skewed. This causes at least 2 problems:
  * the training and validation datasets can have different distributions, and some classes may be present
in one set and not the other
  * the model may try to learn more the over-represented classes and less the under-represented classes

To solve the first problem, the dataset is split using a stratified splitting which maintains equal
label distributions in both sets.

To solve the second problem, a weighted loss function is used.

Finally, a hold out test set is always created in the same way (using random seed 0) before creating
the training and the validation sets.

## Model architecture
The repo includes code for the following architectures:
  * `simple cnn`: a simple parameterizable convolutional neural network
  * `resnet`: a pretrained `resnet34` model, after replacing its last fully connected layer and training
this layer. This is a common practice for transfer learning in image classification
  * `densenet`: same as for `resnet` but using `densenet121`
  * `hybrid`: multiple pretrained models are run on the same image after removing their classification
    fully connected layer. The feature vectors obtained from these models are concatenated and fed to
    a multilayer perceptron. Only the latter is trained.
    
## Metrics used
Since the problem is a multiclass classification with imbalanced dataset, we used the metrics
`Matthews correlation coefficient` and `f1 score` as they are suitable for this situation.
    
# The hybrid model architecture
The main idea about using multiple models is whether we may get the best of each model by combining
their feature vectors and if this works, would it be cheaper than fine-tuning a single model after
training its final layer.

This is done by concatenating the feature vectors of multiple
pretrained models (before the fully connected classification layer) and feeding the resulting vector
to a multilayer perceptron that has to be trained.

The first implementation (currently in progress) will run the augmentation algorithm on the input image
and feed the same augmented image to all the pretrained models.
![single augmentation](images/single-augmentation.svg?raw=true "Single augmentation")

Later, we will try the configuration where the augmentation algorithm will be run
separately for each of the pretrained model. We expect the resulting model to be
more robust to data variation and less keen to overfitting.
![multiple augmentation](images/multiple-augmentation.svg?raw=true "Multiple augmentation")
