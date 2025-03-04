# NLP-Binary-Classifier-CW

## Description
This repository contains the code for the binary classifier coursework that is part of the NLP module at Imperial College London. The shortcode of the group members are hsh121 and is321.

## Installation
Create a Python 3.12 virtual environment, then in the virtual environment run the following to install all required dependencies.

```bash
# Install dependencies
pip install -r requirements.txt
```

## Repository Outline
During our experiments, we did not consolidate the code into a single Jupyter notebook. Instead, we organized it into separate folders, each dedicated to a specific experiment. To clarify the process of developing our final model, the `main.ipynb` file in this directory contains the code for the methods incorporated into the final model. Some of the code are copy pasted from the original files in the experiment subfolders into the `main.ipynb` file in order to create a much more structured `main.ipynb` file.

## Experiments
Throughout the coursework, the experiments were conducted within their respective subfolders. Here are the descriptions of what each subfolder contains:
- baselines: Files that contain code for the bag-of-words model baseline as well as the DeepSeek v3 baseline.
- cleaning: Files that contain code for cleaning the training dataset via a pre-trained T5-based grammar correction model. This experiment was not incorporated into the final model.
- data: All datasets that were produced by our experiments.
- paraphase: Files that contain code for upsampling the PCL labeled entries by creating new entries via paraphrasing.