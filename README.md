# Deploying Neural Networks to Predict Ab Interno Trabeculotomy Outcomes using Infrared Ocular Surface Images from Pre-op Biometry

## Overview
This repository contains the code and resources for deploying neural networks to predict outcomes of ab interno trabeculotomy using infrared ocular surface images obtained during pre-operative biometry. The models are trained and tested on this data to provide predictions for surgical outcomes.

## Usage
To use this repository, follow these steps:

1. **Install Required Python Modules:**
   Install all the necessary Python modules specified in the `requirements.txt` file.

2. **Setup:**
   Run `setup.py` to create the necessary folders and set up the project environment.

3. **Configuration:**
   Modify the `config.ini` file to customize the parameters and tailor the experience to your needs.

4. **Run Main Script:**
   Execute `main.py` to generate the results based on the configured settings.

## Configuration Parameters

- **Run_group:**
  Used for organizing the ROC experiment. Models belonging to the same run group will be plotted in the same ROC plot.

- **Learning_rate:**
  Learning rate for the machine learning model during training.

- **Flipping:**
  Controls whether to flip the image to align the nasal and temporal sides.

- **Num_runs:**
  Number of runs for cross-fold validation in the experiment.

- **ROC:**
  Determines whether the experiment will generate a ROC plot or not.

- **Draw:**
  Controls whether to plot the training loss and ROC.

- **Epochs:**
  Controls the number of training epochs to run.

- **Model_name:**
  Unique identifier model name for stored plots.

- **Model:**
  Machine learning model used for the experiment.

- **Masking:**
  Specifies the area to be masked for ablation study.

Ensure you customize the configuration parameters according to your specific use case and experimental requirements.

---

