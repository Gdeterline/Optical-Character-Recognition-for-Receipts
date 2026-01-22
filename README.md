# Optical Character Recognition for Receipts Total Amount Extraction

This repository contains code for extracting the total amount from receipt images using Optical Character Recognition (OCR) techniques. The project leverages preprocessing, deep learning models, and post-processing to accurately identify and extract the total amount from various receipt formats.

## Table of Contents

- [Abstract](#abstract)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Usage](#usage)

## Abstract

This project is done as part of an interview assignment for a Data Scientist internship at a given company. The objective is to build an OCR system that can accurately extract the total amount from receipt images, without relying on pre-trained OCR libraries/APIs (which can be black-boxes).

## Project Overview

The goal of this project is to develop an OCR system capable of accurately extracting the total amount from receipt images. This involves several key steps, including image preprocessing to enhance quality, text detection to locate relevant regions, text recognition to convert images to text, and post-processing to extract and validate the total amount. The project utilizes a dataset of receipt images with annotated total amounts for training and evaluation.


## Installation

To set up the project, given you already have the data, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Gdeterline/Optical-Character-Recognition-for-Receipts.git
cd Optical-Character-Recognition-for-Receipts
pip install -r requirements.txt
```

## File Structure

```bash
Optical-Character-Recognition-for-Receipts/
├── data/                                           # Directory for storing receipt images and annotations
│   ├── images/                                     # Raw receipt images
│   │   ├── dev_receipt_00000.png                   # Example receipt image for development
│   │   ├── ...                                     # More receipt images
│   │   ├── test_receipt_00001.png                  # Example receipt image for testing
│   │   └── ...                                     # More receipt images
│   │
│   ├── preprocessed_images/                        # Preprocessed images ready for OCR
│   │   ├── dev_receipt_00000_preprocessed.png      # Preprocessed version of the receipt image
│   │   └── ...                                     # More preprocessed images
│   │
│   └── metadata.pkl                                # Annotations for the images - ground truth data
│
├── src/                                            # Source code
│   ├── preprocessing/                              # Image preprocessing scripts
│   │   ├── utils.py                                # Utility functions for preprocessing (deskewing, normalizing, etc.)
│   │   └── preprocessing_pipeline.py               # Main preprocessing pipeline
│   │
│   ├── dataset/                                    # Dataset handling scripts
│   │   └── dataset_loader.py                        # Functions to load and manage the dataset
│   │
│   ├── models/                                     # Model definitions and training scripts
│   │   ├── detection_model.py                      # Model for detecting text regions
│   │   ├── recognition_model.py                    # Model for recognizing text from detected regions
│   │   └── train_models.py                         # Script to train detection and recognition models
│   │
│   ├── evaluation/                                 # Evaluation scripts
│   │   ├── metrics.py                              # Functions to compute evaluation metrics
│   │   └── evaluate_model.py                       # Script to evaluate model performance
│   │
│   └── postprocessing/                             # Post-processing scripts
│       └── utils.py                                # Functions for post-processing (e.g., regex for amount extraction, validation, etc.)
│
├── notebooks/                                      # Jupyter notebooks for exploration and experimentation (not part of the main pipeline)
│   ├── exploratory_data_analysis.ipynb             # EDA on the receipt dataset 
│   ├── exploring_detection_models.ipynb            # Experimenting with text detection models (for extracting receipt from background)
│   └── exploring_recognition_models.ipynb          # Experimenting with text recognition models (for extracting text from regions)
│
├── reports/
│   ├── figures/                                    # Figures for the report        
│   │   ├── preprocessing_examples.png              # Examples of preprocessing steps image
│   │   └── ...
│   │
│   ├── slides.pdf                                  # Presentation slides    
│   ├── report.md                                   # Written report in markdown/latex format (adapted to be used with Pandoc)
│   └── report.pdf                                  # Written report in PDF format
│
├── main.py                                         # Main script to run the entire OCR pipeline given parameters
├── requirements.txt                                # List of required Python packages
├── README.md                                       # Project overview and instructions
└── .gitignore                                      # Git ignore file
```

**Need to assess whether or not to include dataset_loader.py directly in detection and recognition folders rather than having a separate dataset folder.**

**Also need to assess whether it is actually necessary to proceed to do detection and recognition separately rather than having a single model that does both tasks (depends on time available).**

<ins>Note:</ins> The `data/` directory is included in the `.gitignore` file for two reasons. First, to avoid pushing potentially large datasets to the repository, which can overload the version control system and make cloning the repository difficult. Second, though the considered dataset is not sensitive, it could have been. Therefore, excluding the `data/` directory is of good practice, and helps prevent accidental exposure of sensitive information. Should the user wish to run the project, they would need to have the dataset available locally and place it in a created `data/` directory.

<ins>Note 2:</ins> The `notebooks/` directory contains Jupyter notebooks used for exploratory data analysis and experimentation with different models. These are not part of the main OCR pipeline but were created to understand the data, the problem, and to experiment with various approaches. They would not be included in a production deployment of the project.

## Usage

To run the OCR pipeline, use the `main.py` script with appropriate parameters. For example:

```bash
python main.py [TO BE COMPLETED WHEN PROJECT IS FINALIZED]
```

## Author

Gdeterline