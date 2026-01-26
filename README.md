# Optical Character Recognition for Receipts Total Amount Extraction

This repository contains code for extracting the total amount from receipt images using Optical Character Recognition (OCR) techniques. The project leverages preprocessing, deep learning models, and post-processing to accurately identify and extract the total amount from various receipt formats.

## Table of Contents

- [Abstract](#abstract)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Main Pipeline](#main-pipeline)
- [Author](#author)

## Abstract

This project is done as part of an interview assignment for a Data Scientist internship at a given company. The objective is to build an OCR system that can accurately extract information from receipt images, without relying on pre-trained OCR libraries/APIs (which can be black-boxes).

## Project Overview

The goal of this project is to develop an OCR system capable of accurately extracting the objects and their associated amounts from receipt images. This involves several key steps, including image preprocessing to enhance quality, text detection to locate relevant regions, text recognition to convert images to text, and post-processing to extract and validate the (object, amount) tuples. The project utilizes a dataset of receipt images with annotations for training and evaluation.


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
│   │   └── preprocessing_pipeline.py               # Main preprocessing pipeline for the raw images
│   │
│   ├── dataset/                                    # Dataset handling scripts
│   │   └── dataset_loader.py                       # Functions to load and manage the dataset
│   │
│   ├── optical_character_recognition/              # OCR pipeline scripts
│   │   ├── detection.py                            # Text detection functions (using morphological operations and contour detection)
│   │   ├── detection_pipeline.py                   # Pipeline for detecting text regions in images
│   │   ├── recognition.py                          # Model for recognizing text from detected regions
│   │   └── training_pipeline.py                    # Pipeline for training the recognition model
│   │
│   ├── evaluation/                                 # Evaluation scripts
│   │   ├── metrics.py                              # Functions to compute evaluation metrics
│   │   └── evaluate_model.py                       # Script to evaluate model performance
│   │
│   └── postprocessing/                             # Post-processing scripts
│       └── utils.py                                # Functions for post-processing (e.g., regex for amount extraction, validation, etc.)
│
├── notebooks/                                      # Jupyter notebooks for exploration and experimentation (not part of the main pipeline)
│   ├── exploring_preprocessing.ipynb               # Exploring preprocessing techniques (to remove background, explore annotations, etc.)
│   ├── denoising_images.ipynb                      # Experimenting with image denoising and wrinkle removal using morphological operations
│   └── textbox_detection.ipynb                     # Experimenting with text detection using morphological operations and contour detection
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

**Need to assess whether or not to include dataset_loader.py directly in recognition folders rather than having a separate dataset folder.**

**Also need to assess whether it is sufficient to have preprocessing with standard morphological operations and machine learning, or if deep learning-based approaches should be included as well.**

<ins>Note:</ins> The `data/` directory is included in the `.gitignore` file for two reasons. First, to avoid pushing potentially large datasets to the repository, which can overload the version control system and make cloning the repository difficult. Second, though the considered dataset is not sensitive, it could have been. Therefore, excluding the `data/` directory is of good practice, and helps prevent accidental exposure of sensitive information. Should the user wish to run the project, they would need to have the dataset available locally and place it in a created `data/` directory.

<ins>Note 2:</ins> The `notebooks/` directory contains Jupyter notebooks used for exploratory data analysis and experimentation with different models. These are not part of the main OCR pipeline but were created to understand the data, the problem, and to experiment with various approaches. They would not be included in a production deployment of the project.

## Usage

To run the OCR pipeline, use the `main.py` script with appropriate parameters. For example:

```bash
python main.py [TO BE COMPLETED WHEN PROJECT IS FINALIZED]
```

## Main Pipeline

The Optical Character Recognition pipeline for the receipt dataset consists of the following main steps:
- **Preprocessing**: Enhance the quality of receipt images using techniques such as (several if needed) "smart" cropping(s) based on segmentation masks, de-wrinkling and denoising using morphological operations, deskewing, and binarization. The transformations operated on the raw images are also made on the bounding box annotations to keep them consistent. The bounding boxes are then used for training the text recognition model, on each individual text region, for each receipt. That way, we augment the size of our dataset significantly (from 100 receipt images to about 15 text regions per receipt on average, i.e., about 1500 samples).
- **Text Detection**: Identifying the bounding boxes of text regions in the preprocessed images using morphological operations and contour detection. This step is particularly important for the test set: though we have bounding box annotations the said set, we want to simulate a real-world scenario where no annotations are available at inference time. Given the detection function is based on morphological operations and contour detection, it also extracts artifacts. To mitigate this, we can add artifacts to the training set (by running the detection function on the training images as well) to make the recognition model more robust to such noise. Either way, the function we implemented already adds constraints on the bounding boxes it extracts to filter out some of the artifacts.
- **Text Recognition**: Using a deep learning model (which remains to be defined) to recognize text from the detected text regions. The model is trained on the preprocessed bounding box regions extracted from the training set.
- **Post-processing**: Extracting the tuples from the recognized text, mainly using regular expressions to identify monetary amounts and common object names on receipts, and formatting the results.
- **Analysis and Evaluation**: Evaluating the performance of the OCR system using appropriate metrics (to be defined) and analyzing the results to identify strengths and weaknesses.

## Author

Gdeterline - Guillaume Macquart de Terline - January 2026