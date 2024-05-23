# SentimentGPT
![](https://img.shields.io/badge/Python-3.10-blue)
![](https://img.shields.io/badge/CUDA-12.1-yellow)
![](https://img.shields.io/badge/PyTorch-2.3.0-red)

The pilot coding work of "Advancing Sentiment Analysis on Social Media: Integrating Feature Extraction and GPT-4 for Enhanced Emotional Recognition"

This project uses a GPT-2 model to perform sentiment and emotion analysis on tweets. The model is fine-tuned on a custom dataset containing tweets with annotated sentiments and emotions.

## Table of Contents
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Results](#results)
- [References](#references)

## Project Overview

This project involves the following key steps:
1. Data preprocessing and tokenization.
2. Fine-tuning the GPT-2 model on the sentiment and emotion dataset.
3. Evaluating the model performance on a validation set.
4. Analyzing the results and metrics.

## Environment Setup

To set up the environment, ensure you have the following libraries installed:

```bash
pip install torch transformers pandas scikit-learn
```

## Data Preparation

1. **Download the Dataset**: Place your dataset CSV file in the project directory. The dataset should have the following columns:
    - `Datetime`: The date and time of the tweet.
    - `Tweet Id`: The unique identifier of the tweet.
    - `Text`: The content of the tweet.
    - `Username`: The username of the tweet author.
    - `sentiment`: The sentiment label (e.g., "neutral", "negative", "positive").
    - `sentiment_score`: The confidence score of the sentiment label.
    - `emotion`: The emotion label (e.g., "happy", "sad", etc.).
    - `emotion_score`: The confidence score of the emotion label.

2. **Preprocess the Data**: The data preprocessing steps include cleaning the text, encoding the labels, and splitting the data into training and validation sets.

## Training the Model

Run the `main.py` script to start training the model.

## Evaluating the Model

After training, you can evaluate the model's performance on the validation set using various metrics such as accuracy, precision, recall, and F1 score.

## Results

The evaluation results, including the final accuracy, precision, recall, and F1 score for both sentiment and emotion classification, will be logged and can be analyzed to understand the model's performance.

## References

- [Transformers Library by Hugging Face](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
