
- Student Name: Chinni Veeravenkatasivanagarajugovind
- Student ID: 700771324
- Course: Neural Network & Deep Learning 

## 📘 Overview

This repository contains source code for five core NLP tasks using Python and popular libraries such as TensorFlow, NumPy, NLTK, spaCy, and HuggingFace Transformers. The goal is to demonstrate practical understanding of key NLP concepts including RNNs, attention, sentiment analysis, and named entity recognition.

## 📂 Contents

- `text_generator.py`: Character-level text generation using LSTM (Q1)
- `nlp_preprocessing.py`: Basic tokenization, stopword removal, and stemming using NLTK (Q2)
- `ner_spacy.py`: Named Entity Recognition using spaCy (Q3)
- `scaled_attention.py`: NumPy implementation of Scaled Dot-Product Attention (Q4)
- `sentiment_huggingface.py`: Sentiment analysis using HuggingFace Transformers (Q5)

---

## Assignment Tasks

### **Q1: Character-Level Text Generation Using LSTM**
- Loads the Shakespeare dataset.
- Preprocesses the text and converts it to character-level sequences.
- Builds and trains an LSTM-based RNN.
- Generates text with temperature scaling to control randomness.

### **Q2: NLP Preprocessing with NLTK**
- Tokenizes a given sentence.
- Removes English stopwords.
- Applies stemming to reduce words to their root form.
- Displays tokens at each stage of processing.

### **Q3: Named Entity Recognition (NER) with spaCy**
- Uses spaCy to extract named entities from a sentence.
- Prints entity text, label (e.g., PERSON, DATE), and character positions.

### **Q4: Scaled Dot-Product Attention Implementation**
- Implements core attention logic using NumPy.
- Applies softmax and scales dot product scores by √d.
- Demonstrates how attention weights determine the output values.

### **Q5: Sentiment Analysis with HuggingFace Transformers**
- Loads a pre-trained pipeline for sentiment classification.
- Analyzes a given sentence.
- Prints the sentiment label and confidence score.

---

## 💻 Requirements

Install the required libraries before running any scripts:

pip install numpy nltk spacy transformers
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
