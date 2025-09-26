# HNVec-Driven MultiDL Framework for Hindi News Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A novel deep learning framework for the content-based classification of Hindi news articles, achieving **88.61% accuracy**. [cite_start]This project introduces **HNVec (Hindi News Vectorizer)**, a custom word embedding model tailored for the semantic and contextual nuances of the Hindi language[cite: 92]. [cite_start]It also implements a hybrid learning model (**HLM-CLS**) that combines a Convolutional Neural Network (CNN) with Logistic Regression (LR) and a Support Vector Machine (SVM) to achieve high classification accuracy[cite: 94].

This system was developed to address the challenges of text classification for low-resource languages, demonstrating significant improvements over baseline vectorization methods.

---
## ✨ Key Features

* [cite_start]**Custom Hindi Embeddings (HNVec):** A novel, distance-based context-sensitive vectorizer designed specifically to capture the morphological richness of Hindi text[cite: 92, 917].
* [cite_start]**Hybrid Learning Architecture:** A multi-layer model that leverages the feature extraction power of CNNs and the robust classification capabilities of SVM and LR[cite: 94, 880].
* [cite_start]**High Performance:** Achieved an **accuracy of 88.61%** and an F1-score of 88.51% on the test dataset[cite: 97].
* [cite_start]**Comprehensive Preprocessing:** Includes a robust pipeline for cleaning and preparing Hindi text, handling tokenization, stopword removal, and normalization[cite: 628].

---
## 🏛️ Project Architecture

The framework follows a multi-stage pipeline that integrates feature extraction, feature reduction, and final classification. [cite_start]This hybrid approach is designed to capture complex patterns while maintaining robust classification performance[cite: 882].



**1. Feature Extraction:**
* [cite_start]Input Hindi news articles undergo extensive preprocessing[cite: 628].
* The novel **HNVec vectorizer** converts the text into meaningful, context-aware embeddings.
* [cite_start]Simultaneously, a **Convolutional Neural Network (CNN)** processes the tokenized text to extract high-level features and local patterns[cite: 911].

**2. Feature Combination & Reduction:**
* The features from HNVec and the CNN are combined into a single, enriched feature set.
* This combined set is used to train parallel **Logistic Regression (LR)** and **Support Vector Machine (SVM)** models. The outputs from these models act as a reduced feature set.

**3. Final Classification:**
* A final **SVM classifier** is trained on this reduced feature set to make the ultimate prediction, categorizing the news article into one of the predefined classes (e.g., Sports, Politics, Entertainment).

---
## 🚀 Performance & Results

The HLM-CLS model combined with our custom HNVec embeddings delivered the best performance, validating our approach. [cite_start]The proposed model with HNVec provides the highest accuracy, precision, recall, and F1-Score compared to other combinations[cite: 97].

| Metric    | Score     |
| :-------- | :-------- |
| **Accuracy** | **88.61%** |
| Precision | 88.50%    |
| Recall    | 88.61%    |
| F1-Score  | 88.51%    |

_These results were achieved with a 70:30 train-test split_.

[cite_start]The model showed strong convergence with minimal training loss, indicating that the HNVec-trained model has good generalization without overfitting[cite: 1299].



---
## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Core Libraries:**
    * TensorFlow / Keras
    * Scikit-learn
    * Pandas
    * NumPy
* **Environment:** Google Colab

---
## 🔮 Future Work

[cite_start]Based on the findings of our research, future extensions of this work could include[cite: 1499]:

* [cite_start]**Handling Hinglish:** Developing models to classify mixed Hindi-English text[cite: 1500].
* [cite_start]**Fine-Grained Classification:** Extending the model to classify news into more specific subcategories (e.g., 'Cricket' within 'Sports')[cite: 1501].
* [cite_start]**Personalized News Delivery:** Using the classification system as a backbone for a personalized news recommendation engine[cite: 1504].
* [cite_start]**Exploring Niche Domains:** Adapting the model for specialized topics like agriculture, finance, or regional development updates[cite: 1502].

---
## 📄 Citation

This project is based on the Bachelor of Technology thesis submitted to the Bhilai Institute of Technology, Durg. If you use this code or the concepts in your research, please cite the original work.

**Author:** Devdeep Sarkar.

---
## ©️ Copyright & Usage

**Copyright (c) 2025 Devdeep Sarkar. All Rights Reserved.**

This project is for demonstration purposes only. The code is proprietary and may not be used, copied, modified, or distributed without the express written permission of the author.
