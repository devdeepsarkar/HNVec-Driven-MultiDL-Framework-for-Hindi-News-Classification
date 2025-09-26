# HNVec-Driven-MultiDL-Framework-for-Hindi-News-Classification
A hybrid deep learning framework (CNN-LR-SVM) using novel HNVec embeddings to classify Hindi news articles with more than 85% accuracy.
# HNVec-Driven MultiDL Framework for Hindi News Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A novel deep learning framework for the content-based classification of Hindi news articles. [cite_start]This project introduces **HNVec (Hindi News Vectorizer)**, a custom word embedding model tailored for the semantic and contextual nuances of the Hindi language[cite: 92]. [cite_start]It also implements a hybrid learning model (**HLM-CLS**) that combines a Convolutional Neural Network (CNN) with Logistic Regression (LR) and a Support Vector Machine (SVM) to achieve high classification accuracy[cite: 94].

This system was developed to address the challenges of text classification for low-resource languages, demonstrating significant improvements over baseline vectorization methods like TF-IDF and FastText.

## Key Features

* **Custom Hindi Embeddings (HNVec):** A novel, distance-based context-sensitive vectorizer designed specifically to capture the morphological richness of Hindi text.
* **Hybrid Learning Architecture:** A multi-layer model that leverages the feature extraction power of CNNs and the robust classification capabilities of SVM and LR.
* **High Performance:** Achieved an **accuracy of 88.61%** and an F1-score of 88.51% on the test dataset.
* **Superior to Baselines:** Outperformed standard TF-IDF and FastText embeddings by 15-25% in precision, proving the effectiveness of domain-specific embeddings.
* **Comprehensive Preprocessing:** Includes a robust pipeline for cleaning and preparing Hindi text, handling tokenization, stopword removal, and normalization.

---

## Project Architecture

The framework follows a multi-stage pipeline that integrates feature extraction, feature reduction, and final classification.


**1. Feature Extraction:**
* Input Hindi news articles are processed and cleaned.
* The novel **HNVec vectorizer** converts the text into meaningful, context-aware embeddings.
* Simultaneously, a **Convolutional Neural Network (CNN)** processes the tokenized text to extract high-level features and local patterns.

**2. Feature Combination & Reduction:**
* The features from HNVec and the CNN are combined into a single, enriched feature set.
* This combined set is then used to train parallel **Logistic Regression (LR)** and **Support Vector Machine (SVM)** models. The probabilistic outputs from these models act as a reduced feature set.

**3. Final Classification:**
* A final **SVM classifier** is trained on this reduced feature set to make the ultimate prediction, categorizing the news article into one of the predefined classes (e.g., Sports, Politics, Entertainment).

---

## Performance & Results

The HLM-CLS model combined with our custom HNVec embeddings delivered the best performance, validating our approach.

| Metric    | Score     |
| :-------- | :-------- |
| **Accuracy** | **88.61%** |
| Precision | 88.50%    |
| Recall    | 88.61%    |
| F1-Score  | 88.51%    |

_These results were achieved with a 70:30 train-test split_.


The model showed strong convergence with minimal training loss, demonstrating its ability to generalize well without overfitting.

---

## Tech Stack

* **Language:** Python 3.9+
* **Core Libraries:**
    * TensorFlow / Keras
    * Scikit-learn
    * Pandas
    * NumPy
* **Environment:** Google Colab

---

## Future Work

Based on the findings of our research, future extensions of this work could include:

* **Handling Hinglish:** Developing models to classify mixed Hindi-English text.
* **Fine-Grained Classification:** Extending the model to classify news into more specific subcategories (e.g., 'Cricket' within 'Sports').
* **Personalized News Delivery:** Using the classification system as a backbone for a personalized news recommendation engine.
* **Exploring Niche Domains:** Adapting the model for specialized topics like agriculture, finance, or regional development updates.

---

## Citation

This project is based on the Bachelor of Technology thesis submitted to the Bhilai Institute of Technology, Durg. If you use this code or the concepts in your research, please cite the original work.

**Authors:** Devdeep Sarkar.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
