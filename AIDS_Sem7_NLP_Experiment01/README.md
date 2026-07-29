# Informal-to-Formal Text Conversion using Hugging Face Transformers

## Aim

The aim of this project is to develop a Natural Language Processing (NLP) system that automatically converts informal English text into grammatically correct and professionally written formal text using a fine-tuned Hugging Face Transformer model. The system is designed to improve written communication by transforming casual language, slang, abbreviations, and internet-style expressions into formal English while preserving the original meaning.

---

# Problem Statement

With the rapid growth of social media, instant messaging, emails, and online communication, people frequently use informal language containing abbreviations, slang, emojis, grammatical mistakes, and conversational expressions. While such language is acceptable in casual conversations, it is unsuitable for academic writing, professional emails, business reports, and official documentation.

Manually converting informal text into formal language is both time-consuming and inconsistent. Therefore, there is a need for an automated system capable of accurately rewriting informal text into formal English while preserving the intended meaning and context.

This project addresses this challenge by fine-tuning a pre-trained Transformer model from Hugging Face using an informal-to-formal text dataset.

---

# Brief Theory

Natural Language Processing (NLP) is a branch of Artificial Intelligence (AI) that enables computers to understand, process, and generate human language.

Earlier approaches to text transformation relied on handcrafted grammar rules and dictionaries. Although useful for simple corrections, these methods struggled with contextual understanding and modern internet language.

Transformer-based models have significantly improved NLP tasks by learning contextual relationships between words using the **Self-Attention Mechanism**. Instead of relying on manually designed rules, Transformers learn language patterns directly from large datasets.

This project uses the **T5 (Text-to-Text Transfer Transformer)** model available through the Hugging Face Transformers library. T5 treats every NLP task as a text-to-text generation problem, making it well-suited for converting informal sentences into formal ones.

### Example

**Input**

```text
hey prof i cant submit coz my laptop isnt working
```

**Output**

```text
Dear Professor, I cannot submit because my laptop is not working.
```

By fine-tuning the T5 model on paired informal-formal sentence datasets, the model learns how to generate professional and grammatically correct text while preserving the original meaning.

---

# Implementation Explanation

The project follows a complete NLP pipeline consisting of the following stages.

## 1. Dataset Collection

- Collected an Informal-to-Formal sentence dataset.
- Dataset consists of paired sentences:
  - **Input:** Informal sentence
  - **Output:** Formal sentence

Example:

| Informal Text | Formal Text |
|--------------|-------------|
| lol this is awesome | This is wonderful. |
| brb | I will be back shortly. |

---

## 2. Dataset Loading

A custom Python function loads JSON datasets by:

- Checking whether the file exists
- Reading JSON safely
- Supporting multiple JSON formats
- Converting data into a Pandas DataFrame
- Adding a source column for dataset tracking

---

## 3. Data Preprocessing

To improve data quality before training, the following preprocessing steps are performed:

- Remove duplicate records
- Handle missing values
- Remove unnecessary URLs
- Remove emojis (optional)
- Remove extra white spaces
- Normalize sentence formatting

These preprocessing steps help improve the consistency and quality of the training data.

---

## 4. Exploratory Data Analysis (EDA)

The dataset is analyzed before training to better understand its characteristics.

The analysis includes:

- Total number of samples
- Dataset dimensions
- Average sentence length
- Vocabulary statistics
- Most frequent words
- Sentence length distribution
- Word Cloud visualization

EDA helps identify potential issues such as duplicate records, imbalanced sentence lengths, or noisy data.

---

## 5. Tokenization

The Hugging Face tokenizer converts text into numerical tokens that can be processed by the Transformer model.

```
Input Sentence
      ↓
Tokenizer
      ↓
Token IDs
```

Tokenization enables the model to understand and process natural language efficiently.

---

## 6. Model Selection

The project uses the **T5-small** model from Hugging Face.

Reasons for selecting T5-small:

- Pre-trained on massive text corpora
- Designed specifically for text generation tasks
- Lightweight enough for Google Colab
- High performance with limited training data
- Easy integration using Hugging Face Transformers

---

## 7. Model Fine-Tuning

The pre-trained T5 model is fine-tuned using the informal-formal sentence dataset.

Training procedure:

- Split dataset into Training, Validation, and Test sets
- Tokenize input and output text
- Train the model for multiple epochs
- Optimize model parameters using backpropagation
- Save the best-performing model

---

## 8. Model Evaluation

The trained model is evaluated using standard Natural Language Processing metrics.

Evaluation metrics include:

- BLEU Score
- ROUGE-1
- ROUGE-2
- ROUGE-L

Additionally, sample predictions are compared with expected outputs to evaluate the quality of generated formal text.

---

## 9. Prediction

After training, users can provide any informal sentence for conversion.

### Example

**Input**

```text
gonna be late coz traffic is crazy
```

**Predicted Output**

```text
I am going to be late because the traffic is extremely heavy.
```

The model generates grammatically correct and contextually appropriate formal text.

---

## 10. Model Saving

The trained model and tokenizer are saved locally after fine-tuning.

This allows the model to be reused for inference without retraining.

---

# Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NLTK
- Evaluate Library (BLEU & ROUGE)

---

# Conclusion

This project demonstrates the practical application of Transformer-based Natural Language Processing for converting informal English text into formal language. By fine-tuning the T5 Transformer model, the system successfully generates grammatically correct, professional, and context-aware formal text while preserving the original meaning.

The project also illustrates the complete NLP workflow, including dataset loading, preprocessing, exploratory data analysis, tokenization, model fine-tuning, evaluation, and prediction. The developed system can assist students, professionals, and organizations in improving the quality of written communication for academic, business, and professional use.

---

# References

1. Raffel, C., et al. (2020). **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).** Journal of Machine Learning Research.

   https://jmlr.org/papers/v21/20-074.html

2. Hugging Face Transformers Documentation

   https://huggingface.co/docs/transformers

3. Hugging Face Datasets Documentation

   https://huggingface.co/docs/datasets

4. Wolf, T., et al. (2020). **Transformers: State-of-the-Art Natural Language Processing.** EMNLP 2020.

   https://aclanthology.org/2020.emnlp-demos.6/

5. Papineni, K., et al. (2002). **BLEU: A Method for Automatic Evaluation of Machine Translation.**

6. Lin, C. Y. (2004). **ROUGE: A Package for Automatic Evaluation of Summaries.**

7. Pedregosa, F., et al. (2011). **Scikit-learn: Machine Learning in Python.**

8. Bird, S., Klein, E., & Loper, E. (2009). **Natural Language Processing with Python.**

9. Paszke, A., et al. (2019). **PyTorch: An Imperative Style, High-Performance Deep Learning Library.**

10. Informal-to-Formal Sentence Pair Dataset (JSON format) used for fine-tuning the T5 model.