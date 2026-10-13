# NLP Experiment 10 – Exploratory Data Analysis and Text Preprocessing on the IMDB Dataset

## Aim
To perform Exploratory Data Analysis (EDA) and text preprocessing on the IMDB movie review dataset using Python. The experiment analyzes review length, vocabulary size, frequent words, word clouds, and bigrams to understand patterns in textual data.

## Problem Statement
Textual datasets contain unstructured data that must be cleaned and analyzed before applying Natural Language Processing (NLP) techniques. This experiment uses IMDB movie reviews to examine the structure of text, identify frequently occurring words and word combinations, and prepare the reviews for further NLP tasks.

## Objectives
- Load and explore the IMDB movie review dataset.
- Examine dataset dimensions, columns, labels, missing values, and duplicate rows.
- Calculate character count, word count, unique word count, and average word length.
- Clean text by converting it to lowercase and removing HTML tags, special characters, and extra spaces.
- Visualize the distribution of review lengths.
- Calculate total word count and vocabulary size.
- Identify frequently occurring words after removing English stop words.
- Generate a word cloud to visualize frequent terms.
- Extract and analyze frequently occurring bigrams.

## Technologies and Libraries Used

| Technology / Library | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| Hugging Face Datasets | Loading the IMDB dataset |
| Regular Expressions (`re`) | Text cleaning |
| Matplotlib | Data visualization |
| Scikit-learn | Stop-word filtering and n-gram extraction |
| Collections (`Counter`) | Counting word frequencies |
| WordCloud | Generating word cloud visualizations |

## Dataset Description
The experiment uses the **IMDB movie review dataset**, loaded using the Hugging Face `datasets` library.

Dataset identifier: `stanfordnlp/imdb`

The notebook combines the training and testing splits into a single Pandas DataFrame for exploratory analysis.

The main columns used are:

- **text:** Contains the movie review.
- **label:** Contains the original sentiment label, renamed to `labels` in the notebook.

The experiment examines the dataset's labels and reviews to understand its structure and textual characteristics.

## Implementation

### 1. Loading the Dataset
The IMDB dataset is loaded using `load_dataset()` from the Hugging Face Datasets library. The training and testing splits are converted into Pandas DataFrames and combined into one DataFrame.

### 2. Dataset Exploration
The dataset is examined using Pandas to understand its structure.

The notebook:
- Renames the `label` column to `labels`.
- Displays the dataset shape and column names.
- Counts the occurrences of each label.
- Displays the first few records.
- Checks the total number of labels.

### 3. Data Quality and Text Statistics
The notebook checks for missing values and duplicate rows. It then calculates the following statistics for each review:

- **Character Count:** Total number of characters in a review.
- **Word Count:** Total number of words in a review.
- **Unique Word Count:** Number of distinct words in a review.
- **Average Word Length:** Average number of characters per word in a review.

These statistics help understand review length, vocabulary variation, and the overall characteristics of the dataset.

### 4. Text Preprocessing
A custom `clean_text()` function is used to clean the movie reviews.

The preprocessing steps are:

1. Convert all text to lowercase.
2. Remove HTML tags using regular expressions.
3. Remove characters other than lowercase English letters and whitespace.
4. Replace multiple whitespace characters with a single space.
5. Remove leading and trailing whitespace.

The cleaned reviews are stored in a new column called `clean_text`, while the original reviews remain available in the `text` column.

### 5. Review Length Distribution
Matplotlib is used to create a histogram showing the distribution of review word counts.

The histogram uses:
- X-axis: Word count.
- Y-axis: Frequency.
- 40 bins to display the distribution.

This visualization helps identify common review lengths and understand variation in the size of the reviews.

### 6. Total Word Count and Vocabulary Size
All cleaned reviews are combined into a single text string and split into individual words.

Two measurements are calculated:

- **Total Word Count:** Total number of word occurrences across the cleaned dataset.
- **Vocabulary Size:** Number of distinct words appearing in the cleaned dataset.

These measurements help describe the scale and diversity of the textual data.

### 7. Stop-Word Removal and Frequent Word Analysis
The notebook uses Scikit-learn's `CountVectorizer` to obtain English stop words.

Stop words are common words such as articles and prepositions that may contribute less to certain frequency-based analyses.

The experiment removes these stop words from the combined word list and uses Python's `Counter` to calculate word frequencies.

The 20 most frequent remaining words are displayed in a horizontal bar chart.

### 8. Word Cloud Generation
The `WordCloud` library is used to create a word cloud from the filtered words.

The visualization uses a white background and excludes the selected stop words.

A word cloud provides a visual representation of frequently occurring terms, with more frequent words generally appearing more prominently.

### 9. Bigram Analysis
A **bigram** is a sequence of two consecutive words.

The notebook uses Scikit-learn's `CountVectorizer` with `ngram_range=(2, 2)` and English stop-word removal to extract bigrams from the cleaned reviews.

The frequency of each bigram is calculated, stored in a DataFrame, and sorted in descending order. The notebook then displays the top 10 bigrams.

Bigram analysis helps identify commonly occurring two-word expressions and patterns in movie reviews.

## Key Concepts

| Concept | Description |
|---|---|
| EDA | Exploring and summarizing the structure and characteristics of data |
| Text Preprocessing | Cleaning raw text before further NLP analysis |
| Vocabulary Size | Number of distinct words in the analyzed text |
| Stop Words | Common words removed for selected text analysis tasks |
| Word Frequency | Number of occurrences of a word |
| Word Cloud | Visualization of words based on their frequency |
| Bigram | A sequence of two consecutive words |
| Regular Expression | A pattern used to find or replace text |

## Expected Learning Outcomes
After completing this experiment, students should be able to:

- Load and explore a text dataset using Pandas.
- Identify missing values and duplicate records.
- Calculate basic text statistics.
- Apply regular expressions for text cleaning.
- Visualize the distribution of review lengths.
- Calculate total word count and vocabulary size.
- Filter stop words and analyze frequent terms.
- Generate word clouds.
- Extract and rank bigrams using Scikit-learn.

## Applications
The techniques used in this experiment can support:

- Movie review analysis.
- Sentiment analysis preprocessing.
- Customer feedback analysis.
- Social media text analysis.
- Product review mining.
- Text summarization and other NLP workflows.
- Identification of common phrases in large text collections.

## Limitations
- Removing punctuation and non-English characters may discard useful information.
- Stop-word removal may remove words that are important in some contexts.
- Word frequency alone does not reveal the sentiment or meaning of a review.
- Bigrams capture only two-word sequences and do not fully represent sentence context.
- Combining the training and testing splits is useful for exploratory analysis, but the combined data should not be used indiscriminately for model training and evaluation because it can cause data leakage.

## Conclusion
This experiment demonstrates exploratory data analysis and text preprocessing on the IMDB movie review dataset. It covers dataset inspection, data quality checks, text statistics, cleaning, review-length visualization, vocabulary analysis, stop-word filtering, word cloud generation, and bigram extraction. These techniques provide a foundation for understanding textual data and preparing it for further NLP tasks, including sentiment analysis.
