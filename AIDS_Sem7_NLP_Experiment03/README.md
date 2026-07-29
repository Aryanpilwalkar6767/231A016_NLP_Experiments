# NLP Experiment 3: Text Preprocessing Using NLTK and Regular Expressions

## Aim

To perform various text preprocessing operations on raw textual data using **Natural Language Processing (NLP)** techniques, Regular Expressions, and the NLTK library.

---

## Problem Statement

Raw textual data often contains unnecessary and noisy information such as URLs, email addresses, numbers, emojis, hashtags, user mentions, punctuation, and commonly occurring stop words.

Such elements can reduce the quality of text analysis and increase unnecessary complexity during NLP processing.

The objective of this experiment is to preprocess a given raw text by applying different text-cleaning techniques such as:

- Lowercase conversion
- URL removal
- Email removal
- Number removal
- Emoji removal
- Hashtag and special symbol removal
- Tokenization
- Stop-word removal
- Punctuation removal
- Stemming

The final output should contain cleaner and more standardized textual information suitable for further NLP tasks.

---

## Brief Theory

### Natural Language Processing

Natural Language Processing (NLP) is a branch of Artificial Intelligence that enables computers to process, understand, and analyze human language.

Raw text generally contains a large amount of irrelevant information. Therefore, **text preprocessing** is an important step before performing tasks such as sentiment analysis, text classification, information retrieval, or machine learning.

### Text Preprocessing

Text preprocessing transforms raw text into a cleaner and more structured representation.

The following preprocessing techniques are implemented in this experiment:

### 1. Lowercase Conversion

Text is converted into lowercase so that words with different capitalization are treated as the same word.

For example:

```text
Machine Learning → machine learning
```

### 2. URL Removal

URLs such as:

```text
https://www.example.com
www.kaggle.com
```

are removed using **Regular Expressions (Regex)**.

URLs usually do not provide meaningful linguistic information for many NLP tasks.

### 3. Email Removal

Email addresses present in the text are identified and removed using regular expressions.

### 4. Number Removal

Numerical values such as dates, percentages, record counts, decimal values, and other numbers are removed from the text using Regex.

### 5. Emoji Removal

Emojis are commonly present in social media and informal textual data.

The Python `emoji` library is used to identify and remove emojis from the given text.

### 6. Hashtag and Symbol Removal

Hashtags and unwanted symbols such as `#` and `@` are removed using regular expressions.

### 7. Tokenization

Tokenization is the process of dividing text into smaller units called **tokens**.

For example:

```text
machine learning project
```

becomes:

```text
['machine', 'learning', 'project']
```

The experiment uses NLTK's `word_tokenize()` function for tokenization.

### 8. Stop Word Removal

Stop words are commonly occurring words that may provide little useful information for certain NLP tasks.

Examples include:

```text
the, is, and, of, to, in, on
```

NLTK's English stop-word collection is used to remove these words.

### 9. Punctuation Removal

Punctuation symbols such as:

```text
! , . ? : ; @ # $
```

are removed using Python's `string.punctuation`.

### 10. Stemming

Stemming reduces words to their basic or root-like form by removing prefixes or suffixes.

For example:

```text
studying   → studi
analyzing  → analyz
processing → process
predicting → predict
running    → run
```

The experiment uses the **Porter Stemmer** from NLTK.

---

## Implementation Explanation

### 1. Creating the Input Text

A sample text containing different types of noisy information is created.

The text includes:

- URLs
- Email addresses
- Numbers
- Dates
- Percentages
- Emojis
- Hashtags
- Twitter mentions
- HTML tags
- Repeated punctuation
- Different word forms

This provides suitable data for demonstrating different preprocessing operations.

---

### 2. Converting Text to Lowercase

The complete text is converted into lowercase using:

```python
text_1 = text.lower()
```

This ensures that words with different capitalization are treated consistently.

---

### 3. Removing URLs

A Regular Expression is used to identify and remove URLs.

```python
pattern_1 = r'https?://www\.\S+ | www\.\S+'
text_2 = re.sub(pattern_1, " ", text_1)
```

This removes URLs such as:

```text
https://www.example.com
www.kaggle.com
```

---

### 4. Removing Email Information

Another Regular Expression is applied to remove email-related patterns from the text.

```python
pattern_2 = r'\S\@\S+'
text_3 = re.sub(pattern_2, " ", text_2)
```

This step reduces unwanted email information from the input text.

---

### 5. Removing Numbers

Numerical information is removed using:

```python
pattern_3 = r'\d+\S+'
text_4 = re.sub(pattern_3, " ", text_3)
```

This removes numerical elements such as:

- Dates
- Percentages
- Record counts
- Decimal values
- Other numerical information

---

### 6. Removing Emojis

The `emoji` Python library is installed and imported.

```python
import emoji
```

Emojis are removed using:

```python
text_5 = emoji.replace_emoji(text_4, replace="")
```

This removes emojis such as:

```text
👋 😊 😂
```

from the text.

---

### 7. Removing Hashtags and Special Symbols

A Regular Expression is used to remove hashtag-related information and the `@` symbol.

```python
pattern_4 = r'\#\S+ | \@'
text_6 = re.sub(pattern_4, " ", text_5)
```

This further cleans the text before tokenization.

---

### 8. Tokenizing the Text

The NLTK tokenizer is used to divide the cleaned text into individual tokens.

```python
from nltk import word_tokenize

tokenized = word_tokenize(text_6)
```

The resulting output is a list of individual words and tokens.

Example:

```text
['hey', 'everyone', 'machine', 'learning', 'project', ...]
```

---

### 9. Loading Stop Words and Punctuation

The English stop-word collection is loaded from NLTK.

```python
from nltk.corpus import stopwords
import string

stopwords_english = stopwords.words('english')
```

Python's punctuation collection is also used:

```python
string.punctuation
```

---

### 10. Removing Stop Words and Punctuation

The tokenized words are checked individually.

```python
text_clean = []

for word in tokenized:
    if word not in stopwords_english and word not in string.punctuation:
        text_clean.append(word)
```

Words that are present in the stop-word collection or punctuation list are removed.

The resulting cleaned text contains more meaningful words such as:

```text
['hey', 'everyone', 'really', 'excited', 'completed',
'first', 'machine', 'learning', 'project', 'customer',
'reviews', 'dataset', 'model', 'accuracy', ...]
```

---

### 11. Applying Stemming

Finally, stemming is performed using NLTK's **PorterStemmer**.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

[stemmer.stem(word) for word in text_clean]
```

The stemmer converts different forms of words into shorter root-like forms.

Examples:

```text
studying   → studi
analyzing  → analyz
processing → process
predicting → predict
running    → run
comparing  → compar
```

---

## Results

The experiment successfully applied multiple text preprocessing operations to the given raw text.

The following preprocessing steps were performed successfully:

| Preprocessing Operation | Status |
|---|---|
| Lowercase Conversion | Completed |
| URL Removal | Completed |
| Email Pattern Removal | Completed |
| Number Removal | Completed |
| Emoji Removal | Completed |
| Hashtag/Symbol Removal | Completed |
| Tokenization | Completed |
| Stop Word Removal | Completed |
| Punctuation Removal | Completed |
| Stemming | Completed |

The original text contained several noisy elements including URLs, numbers, emojis, email addresses, hashtags, mentions, and repeated punctuation.

After preprocessing, unnecessary information was reduced and the text was converted into a cleaner collection of meaningful tokens.

Finally, the **Porter Stemmer** was applied to reduce words to their root-like forms.

---

## Conclusion

In this experiment, various **Natural Language Processing text preprocessing techniques** were successfully implemented using Python, Regular Expressions, and the NLTK library.

The raw input text was cleaned by converting it to lowercase and removing URLs, email patterns, numbers, emojis, hashtags, stop words, and punctuation. The cleaned text was then tokenized into individual words.

Finally, **Porter Stemming** was applied to reduce different forms of words to their root-like forms.

This experiment demonstrates the importance of preprocessing in NLP. Clean and standardized textual data can improve the efficiency and effectiveness of further tasks such as **text classification, sentiment analysis, information retrieval, and machine learning-based NLP applications**.

---

## References

1. [NLTK Documentation](https://www.nltk.org/)
2. [NLTK Tokenization](https://www.nltk.org/api/nltk.tokenize.html)
3. [NLTK Stem Package](https://www.nltk.org/api/nltk.stem.html)
4. [NLTK Stopwords Corpus](https://www.nltk.org/howto/corpus.html)
5. [Python Regular Expressions](https://docs.python.org/3/library/re.html)
6. [Python String Documentation](https://docs.python.org/3/library/string.html)
7. [Emoji Python Package](https://pypi.org/project/emoji/)