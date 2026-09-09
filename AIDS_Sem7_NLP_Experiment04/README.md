
# NLP Experiment 4 – Text Preprocessing, Lemmatization and Morphological Analysis

## Aim

To perform basic Natural Language Processing (NLP) operations such as text preprocessing, tokenization, punctuation removal, lemmatization, and morphological analysis using Python and the NLTK library.

## Problem Statement

Natural language text contains punctuation marks, different word forms, and other elements that need to be processed before meaningful analysis can be performed. The objective of this experiment is to preprocess a given text, tokenize it into individual words, remove punctuation, perform lemmatization, and analyze the morphological structure of the resulting words.

## Brief Theory

Natural Language Processing (NLP) is a branch of Artificial Intelligence that deals with the processing and analysis of human language. Since raw text is generally unstructured, preprocessing is an important step before applying further NLP techniques.

### Text Preprocessing

Text preprocessing involves cleaning and normalizing raw text so that it can be easily processed. In this experiment, the input text is converted into lowercase before further processing.

### Tokenization

Tokenization is the process of dividing text into smaller units called tokens. In this experiment, the `word_tokenize()` function from NLTK is used to divide the input text into individual words and punctuation tokens.

### Punctuation Removal

Punctuation marks such as `!`, `?`, `,`, and other symbols can create unnecessary tokens during text processing. Python's `string.punctuation` is used to identify and remove these punctuation characters.

### Lemmatization

Lemmatization is the process of converting a word into its meaningful base or dictionary form. The `WordNetLemmatizer` from NLTK is used in this experiment.

The result of lemmatization depends on the Part of Speech (POS) assigned to a word. For example, when `walking` is treated as a verb, it is converted to `walk`. When it is treated as a noun, it remains `walking`.

### Morphological Analysis

Morphological analysis studies the structure and formation of words. In this experiment, words are analyzed and classified as base/root forms, inflectional forms, or derivational forms.

An inflectional form changes the grammatical form of a word without creating a new word. For example, `walking` is an inflectional form of `walk`.

A derivational form creates a new word or changes its meaning. For example, `computation` is analyzed as a derivational form formed using the suffix `-ation`.

## Technologies and Libraries Used

- Python
- NLTK (Natural Language Toolkit)
- `nltk.tokenize`
- `nltk.stem`
- `string`

## Input

The following text is used as input for the experiment:

```text
:)The walking ? brown fox jumping over the lazy dog!!,computation
```

## Implementation

### 1. Import Required Libraries

The required Python and NLTK libraries are imported for tokenization, lemmatization, and punctuation processing.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
```

### 2. Download Required NLTK Resources

The required NLTK resources are downloaded before performing tokenization and lemmatization.

```python
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

The `punkt_tab` resource is required for tokenization, while `wordnet` and `omw-1.4` support WordNet-based lemmatization.

### 3. Display Punctuation Characters

The `string.punctuation` constant is used to display the punctuation characters that can be removed during preprocessing.

```python
print(string.punctuation)
```

The output contains punctuation characters such as:

```text
!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
```

### 4. Convert Text to Lowercase

The input text is converted into lowercase using the `lower()` function.

```python
text_lower = text.lower()
print(text_lower)
```

The processed text becomes:

```text
:)the walking ? brown fox jumping over the lazy dog!!,computation
```

### 5. Tokenization

The lowercase text is tokenized using NLTK's `word_tokenize()` function.

```python
tokens = word_tokenize(text_lower)
print(tokens)
```

The text is divided into individual tokens, including both words and punctuation marks.

The resulting tokens are:

```text
[':', ')', 'the', 'walking', '?', 'brown', 'fox', 'jumping', 'over', 'the', 'lazy', 'dog', '!', '!', ',', 'computation']
```

### 6. Remove Punctuation

A loop is used to remove tokens that are present in `string.punctuation`.

```python
text_clean = []

for word in tokens:
    if word not in string.punctuation:
        text_clean.append(word)

print(text_clean)
```

The cleaned text becomes:

```text
['the', 'walking', 'brown', 'fox', 'jumping', 'over', 'the', 'lazy', 'dog', 'computation']
```

### 7. Perform Lemmatization

A `WordNetLemmatizer` object is created and the cleaned words are lemmatized by treating them as verbs.

```python
lemmatizer = WordNetLemmatizer()

lemmas = [
    lemmatizer.lemmatize(tok, pos='v')
    for tok in text_clean
]

print(lemmas)
```

The resulting lemmas are:

```text
['the', 'walk', 'brown', 'fox', 'jump', 'over', 'the', 'lazy', 'dog', 'computation']
```

The words `walking` and `jumping` are converted to `walk` and `jump` respectively.

### 8. Demonstrate POS-Based Lemmatization

The experiment also demonstrates how the Part of Speech affects the result of lemmatization.

```python
lemmatizer.lemmatize("walking", pos='n')
```

When `walking` is treated as a noun, the output is:

```text
'walking'
```

This demonstrates that the same word can produce different results depending on the assigned Part of Speech.

### 9. Perform Morphological Analysis

A dictionary is created to store the morphological information of each cleaned token.

```python
morphological_info = {}

for token, lemma in zip(text_clean, lemmas):
    analysis = ""

    if token.endswith("ation"):
        analysis = "Derivational: '{}' + -ation -> result of action".format(
            token.replace('ation', '')
        )

    elif token.endswith("er") and token != lemma:
        analysis = "Derivational: '{}' + -er -> person who does the action".format(
            token.replace('er', '')
        )

    elif token.endswith("ing") and token != lemma:
        analysis = "Inflectional: Present participle/continuous form of '{}'".format(
            lemma
        )

    elif token != lemma:
        analysis = "Inflectional or irregular form of '{}'".format(lemma)

    else:
        analysis = "Base/root form"

    morphological_info[token] = analysis
```

### 10. Morphological Analysis Output

The generated morphological information is:

```text
{
'the': 'Base/root form',
'walking': "Inflectional: Present participle/continuous form of 'walk'",
'brown': 'Base/root form',
'fox': 'Base/root form',
'jumping': "Inflectional: Present participle/continuous form of 'jump'",
'over': 'Base/root form',
'lazy': 'Base/root form',
'dog': 'Base/root form',
'computation': "Derivational: 'comput' + -ation -> result of action"
}
```

The analysis identifies `walking` and `jumping` as inflectional forms, while `computation` is identified as a derivational form. Words such as `the`, `brown`, `fox`, `over`, `lazy`, and `dog` are identified as base/root forms.

### 11. Additional Lemmatization Examples

The experiment also includes additional words with different Parts of Speech to demonstrate POS-based lemmatization.

```python
words = [
    ("playing", "v"),
    ("played", "v"),
    ("studies", "v"),
    ("running", "v"),
    ("better", "a"),
    ("car", "a")
]

pos_map = {
    "n": wordnet.NOUN,
    "v": wordnet.VERB,
    "a": wordnet.ADJ,
    "r": wordnet.ADV
}
```

This demonstrates that assigning the correct Part of Speech is important for obtaining meaningful lemmas.

## Workflow

The overall workflow of the experiment can be represented as:

```text
Raw Text
    ↓
Convert to Lowercase
    ↓
Tokenization
    ↓
Remove Punctuation
    ↓
Clean Text
    ↓
Lemmatization
    ↓
Morphological Analysis
    ↓
Final Processed Output
```

## Results

The experiment successfully performed the complete text preprocessing and analysis process on the given input. The original text was converted into lowercase and tokenized into individual words and punctuation tokens. The punctuation marks were then removed to obtain a clean list of words.

Lemmatization was successfully performed using `WordNetLemmatizer` with the verb Part of Speech. The words `walking` and `jumping` were converted into `walk` and `jump`, while words such as `the`, `brown`, `fox`, `lazy`, and `dog` remained unchanged.

The experiment also demonstrated POS-based lemmatization by treating `walking` as a noun, where the output remained `walking`. Morphological analysis successfully classified the words into base/root, inflectional, and derivational forms.

## Key Learning Outcomes

* Understand the basic Natural Language Processing workflow.
* Understand the importance of text preprocessing.
* Learn how to convert text into lowercase.
* Learn how to tokenize text using NLTK.
* Understand how punctuation can be identified and removed.
* Perform lemmatization using `WordNetLemmatizer`.
* Understand the importance of Part of Speech in lemmatization.
* Perform basic morphological analysis.
* Differentiate between base/root, inflectional, and derivational forms.
* Use Python and NLTK for basic NLP operations.

## Conclusion

The experiment successfully demonstrated fundamental Natural Language Processing techniques using Python and the NLTK library. The given text was preprocessed by converting it to lowercase, tokenizing it, and removing punctuation.

Lemmatization was then performed to obtain meaningful base forms of words, while POS-based lemmatization demonstrated the importance of grammatical information. Finally, morphological analysis was used to classify words into base/root, inflectional, and derivational forms. These techniques form an important foundation for further NLP applications such as text classification, sentiment analysis, information retrieval, and machine learning-based text processing.