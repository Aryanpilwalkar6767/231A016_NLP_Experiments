# NLP Experiment 5 – N-Gram Language Model and Perplexity

## Aim

To design and implement a language model using N-gram techniques, generate and predict words from a text corpus, and evaluate the language model using perplexity.

## Problem Statement

A language model predicts the probability of a sequence of words and can be used to predict the next word in a sentence. The objective of this experiment is to preprocess a text corpus, generate bigrams and trigrams, calculate their frequencies and probabilities, predict the next word, generate a sequence of words, and evaluate a trigram language model using perplexity.

## Brief Theory

A language model is a statistical or computational model that learns patterns in natural language and assigns probabilities to sequences of words. N-gram models are one of the basic approaches used for language modeling.

### N-Gram

An N-gram is a sequence of `N` consecutive words from a text.

Common types of N-grams include:

- **Unigram:** Sequence of one word.
- **Bigram:** Sequence of two consecutive words.
- **Trigram:** Sequence of three consecutive words.

For example, given:

```text
Engineering is a fast growing field
````

The bigrams are:

```text
Engineering is
is a
a fast
fast growing
growing field
```

The trigrams are:

```text
Engineering is a
is a fast
a fast growing
fast growing field
```

### Text Preprocessing

Before generating N-grams, the text is converted to lowercase and tokenized into individual words. Punctuation and English stopwords are removed using NLTK.

### Bigram Frequency

A bigram frequency distribution counts how many times each pair of consecutive words occurs in the corpus. This frequency information can be used to determine the most likely word following a given word.

### Conditional Frequency Distribution

A Conditional Frequency Distribution (CFD) stores the frequency of an event based on a given condition. In this experiment, it is used to determine the words that are likely to follow a particular word.

### Trigram Language Model

A trigram language model predicts the next word using the previous two words. The probability can be represented as:

```text
P(w3 | w1, w2)
```

where `w1` and `w2` are the previous two words and `w3` is the predicted next word.

The experiment creates a model that stores the frequency of the third word for every pair of consecutive words and then converts these frequencies into probabilities.

### Perplexity

Perplexity is a metric used to evaluate a language model. It measures how well the model predicts a given test sentence. Lower perplexity generally indicates that the language model assigns higher probability to the test sequence.

The perplexity used in this experiment is calculated as:

```text
Perplexity = exp(-Σ log(P) / N)
```

where `P` represents the predicted probability of each word and `N` represents the number of predicted words.

## Technologies and Libraries Used

* Python
* NLTK (Natural Language Toolkit)
* `nltk.tokenize`
* `nltk.corpus`
* `nltk.bigrams`
* `nltk.trigrams`
* `nltk.FreqDist`
* `nltk.ConditionalFreqDist`
* `collections.defaultdict`
* `math`

## Datasets and Corpora Used

Two text sources are used in this experiment:

1. A manually defined corpus related to science, engineering, and architecture.
2. The **Reuters Corpus** provided by NLTK for building a larger trigram language model.

The Reuters corpus is downloaded using NLTK's corpus utilities.

## Part 1 – Bigram and Trigram Language Model

### Input Corpus

The following corpus is used:

```text
Science is a most fast growing field. Engineering and architecture make the basic necessity of survival. Engineering is one of the most opted technology field. Engineering has contributed in a large manner. Architecture provided the basic foundation of human civilization. Engineering and architecture are hence required to sustain humanity
```

A test sentence is also defined:

```text
Engineering is a fast growing field
```

### 1. Import Required Libraries

The required NLTK libraries and Python collections are imported.

```python
import nltk
from nltk import word_tokenize, bigrams, FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords
from collections import defaultdict
```

### 2. Download NLTK Resources

The required resources for tokenization and stopword removal are downloaded.

```python
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 3. Text Preprocessing

A preprocessing function is created to convert the input text to lowercase, tokenize it, and remove punctuation and stopwords.

```python
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    tokens = [
        token for token in tokens
        if token.isalnum() and token not in stop_words
    ]

    return tokens
```

The corpus is passed through the preprocessing function:

```python
tokens = preprocess(corpus)
print(tokens)
```

This produces a cleaned sequence of words that can be used to generate N-grams.

### 4. Generate Bigrams

Bigrams are generated using NLTK's `bigrams()` function.

```python
bigrams_list = list(bigrams(tokens))
print(bigrams_list)
```

Each bigram contains two consecutive tokens from the preprocessed corpus.

### 5. Calculate Bigram Frequency

The frequency of each bigram is calculated using `FreqDist`.

```python
bigrams_freq = FreqDist(bigrams_list)
bigrams_freq
```

This identifies how frequently each pair of words occurs in the corpus.

### 6. Calculate Conditional Frequencies

A `ConditionalFreqDist` is created from the bigrams.

```python
cfd = ConditionalFreqDist(bigrams_list)
print(cfd.items())
```

The conditional frequency distribution associates each word with the words that occur immediately after it.

### 7. Generate Trigrams

Trigrams are generated using NLTK's `trigrams()` function.

```python
from nltk import trigrams

trigrams_list = list(trigrams(tokens))
print(trigrams_list)
```

Each trigram contains three consecutive tokens.

### 8. Calculate Trigram Frequency

The frequency of each trigram is calculated using `FreqDist`.

```python
trigrams_freq = FreqDist(trigrams_list)
trigrams_freq
```

This provides the number of occurrences of each three-word sequence.

### 9. Create Next-Word Prediction Function

A function is created to predict the most frequent next word for a given input word.

```python
def predict_next_word(word):
    word = word.lower()

    if word in cfd:
        next_words = cfd[word].max()
        return next_words
    else:
        return "Word not found in the corpus"
```

The function searches the conditional frequency distribution and returns the most frequently occurring next word.

### 10. Test Next-Word Prediction

The prediction function is tested using:

```python
start_word = "science"

next_word = predict_next_word(start_word)

print(
    f"The next word after '{start_word}' is '{next_word}'."
)
```

The model uses the learned bigram relationships to predict the next word after `science`.

### 11. Generate a Sequence of Words

A function is created to repeatedly predict the next word and generate a sequence.

```python
def generate_sequence(start_word, num_words):
    sequence = [start_word]
    current_word = start_word

    for _ in range(num_words):
        next_word = predict_next_word(current_word)

        if next_word == "Word not found in the corpus":
            break

        sequence.append(next_word)
        current_word = next_word

    return " ".join(sequence)
```

The sequence is generated using:

```python
generated_sequence = generate_sequence("science", 9)

print(
    f"Generated sequence starting from 'science': "
    f"{generated_sequence}"
)
```

This demonstrates how a simple bigram model can generate a sequence of words based on previously observed word relationships.

# Part 2 – Trigram Language Model Using Reuters Corpus

## Reuters Corpus

The Reuters corpus available through NLTK is used to construct a larger language model.

### 1. Import Required Libraries

```python
import nltk
from nltk.corpus import reuters
from nltk import bigrams, FreqDist, ConditionalFreqDist
from nltk import trigrams, ngrams
from collections import defaultdict
```

### 2. Download the Reuters Corpus

```python
nltk.download('reuters')
nltk.download('punkt')
```

### 3. Explore the Corpus

The file IDs available in the Reuters corpus are obtained.

```python
file_ids = reuters.fileids()

print(len(file_ids))
print(file_ids[:10])
```

The total number of words in the Reuters corpus can also be checked.

```python
print(len(reuters.words()))
print(reuters.words()[91:100])
```

A particular Reuters document is accessed using its file ID.

```python
words = reuters.words(file_ids[10787])
print(words)
```

### 4. Tokenize the Reuters Corpus

All words from the Reuters corpus are joined and tokenized.

```python
words = nltk.word_tokenize(" ".join(reuters.words()))

print(len(words))
```

This produces a large sequence of tokens that is used to construct the trigram model.

### 5. Create Trigrams

Trigrams are generated from the Reuters corpus.

```python
tri_grams = list(trigrams(words))

print(tri_grams[:10])
```

Each trigram consists of two context words followed by a possible next word.

### 6. Create the Trigram Model

A nested dictionary is created to store the frequency of the third word for each pair of preceding words.

```python
model = defaultdict(lambda: defaultdict(lambda: 0))
```

### 7. Count Frequency of Co-occurrence

The frequency of each third word occurring after a particular pair of words is counted.

```python
tri_grams = list(trigrams(words))

for w1, w2, w3 in tri_grams:
    model[(w1, w2)][w3] += 1
```

For example, the model stores information in the form:

```text
(w1, w2) → w3
```

along with the number of times this combination occurs.

### 8. Convert Frequencies into Probabilities

The stored frequencies are converted into conditional probabilities.

```python
for w1_w2 in model:
    total_count = sum(model[w1_w2].values())

    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count
```

The probabilities for all possible third words following the same two-word context are normalized so that their probabilities sum to approximately 1.

### 9. Predict the Next Word

A function is created to predict the most probable next word given two preceding words.

```python
def predict_next_word(w1, w2):
    next_words = model.get((w1, w2), {})

    if next_words:
        return max(next_words, key=next_words.get)
    else:
        return "Word not found in the corpus"
```

The model selects the word with the highest conditional probability.

### 10. Test Next-Word Prediction

The model is tested with the two-word context:

```python
print(predict_next_word("the", "price"))
```

The model searches for the most probable word following the sequence `the price` in the Reuters corpus.

## Perplexity Evaluation

### 1. Define the Perplexity Function

The model is evaluated using a test sentence and the probabilities learned from the trigram model.

```python
import math

def calculate_perplexity(test_sentence):
    test_tokens = word_tokenize(test_sentence)

    log_prob_sum = 0

    for i in range(2, len(test_tokens)):
        w1 = test_tokens[i - 2]
        w2 = test_tokens[i - 1]
        w3 = test_tokens[i]

        prob = model[(w1, w2)].get(w3, 1e-10)

        log_prob_sum += math.log(prob)

    perplexity = math.exp(
        -log_prob_sum / (len(test_tokens) - 2)
    )

    return perplexity
```

A small probability value of `1e-10` is used when the required trigram is not found in the model, preventing a logarithm of zero.

### 2. Test Sentence

The following sentence is used to evaluate the language model:

```text
ASIAN EXPORTERS FEAR DAMAGE
```

The perplexity is calculated using:

```python
test_sentence = "ASIAN EXPORTERS FEAR DAMAGE"

perplexity = calculate_perplexity(test_sentence)

print("Test Sentence:", test_sentence)
print("Perplexity:", perplexity)
```

## Results

The experiment successfully implemented both bigram and trigram language models.

### Part 1 Results

The manually defined corpus was successfully preprocessed by converting the text to lowercase, tokenizing it, and removing punctuation and stopwords.

Bigrams and trigrams were generated from the cleaned corpus, and their frequencies were calculated using NLTK frequency distributions.

A conditional frequency distribution was used to identify the most frequent word following a given word. The model successfully predicted the next word for a given starting word and generated a sequence of words based on repeated next-word predictions.

### Part 2 Results

The Reuters corpus was successfully loaded using NLTK. The corpus was tokenized and used to generate trigrams.

A trigram model was constructed by counting the co-occurrence of three consecutive words. These frequency counts were then normalized into conditional probabilities.

The model was used to predict the most probable word following a given pair of words, such as:

```text
the price
```

The model was also evaluated using the test sentence:

```text
ASIAN EXPORTERS FEAR DAMAGE
```

The calculated perplexity provides a measure of how well the trained trigram model predicts the given test sentence.

## Key Learning Outcomes

* Understand the concept of statistical language modeling.
* Understand the concept of N-grams.
* Generate bigrams and trigrams from a text corpus.
* Perform text preprocessing using NLTK.
* Calculate N-gram frequencies.
* Use conditional frequency distributions for next-word prediction.
* Build a basic bigram-based word prediction system.
* Generate word sequences using a language model.
* Build a trigram language model using the Reuters corpus.
* Convert trigram frequencies into conditional probabilities.
* Predict the next word using two previous words.
* Understand and calculate perplexity.
* Evaluate the performance of a language model using a test sentence.

## Conclusion

The experiment successfully demonstrated the implementation of N-gram-based language models using Python and NLTK. The first part used a manually defined corpus to perform preprocessing, generate bigrams and trigrams, calculate frequencies, predict the next word, and generate a sequence of words.

The second part used the Reuters corpus to construct a trigram language model. Frequency counts were converted into probabilities, allowing the model to predict the most probable next word based on the previous two words.

Finally, perplexity was calculated for a test sentence to evaluate the language model. The experiment provides a practical understanding of N-gram language modeling, word prediction, probability estimation, and language model evaluation.
