# NLP Experiment 9 – Word Sense Disambiguation Using the Lesk Algorithm

## Aim

To implement Word Sense Disambiguation (WSD) using the Lesk algorithm and identify the correct meaning of an ambiguous word based on its surrounding context.

## Problem Statement

Words can have multiple meanings depending on the context in which they are used. The objective of this experiment is to identify the appropriate meaning of an ambiguous word by comparing the words in a sentence with the definitions and examples of its possible meanings.

For example, the word **bank** can refer to a financial institution or the land beside a river.

The Lesk algorithm is used to determine which meaning best matches the given sentence.

## Brief Theory

### Word Sense Disambiguation (WSD)

Word Sense Disambiguation is an NLP technique used to determine the correct meaning of a word when it has multiple possible meanings.

Examples:

- **Bank:** A financial institution or the side of a river.
- **Book:** A written work or the act of reserving something.

### Lesk Algorithm

The Lesk algorithm is a knowledge-based approach to Word Sense Disambiguation. It uses dictionary definitions, also called glosses, and examples associated with possible word meanings.

The algorithm follows these steps:

1. Tokenize the input sentence.
2. Retrieve the possible meanings of the ambiguous word.
3. Obtain the definition and examples for each meaning.
4. Tokenize the definitions and examples.
5. Calculate the overlap between the sentence words and the definition/example words.
6. Select the meaning with the highest overlap.

The meaning with the greatest overlap is selected as the best sense.

## Technologies and Libraries Used

- Python
- Jupyter Notebook
- NLTK (Natural Language Toolkit)
- WordNet

### Libraries Used

```python
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
````

## Dataset and Input

This experiment uses WordNet, an English lexical database available through NLTK.

The notebook tests the algorithm using sentences containing ambiguous words such as `bank` and `book`.

### Example 1: Financial Bank

```
I went to the bank to deposit money.
```

Ambiguous word:

```
bank
```

Expected interpretation: A financial institution.

### Example 2: River Bank

```
I went to the river bank to swim.
```

Ambiguous word:

```
bank
```

Expected interpretation: The land beside a river.

### Example 3: Book

```
I am going to read a book.
```

Ambiguous word:

```
book
```

Expected interpretation: A written work that can be read.

## Implementation

### 1. Import Required Libraries

The notebook imports NLTK and WordNet.

```
import nltkfrom nltk.corpus import wordnet as wn
```

WordNet provides the possible senses, definitions, and examples for words.

### 2. Download NLTK Resources

The notebook downloads the required resources:

```
nltk.download('wordnet')nltk.download('punkt')nltk.download('punkt_tab')
```

These resources support WordNet access and sentence tokenization.

### 3. Implement the Lesk Algorithm

A function named `lesk()` is implemented to identify the best meaning of an ambiguous word.

```
def lesk(word, sentence):    best_sense = None    max_overlap = 0    # Tokenize the sentence    sentence_tokens = set(nltk.word_tokenize(sentence))    # Get the possible senses of the word    for sense in wn.synsets(word):        # Get the definition and examples        definition = set(            nltk.word_tokenize(sense.definition())        )        examples = set(            nltk.word_tokenize(' '.join(sense.examples()))        )        # Calculate word overlap        overlap = (            len(definition.intersection(sentence_tokens))            + len(examples.intersection(sentence_tokens))        )        # Select the sense with the greatest overlap        if overlap > max_overlap:            max_overlap = overlap            best_sense = sense    return best_sense
```

### 4. Tokenize the Sentence

The input sentence is tokenized using:

```
nltk.word_tokenize(sentence)
```

The resulting tokens are converted into a set to remove duplicate words.

### 5. Retrieve WordNet Senses

The function retrieves the possible meanings of a word using:

```
wn.synsets(word)
```

Each sense contains a definition and may contain example sentences.

### 6. Calculate Word Overlap

For every possible sense, the algorithm compares the sentence tokens with the tokens in the definition and examples.

The overlap score is calculated by adding:

- The number of common words between the sentence and the definition.
- The number of common words between the sentence and the examples.

### 7. Select the Best Sense

The algorithm keeps track of the highest overlap score.

The sense with the greatest score is returned as the best interpretation of the ambiguous word.

If no overlap is found, the function may return `None`.

### 8. Test the Word Bank

The notebook tests the word `bank` in a financial context:

```
sentence = "I went to the bank to deposit money"ambiguous_word = "bank"sense = lesk(ambiguous_word, sentence)print(    f"The best sense for '{ambiguous_word}' "    f"in the given context is: "    f"{sense.name()}-{sense.definition()}")
```

### 9. Test the River Bank Context

The same ambiguous word is tested in a different sentence:

```
sentence = "I went to the river bank to swim"ambiguous_word = "bank"sense = lesk(ambiguous_word, sentence)
```

This demonstrates how the context can influence the selected meaning.

### 10. Test the Word Book

The notebook also tests the word `book`:

```
sentence = "I am going to read a book"ambiguous_word = "book"sense = lesk(ambiguous_word, sentence)
```

The algorithm compares the sentence with the available WordNet definitions and examples.

## Evaluation

The notebook evaluates the Lesk algorithm using a small manually annotated dataset containing three sentences.

```
dataset = [    ("I went to the bank to deposit some money.", "bank.n.01"),    ("He has a bank of knowledge.", "bank.n.02"),    ("I like to go to river bank.", "bank.n.01")]
```

For each sentence:

1. The ambiguous word `bank` is processed.
2. The Lesk algorithm predicts a sense.
3. The predicted sense is compared with the annotated sense.
4. Correct predictions are counted.
5. Accuracy is calculated.

The accuracy formula is:

Accuracy = (Number of Correct Predictions / Total Predictions) × 100

The notebook prints the result as a percentage.

The final accuracy depends on the senses selected by the implementation. The code uses exact WordNet sense-name matching, so the annotated sense labels must match the actual intended WordNet senses for the evaluation to be reliable.

## Results

The experiment demonstrates:

- Retrieval of possible word senses from WordNet.
- Tokenization of sentence context.
- Comparison of context words with definitions and examples.
- Selection of the sense with the greatest overlap.
- Disambiguation of the word `bank` in different contexts.
- Testing of the word `book`.
- Accuracy calculation using a manually annotated dataset.

## Key Learning Outcomes

After completing this experiment, we understand:

1. The concept of Word Sense Disambiguation.
2. Why ambiguous words require contextual interpretation.
3. How WordNet stores word meanings and definitions.
4. How the Lesk algorithm identifies word senses.
5. How tokenization supports word overlap calculations.
6. How dictionary definitions and examples help identify meaning.
7. How to evaluate predictions using accuracy.
8. The limitations of a simple overlap-based WSD algorithm.

## Applications

Word Sense Disambiguation is useful in:

- Machine translation
- Search engines
- Information retrieval
- Question-answering systems
- Chatbots and virtual assistants
- Text understanding
- Semantic analysis
- Natural Language Understanding

## Limitations

- The algorithm depends on the definitions and examples available in WordNet.
- It may fail when the sentence and the definition use different words with similar meanings.
- The basic implementation does not use advanced semantic similarity.
- The algorithm selects a sense based on word overlap rather than deeper contextual understanding.
- If all overlap scores are zero, the function may return `None`.
- The evaluation dataset contains only three annotated examples, so its accuracy may not represent performance on a larger dataset.

## Conclusion

The experiment successfully implements Word Sense Disambiguation using the Lesk algorithm and WordNet. It identifies the possible meanings of ambiguous words and compares their definitions and examples with the surrounding sentence context. The sense with the highest word overlap is selected as the predicted meaning.

The experiment also demonstrates the evaluation of predicted senses using a manually annotated dataset. Overall, it provides a basic understanding of how contextual information can be used to resolve lexical ambiguity in Natural Language Processing.