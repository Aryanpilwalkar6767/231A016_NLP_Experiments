# NLP Experiment 6 – Part-of-Speech Tagging

## Aim

To implement and compare different Part-of-Speech (POS) tagging techniques using NLTK, including rule/statistical tagging, Unigram tagging, and Bigram tagging, and to evaluate their accuracy.

## Problem Statement

Part-of-Speech tagging is the process of assigning a grammatical category to each word in a sentence, such as noun, verb, adjective, pronoun, or determiner. The objective of this experiment is to tokenize a given sentence, perform POS tagging using NLTK, implement Unigram and Bigram taggers using the Treebank corpus, compare their outputs, and evaluate their tagging accuracy.

## Brief Theory

Natural Language Processing requires understanding the grammatical role of words in a sentence. Part-of-Speech (POS) tagging assigns a POS tag to each token based on its grammatical function and surrounding context.

### Part-of-Speech Tagging

POS tagging identifies whether a word is a noun, verb, adjective, pronoun, determiner, and so on.

For example:

```text
The → Determiner
light → Noun
ancient → Adjective
drown → Verb
```

The experiment uses the Penn Treebank POS tagset provided by NLTK.

### Common POS Tags

Some of the tags used in the experiment are:

| Tag    | Meaning                               |
| ------ | ------------------------------------- |
| `DT`   | Determiner                            |
| `NN`   | Noun, singular                        |
| `NNS`  | Noun, plural                          |
| `JJ`   | Adjective                             |
| `VBG`  | Verb, gerund/present participle       |
| `VB`   | Verb, base form                       |
| `VBP`  | Verb, non-3rd person singular present |
| `VBZ`  | Verb, 3rd person singular present     |
| `PRP`  | Personal pronoun                      |
| `PRP$` | Possessive pronoun                    |
| `CC`   | Coordinating conjunction              |
| `WDT`  | Wh-determiner                         |
| `IN`   | Preposition/subordinating conjunction |

### Rule/Statistical POS Tagging

NLTK provides a pretrained POS tagger through the `pos_tag()` function. It analyzes the tokens and assigns appropriate POS tags based on learned language patterns.

### Unigram Tagging

A Unigram tagger assigns a POS tag to a word based primarily on the word itself. It learns the most frequent POS tag associated with each word from a training corpus.

The experiment uses the NLTK Treebank corpus as the training data.

### Bigram Tagging

A Bigram tagger considers the previous tag when determining the POS tag of the current word. Therefore, it uses contextual information from the preceding word/tag.

A Bigram tagger can perform better when the context helps distinguish between different grammatical roles.

### Treebank Corpus

The Penn Treebank corpus available through NLTK contains sentences that have already been annotated with POS tags. It is used in this experiment to train and evaluate the Unigram and Bigram taggers.

### Accuracy

Accuracy measures the percentage of correctly predicted POS tags.

```text
Accuracy = Correctly Tagged Words / Total Words
```

A higher accuracy indicates better POS-tagging performance on the evaluation dataset.

## Technologies and Libraries Used

* Python
* NLTK (Natural Language Toolkit)
* `nltk.tokenize`
* `nltk.tag`
* `nltk.corpus`
* `word_tokenize`
* `pos_tag`
* `UnigramTagger`
* `BigramTagger`
* `treebank`

## NLTK Resources Used

The following NLTK resources are downloaded:

* `averaged_perceptron_tagger_eng`
* `punkt_tab`
* `treebank`
* `tagsets_json`

## Input

The following sentence is used for POS tagging:

```text
A bottomless, suffocating blackness that swallows the light, hiding ancient things that hold their breath and watch you drown.
```

## Implementation

### 1. Import Required Libraries

The required NLTK modules are imported for tokenization, POS tagging, and accessing the Treebank corpus.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import UnigramTagger, BigramTagger, pos_tag
from nltk.corpus import treebank
from nltk import pos_tag_sents
```

### 2. Download Required NLTK Resources

The required NLTK datasets and tagger resources are downloaded.

```python
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
nltk.download('treebank')
nltk.download('tagsets_json')
```

### 3. Define the Sample Sentence

The sample sentence is stored in a variable.

```python
sample_text = "A bottomless, suffocating blackness that swallows the light, hiding ancient things that hold their breath and watch you drown."
```

### 4. Tokenization

The sentence is divided into individual tokens using `word_tokenize()`.

```python
tokens = word_tokenize(sample_text)
```

The tokenizer separates words as well as punctuation marks.

For example:

```text
A
bottomless
,
suffocating
blackness
...
```

### 5. Rule/Statistical POS Tagging

The NLTK `pos_tag()` function is used to assign POS tags to the tokens.

```python
rule_based_tags = pos_tag(tokens)
print(rule_based_tags)
```

The resulting tags include:

```text
A → DT
bottomless → NN
suffocating → VBG
blackness → NN
swallows → VBZ
ancient → JJ
things → NNS
hold → VBP
their → PRP$
you → PRP
drown → VBP
```

### 6. Display POS Tag Information

The NLTK `upenn_tagset()` function is used to display information about a particular POS tag.

```python
nltk.help.upenn_tagset("JJ")
```

The `JJ` tag represents an adjective or numeral, ordinal.

### 7. Load the Treebank Corpus

The tagged sentences from the Treebank corpus are loaded for training the statistical taggers.

```python
train_sents = treebank.tagged_sents()
```

### 8. Create the Unigram Tagger

A Unigram tagger is trained using the Treebank corpus.

```python
unigram_tagger = UnigramTagger(train_sents)
unigram_tag = unigram_tagger.tag(tokens)
```

The tagger assigns a POS tag based mainly on the learned tag associated with each individual word.

Words that are not sufficiently represented in the training corpus may receive `None`.

### 9. Create the Bigram Tagger

A Bigram tagger is trained using the same Treebank corpus.

```python
bigram_tagger = BigramTagger(train_sents)
bigram_tag = bigram_tagger.tag(tokens)
```

The Bigram tagger considers the context of the previous tag while assigning the current POS tag.

### 10. Display Unigram Tagging

The output of the Unigram tagger is displayed.

```python
unigram_tag
```

The output contains POS tags for words recognized by the trained model. Words not found or not sufficiently supported by the training data may receive `None`.

### 11. Display Bigram Tagging

The output of the Bigram tagger is displayed.

```python
bigram_tag
```

The Bigram tagger uses contextual information to determine the tags. For the sample sentence, several words receive `None` because the required word/tag context was not sufficiently available from the training corpus.

### 12. Compare Tagging Results

The original sentence and the outputs of all three tagging approaches are displayed together.

```python
print("original Sentence:", sample_text)
print("\nRule Based Tagging:", rule_based_tags)
print("\nUnigram Tagging:", unigram_tag)
print("\nBigram Tagging:", bigram_tag)
```

This makes it possible to compare the POS tags produced by the different approaches.

## Accuracy Evaluation

### 1. Create Test Dataset

The last portion of the Treebank tagged sentences is selected as the test dataset.

```python
test_sents = treebank.tagged_sents()[3000:]
```

### 2. Calculate Unigram Accuracy

The accuracy of the Unigram tagger is calculated using the test sentences.

```python
unigram_acc = unigram_tagger.accuracy(test_sents)

print(f"Unigram Accuracy:{unigram_acc*100:.2f}%")
```

### 3. Calculate Bigram Accuracy

The accuracy of the Bigram tagger is calculated using the same test dataset.

```python
bigram_acc = bigram_tagger.accuracy(test_sents)

print(f"Bigram Accuracy:{bigram_acc*100:.2f}%")
```

## Additional POS Tagging Example

The experiment also tests POS tagging on the word `class`.

```python
pos_tag(["class"])
```

The output is:

```text
[('class', 'NN')]
```

Here, `class` is tagged as `NN`, which represents a singular noun.

## Results

The experiment successfully performed POS tagging on the given sentence using NLTK.

### Rule/Statistical Tagging

The `pos_tag()` function successfully assigned POS tags to the tokens in the sample sentence.

Examples include:

```text
A → DT
bottomless → NN
suffocating → VBG
blackness → NN
swallows → VBZ
ancient → JJ
things → NNS
hold → VBP
their → PRP$
you → PRP
drown → VBP
```

### Unigram Tagging

The Unigram tagger achieved an accuracy of:

```text
96.32%
```

### Bigram Tagging

The Bigram tagger achieved an accuracy of:

```text
92.16%
```

For the particular training and testing setup used in this experiment, the Unigram tagger achieved higher accuracy than the Bigram tagger.

## Accuracy Comparison

| POS Tagger     | Accuracy |
| -------------- | -------- |
| Unigram Tagger | 96.32%   |
| Bigram Tagger  | 92.16%   |

## Key Learning Outcomes

* Understand the concept of Part-of-Speech tagging.
* Understand common Penn Treebank POS tags.
* Tokenize sentences using NLTK.
* Perform POS tagging using the NLTK `pos_tag()` function.
* Understand Unigram POS tagging.
* Understand Bigram POS tagging.
* Train POS taggers using the Treebank corpus.
* Compare different POS tagging approaches.
* Evaluate POS taggers using accuracy.
* Understand the importance of contextual information in POS tagging.

## Conclusion

The experiment successfully demonstrated Part-of-Speech tagging using Python and NLTK. The given sentence was tokenized and POS tags were assigned using the NLTK `pos_tag()` function.

Unigram and Bigram taggers were trained using the Treebank corpus and their outputs were compared with the standard POS tagging approach. The Unigram tagger achieved an accuracy of **96.32%**, while the Bigram tagger achieved **92.16%** on the selected test dataset.

The experiment provided a practical understanding of POS tagging, statistical tagging methods, training corpora, contextual tagging, and accuracy evaluation in Natural Language Processing.

```
```
