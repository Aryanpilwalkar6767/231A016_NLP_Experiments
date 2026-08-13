# NLP Experiment 7: Chunking and Named Entity Recognition

## Aim

To perform Part-of-Speech (POS) tagging, noun phrase and verb phrase chunking, and Named Entity Recognition (NER) using NLTK and spaCy.

---

## Problem Statement

Natural language text contains different grammatical structures such as nouns, verbs, adjectives, and phrases. It also contains important entities such as people, organizations, locations, dates, and time expressions. Identifying these structures is an important task in Natural Language Processing.

The objective of this experiment is to perform POS tagging, identify noun and verb phrases using chunking techniques, and recognize named entities from textual data using NLTK and spaCy.

---

## Brief Theory

### Part-of-Speech Tagging

Part-of-Speech tagging assigns a grammatical category to each word based on its role in a sentence. Common POS tags include nouns, verbs, adjectives, determiners, adverbs, and modal verbs.

In this experiment, NLTK's `pos_tag()` function is used to assign POS tags to the tokenized text.

### Chunking

Chunking is the process of grouping words into meaningful phrases using their POS tags. In this experiment, NLTK's `RegexpParser` is used with regular expression grammars to identify:

- **Noun Phrases (NP)**
- **Verb Phrases (VP)**

### Named Entity Recognition

Named Entity Recognition identifies important entities present in text and assigns labels to them. Examples include people, organizations, locations, dates, and other entities.

In this experiment, NLTK's `ne_chunk()` and spaCy's `en_core_web_sm` model are used for Named Entity Recognition.

---

## Implementation Explanation

## Part 1: Chunking

### 1. Importing Required Libraries

The required NLTK libraries are imported for tokenization, POS tagging, and chunking.

```python
import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser
```

### 2. Creating the Sentences

Two sample sentences are created:

```python
sentence_1 = "The quick brown fox jumps over the lazy dog"
sentence_2 = "This is the class of natural language processing."
```

The second sentence is used for the chunking operation.

### 3. Tokenization and POS Tagging

The sentence is tokenized using `word_tokenize()` and then POS tagged using `pos_tag()`.

```python
tokens = word_tokenize(sentence_2)
tagged = pos_tag(tokens)
```

This produces individual tokens along with their corresponding grammatical tags.

### 4. Defining Noun Phrase Grammar

A Regular Expression grammar is created to identify noun phrases:

```python
grammer_1 = "NP: {<DT>?<JJ>*<NN>}"
```

The grammar identifies a noun phrase containing an optional determiner, zero or more adjectives, and a noun.

### 5. Creating the Chunk Parser

The grammar is passed to NLTK's `RegexpParser`.

```python
chunk_parser = RegexpParser(grammer_1)
chunked = chunk_parser.parse(tagged)

print(chunked)
```

The resulting structure represents the identified noun phrases.

### 6. Visualizing the Noun Phrase Tree

The `svgling` library is used to visualize the chunked tree.

```python
from nltk.tree import Tree
import svgling

svgling.draw_tree(chunked)
```

This provides a tree-based representation of the identified noun phrase.

### 7. Defining Verb Phrase Grammar

A second grammar is created to identify verb phrases:

```python
grammer_2 = "VP: {<MD>?<VB.*><RB.*><VB.*>*}"
```

This grammar identifies verb phrase patterns using modal verbs, verbs, and adverbs.

### 8. Creating the Verb Phrase Chunk

The grammar is applied using `RegexpParser`.

```python
chunk_parser = RegexpParser(grammer_2)
chunked1 = chunk_parser.parse(tagged)
```

The resulting chunk tree is displayed:

```python
print(chunked1)
```

The tree is also visualized using:

```python
import svgling
svgling.draw_tree(chunked1)
```

### 9. POS Tag Information

The experiment uses NLTK's POS tag documentation to understand modal verb tags.

```python
nltk.help.upenn_tagset("MD*")
```

---

## Part 2: Named Entity Recognition

### 1. Importing NLTK and Downloading Resources

The required NLTK resources are downloaded before performing POS tagging and Named Entity Recognition.

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("tagsets_json")
```

The required NLTK functions are imported:

```python
from nltk import word_tokenize, pos_tag, ne_chunk
```

### 2. Creating the Input Text

A sample text containing different types of entities is created:

```python
text = "The Natural Language Toolkit, or more commonly NLTK, is a wonderful suite of libraries for Python! Dr. John Doe, an expert in AI, visited M.I.T. last Tuesday at 3:00 PM. He spoke about text-processing, running algorithms, and how computers are learning beautifully from human data. Aren't these technologies changing India rapidly."
```

The text contains names, organizations, locations, dates, time expressions, and other linguistic information.

### 3. Tokenization

The text is tokenized using:

```python
tokens = word_tokenize(text)
print(tokens)
```

This divides the text into individual words and punctuation tokens.

### 4. POS Tagging

The tokens are assigned grammatical tags using:

```python
tagged_tokens = pos_tag(tokens)
print(tagged_tokens)
```

Each token is associated with a POS tag based on its grammatical role.

### 5. POS Tag Information

NLTK's built-in tagset information is used to understand specific POS tags:

```python
nltk.help.upenn_tagset("DT")
nltk.help.upenn_tagset("VBZ")
```

This provides descriptions of the selected Penn Treebank POS tags.

### 6. Named Entity Recognition Using NLTK

Named Entity Recognition is performed using NLTK's `ne_chunk()` function:

```python
named_entities = ne_chunk(tagged_tokens)
print(named_entities)
```

The resulting tree contains the identified named entities.

### 7. Extracting Named Entities

The entities are extracted from the NLTK tree using:

```python
for subtree in named_entities:
    if isinstance(subtree, nltk.Tree):
        entity = "".join([word for word, tag in subtree.leaves()])
        label = subtree.label()
        print(f"Entity:{entity}, Label: {label}")
```

This displays the recognized entity along with its corresponding label.

### 8. Installing svgling

The `svgling` package is installed to visualize the NLTK tree:

```python
!pip install svgling --quiet
```

The named entity tree is then displayed:

```python
named_entities
```

This provides a visual representation of the recognized entities.

### 9. Named Entity Recognition Using spaCy

spaCy is imported and its English language model is loaded:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
```

The input text is processed:

```python
doc = nlp(text)
```

Named entities are then extracted using:

```python
for ent in doc.ents:
    print(f"Entity:{ent.text},label:{ent.label_}")
```

This displays each detected entity and its corresponding spaCy label.

---

## Results

### Part 1: Chunking

The text was successfully tokenized and POS tagged. The defined Regular Expression grammars were successfully used with `RegexpParser` to identify noun phrases and verb phrases.

The chunk trees were also successfully generated and visualized using the `svgling` library.

### Part 2: Named Entity Recognition

The given text was successfully tokenized and POS tagged. Named Entity Recognition was performed using both NLTK and spaCy.

The experiment successfully extracted named entities and displayed their corresponding labels. The NLTK named entity tree was also generated for visualization.

---

## Conclusion

In this experiment, two important NLP tasks were successfully implemented. In **Part 1**, POS tagging and syntactic chunking were performed to identify noun phrases and verb phrases using NLTK's `RegexpParser`. The resulting chunk trees were also visualized.

In **Part 2**, Named Entity Recognition was performed using both NLTK and spaCy. The experiment demonstrated how NLP techniques can be used to identify grammatical structures and meaningful entities from natural language text.

---

## References

1. [NLTK Documentation](https://www.nltk.org/)
2. [NLTK POS Tagging Documentation](https://www.nltk.org/api/nltk.tag.html)
3. [NLTK Chunking Documentation](https://www.nltk.org/api/nltk.chunk.html)
4. [NLTK Named Entity Recognition](https://www.nltk.org/api/nltk.chunk.html)
5. [spaCy Documentation](https://spacy.io/)
6. [spaCy Named Entity Recognition](https://spacy.io/usage/linguistic-features#named-entities)
7. [Penn Treebank POS Tags](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)