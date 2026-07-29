# NLP Experiment 2: Twitter Data Preprocessing and Sentiment Analysis

## Aim

To perform basic Natural Language Processing (NLP) operations on Twitter data using the NLTK library and preprocess tweets by removing unnecessary elements such as URLs, user mentions, stop words, punctuation, and other unwanted characters.

---

## Problem Statement

Social media platforms such as Twitter generate large amounts of unstructured textual data. Raw tweets often contain URLs, user mentions, hashtags, punctuation, stop words, emoticons, and other elements that may not be useful for text analysis.

The objective of this experiment is to explore Twitter data using the **NLTK Twitter Samples dataset**, separate positive and negative tweets, and apply preprocessing techniques to convert raw tweets into cleaner text suitable for further NLP tasks such as sentiment analysis.

---

## Brief Theory

### Natural Language Processing

Natural Language Processing (NLP) is a branch of Artificial Intelligence that enables computers to process, understand, and analyze human language.

Before applying machine learning or NLP models to textual data, raw text generally needs to be cleaned and transformed into a suitable format.

### Twitter Samples Dataset

The experiment uses the `twitter_samples` corpus available in NLTK. The corpus contains Twitter data divided into different files, including:

- `positive_tweets.json`
- `negative_tweets.json`
- `tweets.20150430-223406.json`

The complete dataset loaded in the experiment contains **30,000 tweets**.

For sentiment-based analysis, the experiment uses:

- **5,000 positive tweets**
- **5,000 negative tweets**
- **10,000 total labeled tweets**

This provides a balanced dataset for sentiment analysis.

### Text Preprocessing

Text preprocessing is the process of cleaning and transforming raw text into a format that can be effectively used for NLP tasks.

The major preprocessing operations performed in this experiment are:

1. **Removing Retweets**  
   The `RT` prefix used in retweeted posts is removed using regular expressions.

2. **Removing URLs**  
   Hyperlinks present in tweets are removed because they usually do not contribute directly to sentiment analysis.

3. **Removing User Mentions**  
   Twitter usernames beginning with `@` are removed.

4. **Hashtag Processing**  
   The `#` symbol is removed while retaining the associated hashtag word.

5. **Tokenization**  
   `TweetTokenizer` from NLTK is used to divide tweets into individual tokens.

6. **Lowercase Conversion**  
   Tokens are converted to lowercase to maintain consistency.

7. **Stop Word Removal**  
   Common English words such as `the`, `is`, `a`, `and`, and `on` are removed using NLTK stop words.

8. **Punctuation Removal**  
   Punctuation marks are removed using Python's `string.punctuation`.

---

## Implementation Explanation

### 1. Importing Required Libraries

The experiment uses libraries such as:

- **NLTK** for Natural Language Processing
- **Pandas** for data manipulation
- **Matplotlib** for visualization
- **re** for regular expressions
- **string** for punctuation handling

### 2. Loading Twitter Dataset

The `twitter_samples` corpus from NLTK is loaded.

```python
from nltk.corpus import twitter_samples

tweets = twitter_samples.strings()
```

The complete corpus contains **30,000 tweets**.

### 3. Loading Positive and Negative Tweets

Positive and negative tweets are loaded separately.

```python
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
```

The dataset contains:

- 5,000 positive tweets
- 5,000 negative tweets

These tweets are used to create a balanced sentiment dataset.

### 4. Assigning Sentiment Labels

Positive tweets are assigned a **positive** sentiment label, while negative tweets are assigned a **negative** sentiment label.

The two datasets are then combined and shuffled to create a dataset containing **10,000 labeled tweets**.

### 5. Sentiment Visualization

A bar chart is used to visualize the distribution of positive and negative tweets.

Since both classes contain 5,000 tweets, the dataset has an equal distribution of sentiment classes.

### 6. Cleaning Tweets

Regular expressions are used to remove unwanted elements from tweets.

```python
tweet = re.sub(r'^RT[\s]+', '', tweet)
tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
tweet = re.sub(r'@\w+', '', tweet)
tweet = re.sub(r'#', '', tweet)
```

These operations remove:

- Retweet indicators
- URLs
- User mentions
- Hashtag symbols

### 7. Tokenization

The cleaned tweets are tokenized using NLTK's `TweetTokenizer`.

The tokenizer is configured to:

- Convert text to lowercase
- Remove Twitter handles
- Reduce repeated characters

### 8. Stop Word and Punctuation Removal

After tokenization, English stop words and punctuation symbols are removed.

For example:

**Before preprocessing:**

```text
['feels', 'like', 'a', 'lifetime', '!', 'we', 'finish',
'on', 'the', 'same', 'day', '!', '!', 'party', 'time', 'xxx']
```

**After preprocessing:**

```text
['feels', 'like', 'lifetime', 'finish', 'day', 'party', 'time', 'xxx']
```

### 9. Creating the Processing Function

All preprocessing operations are combined into a reusable function:

```python
process_tweet(tweet)
```

The function performs tweet cleaning, tokenization, stop-word removal, and punctuation removal.

The function is applied to the Twitter dataset, and the processed tokens are stored in a new column called:

```text
Tweets_processed
```

---

## Results

The experiment successfully loaded and processed the NLTK Twitter Samples dataset.

### Dataset Results

| Parameter | Result |
|---|---:|
| Total tweets in corpus | 30,000 |
| Positive tweets | 5,000 |
| Negative tweets | 5,000 |
| Total labeled sentiment tweets | 10,000 |
| Sentiment classes | Positive and Negative |

The preprocessing successfully performed:

- Removal of retweet indicators
- Removal of URLs
- Removal of Twitter user mentions
- Removal of hashtag symbols
- Tokenization of tweets
- Conversion to lowercase
- Stop-word removal
- Punctuation removal

The processed tweets contain cleaner and more meaningful tokens compared with the original raw tweets, making them more suitable for further NLP analysis.

---

## Conclusion

In this experiment, Twitter data was successfully loaded and processed using the **NLTK Twitter Samples corpus**. Positive and negative tweets were separated and combined to create a balanced sentiment dataset.

Various NLP preprocessing techniques, including regular-expression-based cleaning, tokenization, lowercase conversion, stop-word removal, and punctuation removal, were successfully implemented.

The experiment demonstrates the importance of text preprocessing in Natural Language Processing. The cleaned tweets can be used for further NLP tasks such as **feature extraction, sentiment classification, and machine learning-based sentiment analysis**.

---

## References

1. [NLTK Documentation](https://www.nltk.org/)
2. [NLTK Twitter Samples](https://www.nltk.org/howto/twitter.html)
3. [NLTK Tokenization Documentation](https://www.nltk.org/api/nltk.tokenize.html)
4. [Python Regular Expressions](https://docs.python.org/3/library/re.html)
5. [Pandas Documentation](https://pandas.pydata.org/docs/)