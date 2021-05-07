## Hello this is Sarah Stueve's webpage!
### I plan on adding more content here eventually, but for now, hi LING 539. Here is my technical tutorial for Stanza.

# Classifying Propaganda with Stanza and gensim Doc2Vec
### Installing Dependencies

Before we get started, make sure you have the correct packages installed. I tried to create a Docker container that runs Anaconda Python, but I was unable to completely do this correctly. The image is tagged at sarahstueve/stanza_tutorial.

The required dependencies I am using for this tutorial are as follows:
* [Anaconda](https://www.anaconda.com/products/individual) Python, or at least packages: pickle, pandas, sklearn, numpy
* Stanza which can be installed via directions [here](https://stanfordnlp.github.io/stanza/installation_usage.html).
* [gensim](https://radimrehurek.com/gensim/) which can be installed using ``pip install gensim``
* spaCy which can be installed via directions [here](https://spacy.io/usage)

### Working with Stanza

Once you have the dependencies installed, let's start working with Stanza. Download the English model, including CONLL03 (this might take a minute): 
```python 
stanza.download(lang="en",package=None,processors={"ner":"conll03"})
```
One of the key features of Stanza is that it provides models for 66 different languages, you simply have to download the model that you’re interested in.

Next, create a pipeline. We’re going to select a series of processors – one for tokenization, one for lemmatization, and one for named entity recognition (ner)
```python
nlp = stanza.Pipeline('en', processors = {'ner':'conll03', 'tokenize':'spacy', 'lemma':'default'})
```
To make sure our Pipeline is working correctly, let’s try an example to see what this can do!  
We pass the text we want to be processed to the Pipeline we just created and then we can access and use attributes:

```python
doc = nlp('Barack Obama was born in Hawaii.')
# traverse over sentences in the document to access word/token info
print(doc.text)
print(doc.ents)
for sent in doc.sentences:
    print(sent.ents)
    for word, token in zip(sent.words, sent.tokens):
        print(word.text, word.lemma, token.ner)
```
```
Barack Obama was born in Hawaii.
[{
  "text": "Barack Obama",
  "type": "PER",
  "start_char": 0,
  "end_char": 12
}, {
  "text": "Hawaii",
  "type": "LOC",
  "start_char": 25,
  "end_char": 31
}]
[{
  "text": "Barack Obama",
  "type": "PER",
  "start_char": 0,
  "end_char": 12
}, {
  "text": "Hawaii",
  "type": "LOC",
  "start_char": 25,
  "end_char": 31
}]
Barack Barack B-PER
Obama Obama E-PER
was be O
born bear O
in in O
Hawaii Hawaii S-LOC
. . O
```

Now that we’ve seen what Stanza can do, let’s apply it to a problem of predicting propaganda in news articles.

### Applying Stanza to propaganda

The dataset we’ll be using is the Proppy corpus which can be found [here](https://zenodo.org/record/3271522#.YGk7lK9KhPY).

It contains news articles pulled from various sources and includes the article text, source information, and a propaganda label ("is the article propaganda or not"). 

In this tutorial, we’ll be walking through applying the Stanza document parser to this data for propaganda vs non-propaganda classification.

First, the data can be read into a pandas DataFrame to make it more manageable, however I will be working with an already processed, stored DataFrame in the form of a Pickle file.

Because the dataset is so large and I don’t have the resources to efficiently do this work for the entire article, I have elected to take a sample of the articles and choose 10 sentences from each article in order to predict whether the sentence contains propaganda or not, working at the sentence-level.

In order to read the proppy corpus from a pickle file, I have to open the file in ‘read bytes’/’rb’ mode and then pass it to the pickle load method.

```python
def read_data_prop(fname):
    with open(fname, 'rb') as fp:
        prop = pkl.load(fp)[['article_text', 'source_URL', 'propaganda_label']]
    return prop
```

This function loads the exact pandas DataFrame as it was organized when the data was pickled.

From there, it’s time to apply Stanza to the data.

```python
def process_data(fname, num_docs = 10):
    # now let's load the propaganda data
    df = read_data_prop(fname)
    # take random sample of 100 documents
    data_dict = df.sample(n=num_docs).to_dict('index')
    # generating a 
    sample_dicts = []
    labels = []
    for key in data_dict:
        doc = nlp(data_dict[key]['article_text'])
        # randomly select 10 sentences
        sent_sample = random.choices(doc.sentences, k = 10)
        for sent in sent_sample:
            sample_dicts.append({'text':sent.tokens})
            labels.append(data_dict[key]['propaganda_label'])
    return sample_dicts, labels
```

For this problem, I applied\ the Stanza model to text from each of the sampled articles. In the case of this implementation, when I called the function to process articles with spacy I set the number of training docs (articles) to 200, which worked out to 2000 training samples or "documents" after 10 sentences were selected frrom eeach. The tokenizer being used by Stanza here is the spaCy tokenizer, which is one of the SOA tokenizers currently available. For each sentence segmented by the model, I can access each of the token values individually and also create lists of the tokens together. The function returns a sample dictionary, one for every document (though the dictionaries wound up being unnecessary) and a label list that contains the correct propaganda label from the document which the training sentence was pulled.


Once the data has been preprocessed into a workable format, we can use the [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html) embedding model to generate sentence-level embeddings of the sampled sentences/documents from the articles. The propaganda vs non-propaganda labels can also be encoded using scikit-learn's [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).

```python
def doc2vec(sample_dicts):
    vocab = {}
    id_count = 0
    tagged_docs = []
    for doc in sample_dicts:
        tokens = []
        tags = []
        for token in doc['text']:
            tokens.append(token.text)
            if token.text not in vocab:
                vocab[token.text] = id_count
                tags.append(id_count)
            else:
                tags.append(vocab[token.text])
            id_count += 1
        tagged_docs.append(TaggedDocument(tokens, tags))

    # convert tokens to doc2vec embeddings
    embedding = Doc2Vec(tagged_docs, min_count=1, window=2)
    features = []
    for doc in tagged_docs:
        # print(embedding.infer_vector(doc.words))
        features.append(embedding.infer_vector(doc.words))
        # features.append({'tokens': embedding.wv, 'source':doc['source']})
    return features

def encode_labels(labels):
    encoder = LabelEncoder()
    labels_trans = encoder.fit_transform(labels)
    return labels_trans
```

Once the documents and labels are encoded, we can train a classifier to predict propagandistic vs non-propagandistic.

```python
# train and test classifier
model = LogisticRegression()
model.fit(features, labels)
predictions = model.predict(dev_vecs)
print("f1-score:", f1_score(dev_labels, predictions))
```

While logistic regression, in this implementation, did not perform well on these documents, there are plenty of other classification options to choose from. (It also probably didn't work because of the struggles I had getting this tutorial to work 🙃. However, Stanza is a very powerful natural language processing tool that has many use - particularly for the analysis of different languages, given it's breadth of language parsers. Future work for this project hopes to attempt to conduct entity-level sentiment analysis for propagandistic vs non-propagandistic documents to determine differences in language used in propaganda vs non-propaganda.


