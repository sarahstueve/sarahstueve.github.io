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

Once you have the dependencies installed, let's start working with Stanza. Download the English model, including CONLL03 (this might take a minute): 
`python 
stanza.download(lang="en",package=None,processors={"ner":"conll03"})
`
One of the key features of Stanza is that it provides models for 66 different languages, you simply have to download the model that you’re interested in.

Next, create a pipeline. We’re going to select a series of processors – one for tokenization, one for lemmatization, and one for named entity recognition (ner)
```python
nlp = stanza.Pipeline('en', processors = {'ner':'conll03', 'tokenize':'spacy', 'lemma':'spacy'})
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
For this problem, I am applying the Stanza model to text from each of the sampled articles. In the case of this implementation, when I called the function I set the number of training docs to 200, which worked out to 2000 training samples after 10 sentences were selected. The tokenizer being used by Stanza here is the spaCy tokenizer, which is one of the SOA tokenizers currently available. For each sentence segmented by the model, I can access each of the token values individually and also create lists of the tokens together. The function returns a sample dictionary, one for every document (though the dictionaries wound up being unnecessary) and a label list that contains the correct propaganda label from the document which the training sentence was pulled.






