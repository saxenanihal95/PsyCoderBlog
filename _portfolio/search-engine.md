---
excerpt: "A Simple Search Engine using nltk"
categories:
  - Machine Learning
tags:
  - machine-learning
  - natural-language-processing
  - web-scraping
header:
  teaser: assets/images/search-engine.jpg
    
---
This is a very basic search engine which we build using the following steps :

* Scraping Documents from any website

* Then the text of the documents will go through 

    * removal of puntuations and stopwords
    * singular words to plural

* Store the final words(keywords) into the database(mysql)

* Create a dictionary of the words(keywords)

* Create a document which will contain lists of list and each list will be a document we scraped and list will contain the frequency(number) of words occured in that document. 

* Create Tf-Idf Model

* Finally search a query in our model.
    

## Imports :


```python
import gensim
import nltk
```

# Scraping Documents

so the first step is scraping the documents and scraped a random website (https://www.technologyreview.com) the method of scraping differs for each site and each so before scraping you need to analyse the tags used in that website and according that the scraping will be done.

### BeautifulSoup used for Scraping the website 

i have used beautifulsoup for scrapping there are some other packages you can use like scrapy,etc.


```python
from bs4 import BeautifulSoup
```

### Request are used to get the html content from the website

i have used simple request of pacakage for getting the html content from the website there are other packages you can use like urllib.


```python
import requests
```

### Used during error handling

i have used sleep that is helpful during the error handling when there is some kind of error in getting the html response.


```python
from time import sleep
```

### Class used for scraping the data

ScrapeData is class in which the whole concept of scraping the documents is achieved.


```python
class ScrapeData :
    docs=[]
    links=[]
    titles=[]
    
    # For scraping data from technologyreview.com website
    def downloadData(self,link) :
        response=requests.get(link)
        soup = BeautifulSoup(response.content, "lxml")
        #print(soup.title.string)
        self.titles.append(soup.title.string)
        article_body=soup.find(attrs={"class": "article-body__content"})
        pTags=article_body.findChildren('p')
        p=''
        for pTag in pTags :
            #print(pTag.get_text())
            p+=pTag.get_text()
            #print('\n') 
        self.docs.append(p)
        self.links.append(link)
        
    def processTechnologyReview(self,baseUrl,categoryUrl) :

        response=requests.get(baseUrl+categoryUrl)

        soup = BeautifulSoup(response.content, "lxml")

        liTags=soup.find('li',attrs={"class": "tech"})

        articleTag=soup.find(attrs={"class": "article"})

        mainTag=soup.find('main')

        if liTags :

            while liTags :
                link =baseUrl+liTags.findChild('a').get('href')
                try:
                    self.downloadData(link)
                except requests.exceptions.MissingSchema:
                    #print("Invalid Url ..")
                    #print("Let me sleep for 5 seconds")
                    #print("ZZzzzz...")
                    sleep(5)
                    #print("Was a nice sleep, now let me continue...")
                    continue
                liTags=liTags.findNextSibling()

        elif articleTag:

            h3Tags=articleTag.findAll('h3')

            for h3Tag in h3Tags :
                if h3Tag.find('a') :
                    link =h3Tag.find('a').get('href')
                    try:
                        self.downloadData(link)
                    except requests.exceptions.MissingSchema:
                        #print("Invalid Url ..")
                        #print("Let me sleep for 5 seconds")
                        #print("ZZzzzz...")
                        sleep(5)
                        #print("Was a nice sleep, now let me continue...")
                        continue
        elif mainTag:

            liClass=mainTag.find('li',attrs={'class':'nav-li nav-li--with-big-dropdown'})
            ulClass=liClass.findChild('ul')
            anchorTags=ulClass.findChild('a')
            while anchorTags :
                link =baseUrl+anchorTags.get('href')
                #print(link)
                try:
                    self.downloadData(link)
                except requests.exceptions.MissingSchema:
                    #print("Invalid Url ..")
                    #print("Let me sleep for 5 seconds")
                    #print("ZZzzzz...")
                    sleep(5)
                    #print("Was a nice sleep, now let me continue...")
                    continue
                anchorTags=anchorTags.findNextSibling()

```

### Instantiate the class to scrape the data

finally we initailized the ScrapeData class and called processTechnologyReview() method which will download the data from the technologyreview website.


```python
scrapeData = ScrapeData()
scrapeData.processTechnologyReview('https://www.technologyreview.com','/lists/technologies/2017/')
scrapeData.processTechnologyReview('https://www.technologyreview.com','/s/609839/our-best-stories-of-2017/')
scrapeData.processTechnologyReview('https://www.technologyreview.com','/lists/innovators-under-35/2017/')
```

## Dummy Documents :

### Considered List of Strings as Documents :

### list to store the docs after removal of puntuations and stopwords


```python
gen_docs=[]
```

### load nltk's English stopwords as variable called 'stopwords'


```python
# nltk.download('stopwords') to download the nltk list of stop words 
stopwords = nltk.corpus.stopwords.words('english')
```

### for removing puntuations( ' , . , etc)


```python
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
```


```python
#nltk.download('punkt') for word tokenizing
import re
from nltk.tokenize import word_tokenize

#for text in raw_documents :
for text in scrapeData.docs :
    word=''
    for w in word_tokenize(text) :
        if re.search('[a-zA-Z]', w.lower()):
            if w.lower() not in stopwords :
                word+=w.lower()
                word+=' '
    gen_docs.append((tokenizer.tokenize(word)))
```

### stemmer is used to convert singular words to plural


```python
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
```

### created a list to store the documents after stemming


```python
stem_docs=[]
```


```python
for text in gen_docs :
    stem_doc=[]
    for t in text :
        if len(t)>1 :
            stem_doc.append(stemmer.stem(t))
    stem_docs.append(stem_doc)
#print(gen_docs)
#print('\n')
#print(stem_docs)
```

## Insert data into mysql

### For mysql connectivity

i have used mysql database to store the keywords,titles and url so that we can retrieve it again.


```python
import mysql.connector
```

### Connectivity with mysql


```python
try:
    cnx = mysql.connector.connect(user='root', password='A1$abcdef',host='localhost',database='search-engine',port='3306')
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
```

### Cursor to manage the data


```python
cursor = cnx.cursor()
```

### Insertion into table


```python
i=0
while i<len(gen_docs) :
    keywords=' '
    for gen_doc in gen_docs[i] :
        keywords+=gen_doc
        keywords+=' '
        query = "INSERT INTO weblink (url, keywords, title) VALUES ('%s',%r,'%s')" % (scrapeData.links[i],keywords, scrapeData.titles[i])
    try :
        cursor.execute(query)
    except mysql.connector.Error as err:
        print(err)
        i+=1
        continue
    i+=1
```

### Commit the changes made


```python
cnx.commit()
```

### Close the connection


```python
cnx.close()
```

### created dictonary of number of words


```python
dictionary = gensim.corpora.Dictionary(stem_docs)
print(dictionary[1])
print(dictionary.token2id['take'])
print("Number of words in dictionary:",len(dictionary))
```

    across
    319
    Number of words in dictionary: 4182


### created a corpus which will contain the mapping of the word to the dictionary of each document.


```python
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in stem_docs]
```

### Now we create a TF-IDF (Term Frequency Inverse Document Frequency) model from the corpus. 

TF is Term Frequency used the calcuate that frequency a word in all documents 

IDF is Inverse Document Frequency that mean word whose frequecy is high in all document will not be considered as it is common to all documents

### num_nnz is the number of tokens.


```python
tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
print(s)
```

    TfidfModel(num_docs=57, num_nnz=12543)
    12543



```python
sims = gensim.similarities.Similarity('/home/nsaxena/Documents',tf_idf[corpus],
                                      num_features=len(dictionary))

print(sims)
print(type(sims))
```

    Similarity index with 57 documents in 0 shards (stored under /home/nsaxena/Documents)
    <class 'gensim.similarities.docsim.Similarity'>


finally the query will be the query of user and we will find the related documents to that.


```python
query="hacking attacks"
word=''
for w in word_tokenize(query) :
        # include only words and in lower case
        if re.search('[a-zA-Z]', w.lower()):
            # for removing common words(the,i,etc)
            if w.lower() not in stopwords :
                word+=w.lower()
                word+=' '
query_doc=(tokenizer.tokenize(word))
query_stem_doc=[]
for t in query_doc :
    if len(t)>1 :
        query_stem_doc.append(stemmer.stem(t))
print(query_stem_doc)
query_doc_bow = dictionary.doc2bow(query_stem_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)
```

    ['hack', 'attack']
    [(1864, 1), (2777, 1)]
    [(1864, 0.5577877243937303), (2777, 0.8299836471374986)]



```python
sims.num_best = 10
similar=sims[query_doc_tf_idf]
```


```python
for sim in similar :
    print( scrapeData.titles[sim[0]])
    print( scrapeData.links[sim[0]])
```

    Hacking Back Makes a Comeback—But It’s Still a Really Bad Idea - MIT Technology Review
    https://www.technologyreview.com/s/609555/hacking-back-makes-a-comeback-but-its-still-a-really-bad-idea/
    Hanqing Wu
    https://www.technologyreview.com/lists/innovators-under-35/2017/pioneer/hanqing-wu/
    Botnets of Things: 10 Breakthrough Technologies 2017 - MIT Technology Review
    https://www.technologyreview.com/s/603500/10-breakthrough-technologies-2017-botnets-of-things/
    Franziska Roesner
    https://www.technologyreview.com/lists/innovators-under-35/2017/inventor/franziska-roesner/
    Eyad Janneh
    https://www.technologyreview.com/lists/innovators-under-35/2017/humanitarian/eyad-janneh/
    Who Will Build the Health-Care Blockchain? - MIT Technology Review
    https://www.technologyreview.com/s/608821/who-will-build-the-health-care-blockchain/
    The Growing Case for Geoengineering - MIT Technology Review
    https://www.technologyreview.com/s/604081/the-growing-case-for-geoengineering/

