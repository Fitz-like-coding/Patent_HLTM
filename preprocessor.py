import re 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, TweetTokenizer

def read_stopwords(fn):
    return set([line.strip() for line in open(fn, encoding='utf-8') if len(line.strip()) != 0])

class preprocessor(object):
    def __init__(self, stopwords):
        # self.lemmatizer = WordNetLemmatizer()
        self.lemmatizer = PorterStemmer()
        self.stopwords = stopwords

    def tokenize(self, text):
        tokens = TweetTokenizer().tokenize(text)
        tokenized = ''
        for token in tokens:
            tokenized += str(token + ' ')
        return tokenized.strip().lower()

    # def tokenize(self, text):
    #     return ' '.join(word_tokenize(text))

    def replaceURLs(self, text):
        return re.sub(r"(\?i)(http:\/\/www\.[^\s]*)|(http:\/\/[^\s]*)|(https:\/\/www\.[^\s]*)|(www\.[^\s]*)|([^\s]*\.com)|(https:\/\/[^\s]*)", '',text)

    def replacEmails(self, text):
        return re.sub('\S+@\S+', '', text)

    def remove_stop(self, text):
        return ' '.join([w for w in text.split() if w not in self.stopwords])

    def remove_nonalph(self, text):
        return ' '.join([w for w in text.split() if re.search('\W+', w) == None and len(w) > 1 and re.search('[a-zA-Z]', w) != None])

    def remove_dig(self, text):
        return ' '.join([w for w in text.split() if w.isdigit() != True])

    def stem(self, text):
        # return ' '.join([self.lemmatizer.lemmatize(w) for w in text.split()])
        return ' '.join([self.lemmatizer.stem(w) for w in text.split()])

    def preprocess(self, text):
        s = text.lower()
        s = self.replaceURLs(s)
        s = self.replacEmails(s)
        s = self.tokenize(s)
        s = self.remove_stop(s)
        s = self.remove_nonalph(s)
        s = self.remove_dig(s)
        s = self.stem(s)
        return s

if __name__ == '__main__':
    stopwords = read_stopwords("./dataset/stopwords.en.txt")
    preprocessor = preprocessor(stopwords)

    from sklearn.datasets import fetch_20newsgroups
    corpus = fetch_20newsgroups(shuffle=True, random_state=1,remove=('headers', 'footers', 'quotes'))
    texts = corpus.data
    print (texts[0])

    for s in texts:
        print (preprocessor.preprocess(s))
        break