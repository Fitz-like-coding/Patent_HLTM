
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os   #Python的标准库中的os模块包含普遍的操作系统功能  
import re   #引入正则表达式对象  
import urllib   #用于对URL进行编解码  
from http.server import HTTPServer, BaseHTTPRequestHandler  #导入HTTP处理相关的模块  
import json
import shutil
import posixpath
import cgi
import nltk

nltk.data.path.append('/home/ubuntu/data/nltk_data')

from numpy import e
from qe import QueryExpansion
from tqdm import tqdm
from preprocessor import read_stopwords, preprocessor
from sklearn.feature_extraction.text import CountVectorizer

import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform

import time

# for topic labeling
from sklearn.feature_extraction.text import (CountVectorizer
                                             as WordCountVectorizer)
from topic_labeling.text import LabelCountVectorizer
from topic_labeling.label_finder import BigramLabelFinder
from topic_labeling.label_ranker import LabelRanker
from topic_labeling.pmi import PMICalculator
from topic_labeling.corpus_processor import (CorpusWordLengthFilter,
                                       CorpusPOSTagger,
                                       CorpusStemmer)
from topic_labeling.data import (load_line_corpus, load_lemur_stopwords)

def translate_path(self, path):
    """Translate a /-separated PATH to the local filename syntax.
    Components that mean special things to the local file system
    (e.g. drive or directory names) are ignored.  (XXX They should
    probably be diagnosed.)
    """
    # abandon query parameters
    path = path.split('?',1)[0]
    path = path.split('#',1)[0]
    path = posixpath.normpath(urllib.unquote(path))
    words = path.split('/')
    words = filter(None, words)
    path = os.getcwd()
    for word in words:
        drive, word = os.path.splitdrive(word)
        head, word = os.path.split(word)
        if word in (os.curdir, os.pardir): continue
        path = os.path.join(path, word)
    return path

def _jensen_shannon(_P, _Q):
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def _pcoa(pair_dists, n_components=2):
    """Principal Coordinate Analysis,
    aka Classical Multidimensional Scaling
    """
    # code referenced from skbio.stats.ordination.pcoa
    # https://github.com/biocore/scikit-bio/blob/0.5.0/skbio/stats/ordination/_principal_coordinate_analysis.py

    # pairwise distance matrix is assumed symmetric
    pair_dists = np.asarray(pair_dists, np.float64)

    # perform SVD on double centred distance matrix
    n = pair_dists.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = - H.dot(pair_dists ** 2).dot(H) / 2
    eigvals, eigvecs = np.linalg.eig(B)

    # Take first n_components of eigenvalues and eigenvectors
    # sorted in decreasing order
    ix = eigvals.argsort()[::-1][:n_components]
    eigvals = eigvals[ix]
    eigvecs = eigvecs[:, ix]

    # replace any remaining negative eigenvalues and associated eigenvectors with zeroes
    # at least 1 eigenvalue must be zero
    eigvals[np.isclose(eigvals, 0)] = 0
    if np.any(eigvals < 0):
        ix_neg = eigvals < 0
        eigvals[ix_neg] = np.zeros(eigvals[ix_neg].shape)
        eigvecs[:, ix_neg] = np.zeros(eigvecs[:, ix_neg].shape)

    return np.sqrt(eigvals) * eigvecs

def js_PCoA(distributions):
    """Dimension reduction via Jensen-Shannon Divergence & Principal Coordinate Analysis
    (aka Classical Multidimensional Scaling)
    Parameters
    ----------
    distributions : array-like, shape (`n_dists`, `k`)
        Matrix of distributions probabilities.
    Returns
    -------
    pcoa : array, shape (`n_dists`, 2)
    """
    dist_matrix = squareform(pdist(distributions, metric=_jensen_shannon))
    return _pcoa(dist_matrix)

def _topic_coordinates(mds, topic_term_dists, topic_proportion):
    K = topic_term_dists.shape[0]
    mds_res = mds(topic_term_dists)
    assert mds_res.shape == (K, 2)
    mds_df = pd.DataFrame({'x': mds_res[:,0], 'y': mds_res[:,1], 'topics': range(1, K + 1), \
                          'cluster': 1, 'Freq': topic_proportion * 100})
    return mds_df

def replaceURLs(text):
    return re.sub(r"(\?i)(http:\/\/www\.[^\s]*)|(http:\/\/[^\s]*)|(https:\/\/www\.[^\s]*)|(www\.[^\s]*)|([^\s]*\.com)|(https:\/\/[^\s]*)",'',text)

def replacEmails(text):
    return re.sub('\S+@\S+', '', text)

def load_data(path, processor):
    titles = []
    authors = []
    original_texts = []
    processed_texts = []
    with open(path, "r") as file:
        for line in tqdm(file.readlines()):
            sample = json.loads(line)
            titles.append(sample['title'])
            # titles.append(sample['text'][:20])
            authors.append(sample['author'])
            # authors.append("none")
            original_texts.append(sample['text'])
            processed_texts.append(processor.preprocess(sample['text']))
    return titles, authors, original_texts, processed_texts

def loadJson(FILE):
    with open(FILE, 'r', encoding='utf-8') as file:
        metaData = json.load(file)
    return metaData

#自定义处理程序，用于处理HTTP请求  
class TestHTTPHandler(BaseHTTPRequestHandler):
    def setup(self):
        BaseHTTPRequestHandler.setup(self)
        self.request.settimeout(10)

    #处理GET请求  
    def do_GET(self):
        try:
            #获取URL
            print ('URL=http://3.10.241.200:'+str(port)+self.path)
            url = urllib.parse.unquote(self.path)
            if '&checkCache' in url:
                print(os.path.exists("./QDTM/temp"))
                if os.path.exists("./QDTM/temp"):
                    self.send_response(200) #设置响应状态码  
                    self.send_header('Content-type', 'application/json')  #设置响应头  
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                    self.end_headers()
                    self.wfile.write(json.dumps(True).encode())  #输出响应内容  
                else:
                    self.send_response(200) #设置响应状态码  
                    self.send_header('Content-type', 'application/json')  #设置响应头  
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                    self.end_headers()
                    self.wfile.write(json.dumps(False).encode())  #输出响应内容  
            elif '&download' in url:
                filename = "HITLTM_models"
                shutil.make_archive("./download/"+filename, 'zip', "./QDTM/temp")
                self.send_response(200) #设置响应状态码  
                self.send_header('Content-type', 'application/json')  #设置响应头  
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                self.end_headers()
                self.wfile.write(json.dumps(filename).encode())  #输出响应内容  
            elif '&removezip' in url:
                filename = "HITLTM_models"
                if os.path.exists("./download/" + filename + '.zip'):
                    os.remove("./download/" + filename + '.zip')
                self.send_response(200) #设置响应状态码  
                self.send_header('Content-type', 'application/json')  #设置响应头  
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                self.end_headers()
                self.wfile.write(json.dumps(filename).encode())  #输出响应内容  
            elif '&clearCache' in url:
                if os.path.exists("./QDTM/temp"):
                    shutil.rmtree("./QDTM/temp") 
                if os.path.exists("./QDTM/temp.log"):
                    os.remove("./QDTM/temp.log")
                self.send_response(200) #设置响应状态码  
                self.send_header('Content-type', 'application/json')  #设置响应头  
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                self.end_headers()
                self.wfile.write(json.dumps(True).encode())  #输出响应内容  
            elif '&preprocess' in url:
                word = url.split('&')[1].split("=")[1]
                # processed_word = processor.preprocess(word)
                meta = {"state": True, "processed_word": ""}
                self.send_response(200) #设置响应状态码  
                self.send_header('Content-type', 'application/json')  #设置响应头  
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                self.end_headers()
                if word in vobs:
                    meta = {"state": True, "processed_word": word}
                    self.wfile.write(json.dumps(meta).encode())  #输出响应内容 
                else:
                    processed_word = processor.preprocess(word)
                    if processed_word in vobs:
                        meta = {"state": True, "processed_word": processed_word}
                        self.wfile.write(json.dumps(meta).encode())  #输出响应内容 
                    else:
                        meta = {"state": False, "processed_word": ""}
                        self.wfile.write(json.dumps(meta).encode())  #输出响应内容 

            elif '&query_expansiom' in url:
                query = url.split('&')[1].split("=")[1]
                rule = url.split('&')[2].split("=")[1]
                method = url.split('&')[3].split("=")[1]
                processed_query = []
                for w in query.split():
                    if w in vobs:
                        processed_query.append(w)
                    else:
                        processed_w = processor.preprocess(w)
                        if processed_w in vobs:
                            processed_query.append(processed_w)
                        else:
                            assert 1 == 0
                processed_query = " ".join(processed_query)
                top_words, top_scores, top_docs = qe.suggest(processed_query, num_words=50, rule=rule, mode=method)
                # print(top_words)
                # print(top_scores)
                # print(original_texts[:10])
                meta = {"keywords": [], "documents": []}
                for term in zip(list(top_words), list(top_scores)):
                    meta["keywords"].append({'word': term[0], 'score': "{:0.2f}".format(term[1]/top_scores[0])})
                for i, doc in enumerate(np.array(original_texts)[top_docs[:-50 - 1:-1]]):
                    meta["documents"].append({'title': np.array(titles)[top_docs[:-50 - 1:-1]][i], 'abstract': doc})

                self.send_response(200) #设置响应状态码  
                self.send_header('Content-type', 'application/json')  #设置响应头  
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                self.end_headers()
                self.wfile.write(json.dumps(meta).encode())  #输出响应内容  

            elif '&dump' in url:
                model_name = url.split('&')[1].split("=")[1]
                beta = float(url.split('&')[2].split("=")[1])
                model_path = "./QDTM/temp/"+model_name
                vobs_path = "./QDTM/input/vobs.txt"
                vocabularies = []
                with open(vobs_path, 'r') as file:
                    for line in file.readlines():
                        vocabularies.append(line.strip())
                # type_tracker = {}
                print(len(original_texts))
                doc_word = np.zeros((len(original_texts), len(vocabularies))) + 0.01
                with open(model_path + '/ndz.txt', 'r') as file:
                    for line in file.readlines():
                        temp = line.strip().split()
                        if temp[0] == 'd':
                            continue
                        doc_word[int(temp[0]), int(temp[1])] += 1
                        # type_tracker[int(temp[-2])] = int(temp[-1])
                topic_word = []
                with open(model_path + '/nzw.txt', 'r') as file:
                    for line in file.readlines():
                        topic_word.append(line.strip().split())

                remove_docs = []
                with open(model_path + '/remove_documents.txt', 'r') as file:
                    for line in file.readlines():
                        docs = line.strip().replace(" ","").split(':')[1].strip("[]").split(',')
                        remove_docs.append(docs)

                suggestions = []
                with open(model_path + '/suggestions.txt', 'r') as file:
                    for line in file.readlines():
                        words = line.strip().replace(" ","").split(':')[1].strip("[]").split(',')
                        suggestions.append(words)
                
                names = []
                with open(model_path + '/names.txt', 'r') as file:
                    for line in file.readlines():
                        name = line.strip()
                        names.append(name)

                topic_word = np.array(topic_word).astype(float)
                print(topic_word[0])
                topic_proportion = topic_word.sum(1)/topic_word.sum()
                print(topic_proportion[0])
                topic_word_distribution =  topic_word/topic_word.sum(1)[:, np.newaxis]
                print(topic_word_distribution[0])
                topic_coordinates = _topic_coordinates(js_PCoA, topic_word_distribution, topic_proportion).to_dict('records')
                print(topic_coordinates[0])

                pzw = topic_word/topic_word.sum(0)
                print(pzw[0])
                results = (doc_word @ pzw.T)
                print(results[0])

                temp = results + 1e-20
                doc_topic_distribution = temp / temp.sum(1).reshape((temp.shape[0], 1))
                print(doc_topic_distribution[0])

                # topIndex = np.argsort(topic_word_distribution, axis=1)[:, :-10-1:-1]
                # print(topIndex.shape)
                # mask = np.zeros_like(topic_word_distribution)
                # print(mask.shape)
                # for i, indic in enumerate(topIndex):
                #     mask[i][indic] = 1.0
                # print(mask.sum(1))
                labels = ranker.top_k_labels(topic_models=topic_word_distribution,
                               pmi_w2l=pmi_w2l,
                               index2label=pmi_cal.index2label_,
                               label_models=None,
                               k=n_labels)
                print(labels[0][:5])

                topics = []
                for topic_idx in range(topic_proportion.shape[0]):
                    topic_words_distribution = topic_word_distribution[topic_idx]
                    top_features_ind = topic_words_distribution.argsort()[::-1]
                    top_doc_ind = doc_topic_distribution[:,topic_idx].argsort()[::-1]
                    
                    name = names[topic_idx] if names[topic_idx]!= "None" else "topic " + str(topic_idx+1)
                    summary = ', '.join(map(lambda l: ' '.join(l), labels[topic_idx][:5]))
                    words = [vocabularies[i] for i in top_features_ind if topic_word[topic_idx][i] > beta]
                    words_weight = [str(round(topic_words_distribution[i]*100, 2)) + "%" for i in top_features_ind if topic_word[topic_idx][i] > beta]
                    assert len(words) == len(words_weight)
                    weight = str(round(topic_proportion[topic_idx] * 100)) + "%"
                    docs_title = [titles[i] for i in top_doc_ind if str(i) not in remove_docs[topic_idx]];
                    docs_author = [authors[i] for i in top_doc_ind if str(i) not in remove_docs[topic_idx]];
                    docs_body = [original_texts[i] for i in top_doc_ind if str(i) not in remove_docs[topic_idx]];
                    docs_id = [str(i) for i in top_doc_ind if str(i) not in remove_docs[topic_idx]];
                    docs_weight = [str(round(doc_topic_distribution[i, topic_idx]*100, 2)) + "%" for i in top_doc_ind if str(i) not in remove_docs[topic_idx]];
                    
                    topic = {}
                    topic["name"] = name;
                    topic["summary"] = summary;
                    topic["words"] = words[:30];
                    topic["words_weight"] = words_weight[:30];
                    topic["words_edited"] = [False] * 30;
                    topic["weight"] = weight;
                    topic["docs_title"] = docs_title[:30];
                    topic["docs_author"] = docs_author[:30];
                    topic["docs_body"] = docs_body[:30];
                    topic["docs_weight"] = docs_weight[:30];
                    topic["docs_id"] = docs_id[:30];
                    topic["keep"] = True;
                    topic["new_topic"] = False;
                    topic["add_words"] = [];
                    topic["remove_words"] = [];
                    topic["remove_documents"] = [];
                    topic["split"] = False;
                    topic["merge"] = False;
                    topic["child"] = [];
                    topic["parent"] = [];
                    topic["coordinate"] = {} ;
                    topic["coordinate"]['x'] = topic_coordinates[topic_idx]["x"];
                    topic["coordinate"]['y'] = topic_coordinates[topic_idx]["y"];
                    topic["suggested_words"] = suggestions[topic_idx]
                    topic["remove_docs"] = remove_docs[topic_idx]
                    
                    topics.append(topic)

                with open(model_path+'/model.json', 'w', encoding='utf-8') as outfile:
                    json.dump(topics, outfile, indent=4)

                self.send_response(200) #设置响应状态码  
                self.send_header('Content-type', 'application/json')  #设置响应头  
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                self.end_headers()
                self.wfile.write(json.dumps(topics).encode())  #输出响应内容  
            
            elif '&view' in url:
                model = url.split('&view=')[1]
                metapath = "./QDTM/temp/" + str(model) + "/model.json"
                data= loadJson(metapath)
                #页面输出模板字符串  
                self.send_response(200) #设置响应状态码  
                self.send_header('Content-type', 'application/json')  #设置响应头  
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())  #输出响应内容  
            else:
                print ("do nothing")
        except e:
            print("error for", url, e)

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        print(content_length)

        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        if ctype == 'multipart/form-data':
            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            pdict['CONTENT-LENGTH'] = int(self.headers['Content-Length'])
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST', 'CONTENT_TYPE':self.headers['Content-Type'], })
            try:
                if os.path.isdir('./QDTM/temp'):
                    shutil.rmtree("./QDTM/temp") 
                os.mkdir("./QDTM/temp")
                if isinstance(form["file"], list):
                    for record in form["file"]:
                        filename = record.filename.split('/')
                        print(filename)
                        if filename[1] == ".DS_Store":
                            continue
                        elif filename[1] == "canvas_nodes.json":
                            open("./QDTM/temp/" + filename[1], "wb").write(record.file.read())
                        else:
                            PATH = './QDTM/temp/' + filename[1]
                            if not os.path.isdir(PATH):
                                os.mkdir(PATH)
                            open(PATH + "/" + filename[2], "wb").write(record.file.read())
                # else:
                #     open("./%s"%form["file"].filename, "wb").write(form["file"].file.read())
            except IOError:
                    print("Can't create file to write, do you have permission to write?")
            print ("Files uploaded")
            metapath = "./QDTM/temp/canvas_nodes.json"
            data= loadJson(metapath)
            self.send_response(200) #设置响应状态码  
            self.send_header('Content-type', 'application/json; charset=utf-8')  #设置响应头  
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())  #输出响应内容  
        
        else:
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8')) # <--- Gets the data itself
            print(post_data.keys())
            if post_data["stage"] == "dumpCanvas":
                model_path = "./QDTM/temp"
                with open(model_path+'/canvas_nodes.json', 'w', encoding='utf-8') as outfile:
                    json.dump(post_data, outfile, indent=4)
                self.send_response(200) #设置响应状态码  
                self.send_header('Content-type', 'application/json; charset=utf-8')  #设置响应头  
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                self.end_headers()
                self.wfile.write(json.dumps("success").encode())  #输出响应内容  
            elif post_data["stage"] == "loadCanvas":
                metapath = "./QDTM/temp/canvas_nodes.json"
                data= loadJson(metapath)
                self.send_response(200) #设置响应状态码  
                self.send_header('Content-type', 'application/json; charset=utf-8')  #设置响应头  
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Method", "POST,GET,OPTION")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())  #输出响应内容  

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', "POST,GET,OPTION")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

        #启动服务函数  
def start_server(port):
        http_server = HTTPServer(('', int(port)), TestHTTPHandler)
        http_server.serve_forever() #设置一直监听并接收请求 

#os.chdir('static')  #改变工作目录到 static 目录  
if __name__ == "__main__":
    stopwords = read_stopwords("./dataset/stopwords.en.txt")
    processor = preprocessor(stopwords)
    titles, authors, original_texts, processed_texts = load_data("./dataset/samples.json", processor)

    vectorizer = CountVectorizer(max_df=0.95, min_df=5, max_features=5000, tokenizer=nltk.word_tokenize, stop_words=stopwords)
    vectors = vectorizer.fit_transform(processed_texts)
    vobs = vectorizer.get_feature_names()

    qe = QueryExpansion(emb_path = "./QDTM/glove_embedding/glove.6B.300d.txt")
    qe.initialize(processed_texts, vobs)

    with open('./QDTM/input/input.txt', 'w', encoding='utf-8') as file:
        for line in tqdm(vectors):
            doc = []
            doc.append(str(len(line.indices)))
            count = 0 
            for index, value in zip(line.indices, line.data):
                doc.append(str(index)+":"+str(value))
                count += 1
            assert len(line.indices) == count
            doc = ' '.join(doc)
            file.write(doc + '\n')
    with open('./QDTM/input/vobs.txt', 'w', encoding='utf-8') as file:
        for v in vobs:
            file.write(v + '\n')

    label_min_df = 20
    n_cand_labels = 100
    n_labels = 3
    docs = []
    for l in tqdm(original_texts):
        temp = replaceURLs(l)
        temp = replacEmails(temp)
        sents = nltk.sent_tokenize(temp.strip().lower())
        docs.append([w for sent in map(
            nltk.word_tokenize, sents) for w in sent ])

    preprocessing_steps = ["wordlen", "stem", "tag"]
    if 'wordlen' in preprocessing_steps:
        print("Word length filtering...")
        wl_filter = CorpusWordLengthFilter(minlen=3)
        docs = wl_filter.transform(docs)

    if 'stem' in preprocessing_steps:
        print("Stemming...")
        stemmer = CorpusStemmer()
        docs = stemmer.transform(docs)

    if 'tag' in preprocessing_steps:
        print("POS tagging...")
        tagger = CorpusPOSTagger()
        tagged_docs = tagger.transform(docs)

    label_tags = ['NN,NN']
    tag_constraints = []
    if label_tags != ['None']:
        for tags in label_tags:
            tag_constraints.append(tuple(map(lambda t: t.strip(),
                                             tags.split(','))))

    if len(tag_constraints) == 0:
        tag_constraints = None

    print("Tag constraints: {}".format(tag_constraints))

    print("Generate candidate bigram labels(with POS filtering)...")
    finder = BigramLabelFinder('pmi', min_freq=label_min_df,
                               pos=tag_constraints)
    if tag_constraints:
        assert 'tag' in preprocessing_steps, \
            'If tag constraint is applied, pos tagging(tag) should be performed'
        cand_labels = finder.find(tagged_docs, top_n=n_cand_labels)
    else:  # if no constraint, then use untagged docs
        cand_labels = finder.find(docs, top_n=n_cand_labels)

    print("Collected {} candidate labels".format(len(cand_labels)))
    
    previous = time.time()
    print("Calculate the PMI scores...")
    pmi_cal = PMICalculator(
        doc2word_vectorizer=WordCountVectorizer(tokenizer=nltk.word_tokenize, vocabulary=vobs),
        doc2label_vectorizer=LabelCountVectorizer())
    pmi_w2l = pmi_cal.from_texts(processed_texts, docs, cand_labels)
    ranker = LabelRanker(apply_intra_topic_coverage=False)
    current = time.time()
    print (current - previous)
    vectors = None
    processed_texts = None
    docs = None
    tagged_docs = None
    cand_labels = None
    port = 8060
    print ('Serving HTTP on local port',port, '(http://3.10.241.200:'+str(port)+'/) ...')
    start_server(port)  #启动服务，监听8000端口