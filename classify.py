#USAGE: python3 classify.py <query_file>
import sys
import string
import numpy as np
from math import sqrt, log
from bs4 import BeautifulSoup


def contain_punctuation(text):
    punctuation = [char for char in string.punctuation + ' ']
    result = any(char in text for char in punctuation)
    return result


def cosine_similarity(v1, v2):
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b))


def log_idfs(dfs, doc_num):
    idfs = {}
    for k,v in dfs.items():
        idfs[k] = log(doc_num / v)
    return idfs


def read_vocab():
    with open('./data/vocab_file.txt','r') as f:
        vocab = f.read().splitlines()
    return vocab


def read_queries(query_file):
    with open(query_file) as f:
        queries = f.read().splitlines()
    return queries


def read_category_vectors():
    vectors = {}
    f = open('./data/category_vectors.txt','r')
    for l in f:
        l = l.rstrip('\n')
        fields = l.split()
        cat = fields[0]
        vec = np.array([float(v) for v in fields[1:]])
        vectors[cat] = vec
    return vectors


def get_ngrams(l,n):
    l = l.lower()
    ngrams = {}
    for i in range(0,len(l)-n+1):
        ngram = l[i:i+n]
        if ngram in ngrams:
            ngrams[ngram]+=1
        else:
            ngrams[ngram]=1
    return ngrams


def normalise_tfs(tfs,total):
    for k,v in tfs.items():
        tfs[k] = v / total
    return tfs


def mk_vector(vocab,tfs):
    vec = np.zeros(len(vocab))
    for t,f in tfs.items():
        if t in vocab:
            pos = vocab.index(t)
            vec[pos] = f
    return vec

vocab = read_vocab()
print(len(vocab))
vectors = read_category_vectors()
queries = read_queries(sys.argv[1])

for q in queries:
    print("\nQUERY:",q)
    ngrams = {}
    cosines = {}
    for i in range(4,7):
        n = get_ngrams(q,i)
        ngrams = {**ngrams, **n}
    qvec = mk_vector(vocab,ngrams)
    for cat,vec in vectors.items():
        cosines[cat] = cosine_similarity(vec,qvec)
    top_category = sorted(cosines, key=cosines.get, reverse=True)[0]
    print('top category: ', top_category)
    with open(f"{top_category}/linear.txt") as f:
        soup = BeautifulSoup(f, 'html.parser')
    texts = []
    for doc in soup.find_all("doc"):
        texts.append(str(doc.contents[0]).strip().lower())
    ns = sys.argv[2:]

    dfs = {}

    vector_list = []
    for text in texts:
        one_doc_tfs = {}
        sum_freq = 0
        for n in ns:
            n = int(n)
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                if not contain_punctuation(ngram):
                    sum_freq += 1
                    if ngram in one_doc_tfs:
                        one_doc_tfs[ngram] += 1
                    else:
                        one_doc_tfs[ngram] = 1
        for ngram in one_doc_tfs.keys():
            if ngram in dfs:
                dfs[ngram] += 1
            else:
                dfs[ngram] = 1
        one_doc_tfs = normalise_tfs(one_doc_tfs, sum_freq)
        one_doc_vector = np.zeros(len(vocab))
        for ngram, tf in one_doc_tfs.items():
            if ngram in vocab:
                position = vocab.index(ngram)
                one_doc_vector[position] = tf
        vector_list.append(one_doc_vector)
        
    all_docs_matrix = np.vstack(vector_list)

    idfs = log_idfs(dfs, len(texts))
    idfs_array = np.array([0 if k not in idfs else idfs[k] for k in vocab])
    for i in range(all_docs_matrix.shape[0]):
        all_docs_matrix[i] = all_docs_matrix[i] * idfs_array
    cosines_docs = {}
    for i in range(all_docs_matrix.shape[0]):
        cosines_docs[i] = cosine_similarity(all_docs_matrix[i], qvec)
    print('top 10 documents: ')
    count = 0
    for doc in sorted(cosines_docs, key=cosines_docs.get, reverse=True):
        print(texts[doc], cosines_docs[doc], '\n\n')
        count += 1
        if count == 10:
            break
