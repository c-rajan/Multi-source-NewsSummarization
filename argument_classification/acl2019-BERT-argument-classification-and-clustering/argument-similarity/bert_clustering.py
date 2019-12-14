from spacy.lang.en import English
from bert_serving.client import BertClient
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


import spacy
import util_bert as util
import os
import numpy as np


DOCS_DIR = "/home/nishitasnani/cs221_summarization/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs/"


def start_bert_client():
    bc = BertClient()
    return bc


def fetch_all_docs(path):
    docs = []
    for doc in os.listdir(path):
        fp = open(path + "/" + doc, "r")
        flag = 0
        doc_text = ''
        for line in fp.readlines():
            # print("-->", line)
           # if line.strip() == "</TEXT>":
            #    flag = 0
             #   break
            #elif flag == 1:
            doc_text += (line.strip() + ' ')
            #elif line.strip() == "<TEXT>":
             #   flag = 1
            #else:
             #   continue
            
        docs.append(doc_text)
    return docs


def fetch_bert_embeddings(topic_docs):
    """
    Fetch BERT embeddings of all sentences within topic_docs
    """
    embeddings = []
    sentences = []
    for doc in topic_docs:
        doc_sents = util.split_into_sentences(nlp, doc)
        sentences.extend(doc_sents)
        doc_embs = bc.encode(doc_sents)
        # print(len(doc_embs[0]))
        embeddings.extend(doc_embs)
    embeddings = np.array(embeddings)
    return embeddings, sentences

    
if __name__ == '__main__':
    bc = start_bert_client()
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated
    topic_codes = ['d30020t']
    for code in topic_codes:
        topic_docs = fetch_all_docs(DOCS_DIR + code)
    print(len(topic_docs))
    # print(topic_docs[0])
    embeddings, sentences = fetch_embeddings(topic_docs)
    print(embeddings.shape)
    km = KMeans(n_clusters=6).fit(embeddings)
    closest, _ = pairwise_distances_argmin_min(
        km.cluster_centers_, embeddings)
    print(closest.shape)
    closest = sorted(closest)
    for idx in closest:
        print(sentences[idx])
