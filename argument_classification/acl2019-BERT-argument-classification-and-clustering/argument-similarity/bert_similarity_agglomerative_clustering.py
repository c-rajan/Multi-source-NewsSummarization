from spacy.lang.en import English
from sklearn.cluster import AgglomerativeClustering

import spacy
import os
import numpy as np
import pdb

import bert_clustering
import util_bert as util
import modular_inference
import modular_classification_inference

DOCS_DIR = "/mnt/disk_1/argument_classification/acl2019-BERT-argument-classification-and-clustering/argument-similarity/"
#DOCS_DIR = "/mnt/disk_1/cs221_summarization/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs/"


def agglomerative_clustering(distance_matrix, n_clusters=5,
                             linkage='complete'):
    """
    Agglomerative clustering on the points in the distance matrix
    """
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, affinity='precomputed', linkage=linkage)
    clustering = clustering.fit(distance_matrix)
    return clustering.labels_
    
    
def compute_args_no_args(topic_code, sentences):
    topics = [''] * len(sentences)

    # For sample testing of argument classification
    # with open("sample_article_3.txt", "r") as fp:
    #     sentences = fp.readlines()
    # topics = ['democracy'] * len(sentences)
    args_idx, no_args_idx = modular_classification_inference.get_argument_labels(topics, sentences)

    # Write all argumentative statements in args/ and non-argumentative ones in no_args/
    print(args_idx, len(args_idx))
    with open("./args/" + topic_codes[0] + "_args.txt", "w") as fp:
        for i in args_idx:
            fp.write(sentences[i] + "\n")

    with open("./no_args/" + topic_codes[0] + "_no_args.txt", "w") as fp:
        for i in no_args_idx:
            fp.write(sentences[i] + "\n")

    with open("./all_sents/" + topic_codes[0] + "_all.txt", "w") as fp:
        for i in range(len(sentences)):
            fp.write(sentences[i] + "\n")

    # max_sentences = 20
    # distance_matrix = modular_inference.get_distance_matrix(
    #     arguments=sentences[max_sentences:max_sentences*2])
    # print(distance_matrix.shape)
    # cluster_ids = agglomerative_clustering(distance_matrix)
    # print(cluster_ids)
    # for idx in range(max_sentences):
    #     print(sentences[idx], cluster_ids[idx])
    return sentences, args_idx, no_args_idx

def compute_clusters(sentences, args_idx, n_clusters=5):
    # Get distance matrix for pairwise sentence distances
    sentences = np.array(sentences)
    args_idx = np.array(args_idx)
    sent_args = sentences[args_idx]
    distance_matrix = modular_inference.get_distance_matrix(
        arguments=list(sent_args))
    print(distance_matrix.shape)
    if len(sent_args) <= n_clusters:
        cluster_ids = np.arange(len(sent_args))
    else:
        cluster_ids = agglomerative_clustering(distance_matrix, n_clusters)
    print(cluster_ids)
    # for idx in range(len(sent_args)):
    #     print(sent_args[idx], cluster_ids[idx])
    return distance_matrix, sent_args, cluster_ids


def find_centroids(distance_matrix, cluster_ids, n_clusters=5):
    """
    Given the clusters in the form of cluster_ids, find the 'centroid' within
    each cluster - the point that has the minimum average distance from 
    all the other points in the cluster. 
    """
    selected_idxs = []
    n_clusters = min(n_clusters, max(cluster_ids) + 1)
    for i in range(n_clusters):
        points = np.where(cluster_ids == i)[0]
        min_dist = None
        selected = None
        for point in points:
            point_dist = 0.0
            for sec_point in points:
                point_dist += distance_matrix[point, sec_point]
            if min_dist and point_dist < min_dist:
                selected = point
            if not min_dist:
                selected = point
        selected_idxs.append(selected)
    return selected_idxs


if __name__ == '__main__':
    topic_codes = ['1']
    #topic_codes = ['d30028t', 'd30036t', 'd30049t', 'd30040t', 'd30017t']
    # 'd30029t', 'd30007t', 'd30046t', 'd30034t', 'd30001t', 'd30050t', 'd30048t', 'd30008t', 'd30005t', 'd30022t', 'd30059t', 'd30051t', 'd30055t', 'd30038t', 'd30002t', 'd31008t', 'd31033t', 'd30031t', 'd30011t', 'd30042t', 'd31013t', 'd30027t', 'd30020t', 'd30037t', 'd30003t', 'd30044t', 'd31050t', 'd30053t', 'd31009t', 'd31001t', 'd30010t', 'd31032t', 'd30033t', 'd30056t', 'd30006t', 'd31022t', 'd31038t', 'd31031t', 'd30047t', 'd31043t', 'd31026t', 'd30015t', 'd30026t', 'd30024t', 'd30045t']
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    
    for code in topic_codes:
        topic_docs = bert_clustering.fetch_all_docs(DOCS_DIR + code)
        print(len(topic_docs))
        sentences = []
        for doc in topic_docs:
            doc_sents = util.split_into_sentences(nlp, doc)
            sentences.extend(doc_sents)
        print(len(sentences))
        
        sents, args_idx, no_args_idx = compute_args_no_args(code, sentences)
        sents = sentences
        args_idx = np.arange(len(sentences))
        distance_matrix, sent_args, cluster_ids = compute_clusters(
            sents, args_idx)
        # pdb.set_trace()
        print("sent_args: ", sent_args)
        selected_idxs = find_centroids(distance_matrix, cluster_ids)
        print(selected_idxs)
        selected_idxs = sorted(selected_idxs)
        summary = []
        for idx in selected_idxs:
            summary.append(str(sent_args[idx]))
        # pdb.set_trace()
        string_summary = ' '.join([sent for sent in summary])
        with open("full_docs_summaries_agglo/full_agglo_summary_" + code + ".txt", "w") as fp:
            fp.write(string_summary)
        
