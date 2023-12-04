import random
import re
import numpy as np
import os

import lightgbm
import nltk
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download("punkt")

class Letor():
    def __init__(self, num_negatives = 1, num_latent_topics = 200):
        self.documents = {}
        self.queries = {}
        self.dataset = []
        self.group_qid_count = []
        self.bow_corpus = []
        self.lsi_model = None
        self.dictionary = Dictionary()
        self.ranker = None
        self.q_docs_rel = {}
        self.num_negatives = num_negatives
        self.num_latent_topics = num_latent_topics
        self.stemmer = PorterStemmer()
        self.stop_words_set = set(stopwords.words('english'))

    def generate_whole_dataset(self):
        # parse documents
        with open("qrels-folder/train_docs.txt", encoding='utf-8') as file:
            for line in file:
                line = line.split(' ')
                doc_id = line[0]
                content = line[1:]
                self.documents[doc_id] = content

        # create bow corpus
        self.bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
        self.lsi_model = LsiModel(self.bow_corpus, num_topics=self.num_latent_topics)

        # parse queries
        self.queries = self.parse_queries("qrels-folder/train_queries.txt")
        self.q_docs_rel = self.parse_qrels(self.queries, "qrels-folder/train_qrels.txt")

        # group_qid_count untuk model LGBMRanker
        dataset, group_qid_count = self.make_dataset(self.queries, self.q_docs_rel)
        self.dataset = dataset
        self.group_qid_count = group_qid_count

    def make_dataset(self, queries, q_docs_rel):
        dataset = []
        group_qid_count = []
        for q_id in q_docs_rel:
            docs_rels = q_docs_rel[q_id]
            group_qid_count.append(len(docs_rels) + self.num_negatives)
            for doc_id, rel in docs_rels:
                dataset.append((queries[q_id], self.documents[doc_id], rel))
            # negative random sampling
            for _ in range(self.num_negatives):
                dataset.append((queries[q_id], random.choice(list(self.documents.values())), 0))

        return dataset, group_qid_count

    def parse_queries(self, file_name):
        queries = {}
        with open(file_name, encoding='utf-8') as file:
            for line in file:
                line = line.split(' ')
                q_id = line[0]
                content = line[1:]
                queries[q_id] = content
        return queries
    
    def parse_qrels(self, queries, file_name):
        q_docs_rel = {}
        with open(file_name, encoding='utf-8') as file:
            for line in file:
                line = line.split(' ')
                q_id = line[0]
                doc_id = line[1]
                rel = int(line[2])
                if (q_id in queries) and (doc_id in self.documents):
                    if q_id not in q_docs_rel:
                        q_docs_rel[q_id] = []
                    q_docs_rel[q_id].append((doc_id, int(rel)))
        return q_docs_rel

    def vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.lsi_model[self.dictionary.doc2bow(text)]]
        if len(rep) == self.num_latent_topics:
            return rep
        return [0.] * self.num_latent_topics
    
    def generate_validation_set(self):
        queries = self.parse_queries("qrels-folder/val_queries.txt")
        qrels = self.parse_qrels(queries, "qrels-folder/val_qrels.txt")

        dataset, group_qid_count = self.make_dataset(queries, qrels)
        
        return dataset, group_qid_count
    
    def get_features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        common_words_count = len(q & d)
        query_length = len(q)
        doc_length = len(d)
        average_word_length_q = sum(len(word) for word in q) / query_length if query_length > 0 else 0
        average_word_length_d = sum(len(word) for word in d) / doc_length if doc_length > 0 else 0
        return v_q + v_d + [jaccard] + [cosine_dist] + [common_words_count] + [average_word_length_q] + [average_word_length_d]
    
    def split_data(self, dataset):
        X = []
        Y = []
        for (query, doc, rel) in dataset:
            X.append(self.get_features(query, doc))
            Y.append(rel)
        return np.array(X), np.array(Y)
    
    def get_document_contents(self, file_name):
        document_contents = []
        with open(os.path.join(os.getcwd(), file_name), 'rb') as file:
            try:
                document = file.read().decode().lower()
            except UnicodeDecodeError:
                document = file.read().decode('latin-1').lower()
            tokenized_document = re.findall(r'\w+', document)
            document_contents = document_contents + self.preprocess_text(" ".join(tokenized_document))
        return " ".join(document_contents)
    
    def preprocess_text(self, text):
        result = []
        tokenized_document = re.findall(r'\w+', text)
        for term in tokenized_document:
            if term not in self.stop_words_set:
                result.append(self.stemmer.stem(term.lower()))
        return result

    def train(self):
        X, Y = self.split_data(self.dataset)
        dataset_val, group_qid_count_val = self.generate_validation_set()
        X_val, Y_val = self.split_data(dataset_val)
        eval_set = [(X_val, Y_val)]
        
        self.ranker = lightgbm.LGBMRanker(
                    objective = "lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 150,
                    importance_type = "gain",
                    metric = "ndcg",
                    num_leaves = 70,
                    learning_rate = 0.04,
                    max_depth = 7,
                    random_state=2023
                    )
        
        self.ranker.fit(X, Y,
                        group=self.group_qid_count,
                        eval_set=eval_set, eval_group=[group_qid_count_val]
                        )

    def get_x_unseen(self, query, docs):
        X_unseen = []
        for _, doc in docs:
            X_unseen.append(self.get_features(query.split(), doc.split()))

        X_unseen = np.array(X_unseen)

        return X_unseen

    def rerank(self, query, docs):
        X_unseen = self.get_x_unseen(query, docs)
        scores = self.ranker.predict(X_unseen)

        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)
        return sorted_did_scores

    def initialize_model(self):
        self.generate_whole_dataset()
        self.train()


if __name__ == '__main__':
    letor = Letor()
    letor.initialize_model()
