import os
from bsbi import BSBIIndex
from compression import VBEPostings
from tqdm import tqdm
from math import log
from letor import Letor
from collections import defaultdict


# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP
def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    if len(ranking) == 0:
        return 0.
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan 
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    # Referensi: https://www.geeksforgeeks.org/normalized-discounted-cumulative-gain-multilabel-ranking-metrics-ml/
    if len(ranking) == 0:
        return 0.
    
    score = 0
    for i in range(1, len(ranking) + 1):
        score += ranking[i - 1] / log(i + 1, 2)
    return score


def prec(ranking, k):
    """ menghitung search effectiveness metric score dengan 
        Precision at K

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        k: int
          banyak dokumen yang dipertimbangkan atau diperoleh

        Returns
        -------
        Float
          score Prec@K
    """
    # Referensi: https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_484
    number_of_relevant_documents = 0
    for i in range(k):
        number_of_relevant_documents += ranking[i]

    score = number_of_relevant_documents / k
    return score


def ap(ranking):
    """ menghitung search effectiveness metric score dengan 
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    # Reference: https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_482
    cumulative_precision_at_k = 0
    number_of_relevant_documents = 0
    for i in range(len(ranking)):
        if ranking[i] == 1:
            cumulative_precision_at_k += prec(ranking, i + 1)
            number_of_relevant_documents += 1
    if number_of_relevant_documents == 0:
        return 0
    return cumulative_precision_at_k / number_of_relevant_documents

# >>>>> memuat qrels
def load_qrels(qrel_file="qrels-folder/test_qrels.txt"):
    qrels = defaultdict(lambda: defaultdict(lambda: 0)) 
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


# >>>>> EVALUASI !
def eval_retrieval(qrels, query_file="qrels-folder/test_queries.txt", k=100):
    """ 
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-100 documents
    """
    bsbi = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    bsbi.load()

    letor = Letor()
    letor.initialize_model()

    with open(query_file, encoding='utf-8') as file:
        rbp_scores_tfidf = []
        dcg_scores_tfidf = []
        ap_scores_tfidf = []

        rbp_scores_bm25 = []
        dcg_scores_bm25 = []
        ap_scores_bm25 = []

        rbp_scores_tfidf_letor = []
        dcg_scores_tfidf_letor = []
        ap_scores_tfidf_letor = []

        rbp_scores_bm25_letor = []
        dcg_scores_bm25_letor = []
        ap_scores_bm25_letor = []


        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            """
            Evaluasi TF-IDF
            """
            ranking_tfidf = []
            for (score, doc) in bsbi.retrieve_tfidf(query, k=k):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if (did in qrels[qid]):
                    ranking_tfidf.append(1)
                else:
                    ranking_tfidf.append(0)
            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))

            """
            Evaluasi TF-IDF dengan Letor
            """
            ranking_tfidf_letor = []
            search_results = bsbi.retrieve_tfidf(query, k=k)
            docs = []
            for (_, doc_path) in search_results:
                doc_text = letor.get_document_contents(doc_path)
                doc_id = bsbi.doc_id_map[doc_path]
                doc_representation = (doc_id, doc_text)
                docs.append(doc_representation)
            
            if len(docs) == 0:
                rbp_scores_tfidf_letor.append(0)
                dcg_scores_tfidf_letor.append(0)
                ap_scores_tfidf_letor.append(0)
            else:
                reranking_result = letor.rerank(query, docs)
                for (doc, _) in reranking_result:
                    did = int(os.path.splitext(os.path.basename(bsbi.doc_id_map[doc]))[0])
                    if (did in qrels[qid]):
                        ranking_tfidf_letor.append(1)
                    else:
                        ranking_tfidf_letor.append(0)
                rbp_scores_tfidf_letor.append(rbp(ranking_tfidf_letor))
                dcg_scores_tfidf_letor.append(dcg(ranking_tfidf_letor))
                ap_scores_tfidf_letor.append(ap(ranking_tfidf_letor))

            
            """
            Evaluasi BM25
            """
            ranking_bm25 = []
            k1 = 1.2
            b = 0.75
            for (score, doc) in bsbi.retrieve_bm25(query, k=k, k1=k1, b=b):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if (did in qrels[qid]):
                    ranking_bm25.append(1)
                else:
                    ranking_bm25.append(0)
            rbp_scores_bm25.append(rbp(ranking_bm25))
            dcg_scores_bm25.append(dcg(ranking_bm25))
            ap_scores_bm25.append(ap(ranking_bm25))

            """
            Evaluasi BM25 dengan Letor
            """
            ranking_bm25_letor = []
            k1_letor = 1.2
            b_letor = 0.75
            search_results = bsbi.retrieve_bm25(query, k=k, k1=k1_letor, b=b_letor)
            docs = []
            for (_, doc_path) in search_results:
                doc_text = letor.get_document_contents(doc_path)
                doc_id = bsbi.doc_id_map[doc_path]
                doc_representation = (doc_id, doc_text)
                docs.append(doc_representation)

            if len(docs) == 0:
                ranking_bm25_letor.append(0)
                dcg_scores_bm25_letor.append(0)
                ap_scores_bm25_letor.append(0)
            else:
                reranking_result = letor.rerank(query, docs)
                for (doc, _) in reranking_result:
                    did = int(os.path.splitext(os.path.basename(bsbi.doc_id_map[doc]))[0])
                    if (did in qrels[qid]):
                        ranking_bm25_letor.append(1)
                    else:
                        ranking_bm25_letor.append(0)
                rbp_scores_bm25_letor.append(rbp(ranking_bm25_letor))
                dcg_scores_bm25_letor.append(dcg(ranking_bm25_letor))
                ap_scores_bm25_letor.append(ap(ranking_bm25_letor))

    print("Hasil evaluasi TF-IDF terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
    print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
    print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))
    
    print("Hasil evaluasi TF-IDF dengan Letor terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf_letor) / len(rbp_scores_tfidf_letor))
    print("DCG score =", sum(dcg_scores_tfidf_letor) / len(dcg_scores_tfidf_letor))
    print("AP score  =", sum(ap_scores_tfidf_letor) / len(ap_scores_tfidf_letor))

    print(f"Hasil evaluasi BM25 terhadap 150 queries (k1={k1}, b={b})")
    print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
    print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
    print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))

    print(f"Hasil evaluasi BM25 dengan Letor terhadap 150 queries (k1={k1_letor}, b={b_letor})")
    print("RBP score =", sum(rbp_scores_bm25_letor) / len(rbp_scores_bm25_letor))
    print("DCG score =", sum(dcg_scores_bm25_letor) / len(dcg_scores_bm25_letor))
    print("AP score  =", sum(ap_scores_bm25_letor) / len(ap_scores_bm25_letor))


if __name__ == '__main__':
    qrels = load_qrels()
    eval_retrieval(qrels)
