from bsbi import BSBIIndex
from compression import VBEPostings
import time
from letor import Letor

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')
BSBI_instance.load()
queries = ["ice is melting due to heat"]

# letor = Letor()
# letor.initialize_model()

# Retrieval dengan BM25 dan BM25 WAND
for query in queries:
    print("Query  : ", query)
    print("Results:")
    # print('With Letor')
    # start = time.time()
    # result =  BSBI_instance.retrieve_bm25(query, k=100)
    # docs = []
    # for (_, doc_path) in result:
    #     doc_text = letor.get_document_contents(doc_path)
    #     doc_id = BSBI_instance.doc_id_map[doc_path]
    #     doc_representation = (doc_id, doc_text)
    #     docs.append(doc_representation)
    # reranking_result = letor.rerank(query, docs)
    # for (doc, score) in reranking_result:
    #     print(f"{BSBI_instance.doc_id_map[doc]:30} {score:>.3f}")
    # end = time.time()
    # time1 = end - start
    # print()
    # print('========================================================================')
    print('Without Letor')
    start = time.time()
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=100):
        print(f"{doc:30} {score:>.3f}")
    end = time.time()
    time2 = end - start
    print()
