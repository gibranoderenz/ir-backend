import ir_datasets
import os

# only run this file if the dataset has not been generated in your directory
def get_collections():
    dataset = ir_datasets.load("beir/climate-fever")
    i = 0
    block_no = 0
    for doc in dataset.docs_iter()[::25]:
        if i == 10_000:
            i = 0
            block_no += 1
        base_path = f'collections/{block_no}'
        if not os.path.exists(base_path):
             os.makedirs(base_path)
        with open(f'{base_path}/{i}.txt', 'w', encoding="utf-8") as file:
            file.write(doc.text)  
        i += 1

def get_training_data():
    dataset = ir_datasets.load("beir/climate-fever")
    docstore = dataset.docs_store()
    doc_ids = set()
    with open('qrels-folder/train_qrels.txt', 'w', encoding='utf-8') as file:
        for qrel in dataset.qrels_iter():
            file.write(f'{qrel.query_id} {qrel.doc_id} {qrel.relevance}\n')
            doc_ids.add(qrel.doc_id)
            
    with open('qrels-folder/train_queries.txt', 'w', encoding='utf-8') as file:
        for query in dataset.queries_iter():
            file.write(f'{query.query_id} {query.text}\n') 

    with open('qrels-folder/train_docs.txt', 'w', encoding='utf-8') as file:
        for doc_id in doc_ids:
            file.write(f'{doc_id} {docstore.get(doc_id).text}\n') 


if __name__ == '__main__':
    # uncomment to run
    # get_collections()
    # get_training_data() -> do not run because it's generated already
    print()
