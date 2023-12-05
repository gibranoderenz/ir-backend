import ir_datasets
import os
import pickle

# only run this file if the dataset has not been generated in your directory
def get_collections():
    dataset = ir_datasets.load("beir/climate-fever")
    i = 0
    block_no = 0 # format: {file_location: doc_title}
    doc_titles = {}
    for doc in dataset.docs_iter():
        if i == 10_000:
            i = 0
            block_no += 1
        base_path = f'collections/{block_no}'
        if not os.path.exists(base_path):
             os.makedirs(base_path)
        file_location = f'{base_path}/{i}.txt'
        with open(file_location, 'w', encoding="utf-8") as file:
            file.write(doc.text)  
        doc_titles[file_location] = doc.title
        i += 1

    with open('doc_titles.pkl', 'wb') as file:
        pickle.dump(doc_titles, file)

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
    get_collections()
    # get_training_data() -> do not run because it's generated already
    # print()
