import pickle
from flask import Flask, request, jsonify
from bsbi import BSBIIndex
from compression import VBEPostings
from flask_cors import CORS
from letor import Letor

app = Flask(__name__)
CORS(app, resources={
    r'/retrieve/*': {
        "origins": "*",
        "methods": ['GET']
    }
})

bsbi_loaded = False
BSBI_instance = None
letor_loaded = False
letor = None

def load_bsbi_instance():
    global bsbi_loaded, BSBI_instance
    if not bsbi_loaded:
        BSBI_instance = BSBIIndex(data_dir='collections',
                                  postings_encoding=VBEPostings,
                                  output_dir='index')
        BSBI_instance.load()
        bsbi_loaded = True

    return BSBI_instance

def load_titles():
    with open('doc_titles.pkl', 'rb') as file:
        doc_titles = pickle.load(file)
    
    return doc_titles

def load_letor():
    global letor_loaded, letor
    if not letor:
        letor = Letor()
        letor.initialize_model()
    return letor

@app.route('/retrieve/detail', methods=['GET'])
def retrieve_detail():
    doc = request.args.get('doc', '')
    response = {}
    doc_titles = load_titles()
    try:
        with open(doc, 'r', encoding='utf-8') as file:
            content = file.read()
            response['title'] = doc_titles[doc]
            response['content'] = content
        return jsonify(response)
    except FileNotFoundError:
        return jsonify({'message': 'Document not found'})

@app.route('/retrieve', methods=['GET'])
def retrieve():
    global BSBI_instance, letor
    BSBI_instance = load_bsbi_instance()
    doc_titles = load_titles()
    letor = load_letor()

    query = request.args.get('query', '')
    with_letor = request.args.get('with_letor', 'false')
    results = BSBI_instance.retrieve_bm25(query, k=100)
    if with_letor.lower() == 'true':
        docs = []
        for (_, doc_path) in results:
            doc_text = letor.get_document_contents(doc_path)
            doc_id = BSBI_instance.doc_id_map[doc_path]
            doc_representation = (doc_id, doc_text)
            docs.append(doc_representation)
        results = letor.rerank(query, docs)
        results = map(lambda pair: (pair[0], BSBI_instance.doc_id_map[pair[1]]),results)

    contents = []
    for _, doc in results:
        with open(doc, 'r', encoding='utf-8') as file:
            content = file.read()
            contents.append({'file_name': doc, 'title': doc_titles[doc], 'content': content})
    data = {'results': contents}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)