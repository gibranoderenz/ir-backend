from flask import Flask, request, jsonify
from bsbi import BSBIIndex
from compression import VBEPostings

app = Flask(__name__)

@app.route('/retrieve/detail', methods=['GET'])
def retrieve_detail():
    doc = request.args.get('doc', '')
    response = {}
    try:
        with open(doc, 'r', encoding='utf-8') as file:
            content = file.read()
            response['content'] = content
        return jsonify(response)
    except FileNotFoundError:
        return jsonify({'message': 'Document not found'})

@app.route('/retrieve', methods=['GET'])
def retrieve():
    BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')
    BSBI_instance.load()
    query = request.args.get('query', '')
    results = BSBI_instance.retrieve_bm25(query, k=100)
    contents = []
    for _, doc in results:
        with open(doc, 'r', encoding='utf-8') as file:
            content = file.read()
            contents.append({'file_name': doc, 'content': content})
    data = {'results': contents}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
