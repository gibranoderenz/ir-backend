import os
import pickle
import contextlib
import heapq
from math import log

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm

from operator import itemgetter

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download("punkt")

stemmer = PorterStemmer()
stop_words_list = stopwords.words('english')
stop_words_set = set(stop_words_list)

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""
        module_dir = os.path.dirname(__file__)  
        with open(os.path.join(module_dir + '\\' + self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(module_dir + '\\' + self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # initializers
        base_path = f'collections/{block_path}'
        files = os.listdir(os.path.join(os.getcwd(), base_path))

        pairs = []

        for file_name in files:
            file_path = f'{base_path}/{file_name}'

            # document mapping
            document_id = self.doc_id_map[file_path]

            with open(os.path.join(os.getcwd(), file_path), 'rb') as file:
                # preprocessing
                try:
                    document = file.read().decode()
                except UnicodeDecodeError:
                    document = file.read().decode('latin-1')
                tokenized_document = re.findall(r'\w+', document)
                document_without_stopwords = []
                for term in tokenized_document:
                    if term not in stop_words_set:
                        document_without_stopwords.append(
                            stemmer.stem(term.lower()))

                # term mapping
                for term in document_without_stopwords:
                    term_id = self.term_id_map[term]
                    pair = (term_id, document_id)
                    pairs.append(pair)

        return pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = {}
            term_dict[term_id][doc_id] = term_dict[term_id].get(doc_id, 0) + 1

        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            index.append(term_id, sorted_doc_id, [
                         term_dict[term_id][doc_id] for doc_id in sorted_doc_id])

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        scores = {}
        stemmed_query = [stemmer.stem(keyword.lower())
                         for keyword in re.findall('\w+', query) if keyword not in stop_words_set]
        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as reader:
            N = len(reader.doc_length)
            for term in stemmed_query:
                term_id = self.term_id_map[term]
                term_data = ()
                try:
                    term_data = reader.postings_dict[term_id]
                except KeyError:
                    continue
                DF = term_data[1]

                alpha = log(N / DF, 10)
                postings_list, tf_list = reader.get_postings_list(term_id)
                postings_list = self.postings_encoding.decode(postings_list)
                tf_list = self.postings_encoding.decode_tf(tf_list)
                # asumsi: len(postings_list) == len(tf_list)
                for i in range(len(postings_list)):
                    current_doc_id = postings_list[i]
                    current_tf = tf_list[i]
                    wtd = 1 + log(current_tf, 10)
                    score = alpha * wtd
                    scores[current_doc_id] = scores.get(
                        current_doc_id, 0) + score

        top_k_doc_ids = heapq.nlargest(k, scores, key=scores.get)
        result = [(scores[doc_id], self.doc_id_map[doc_id])
                  for doc_id in top_k_doc_ids]
        return result

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        scores = {}
        stemmed_query = [stemmer.stem(keyword.lower())
                         for keyword in re.findall('\w+', query)]
        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as reader:
            N = reader.length_of_doc_length
            average_document_length = reader.average_document_length

            for term in stemmed_query:
                term_id = self.term_id_map[term]
                term_data = ()
                try:
                    term_data = reader.postings_dict[term_id]
                except KeyError:
                    continue
                DF = term_data[1]

                alpha = log(N / DF, 10)
                postings_list, tf_list = reader.get_postings_list(term_id)
                postings_list = self.postings_encoding.decode(postings_list)
                tf_list = self.postings_encoding.decode_tf(tf_list)
                # asumsi: len(postings_list) == len(tf_list)
                for i in range(len(postings_list)):
                    current_doc_id = postings_list[i]
                    current_tf = tf_list[i]

                    document_length = reader.doc_length[current_doc_id]

                    normalization_factor = (
                        1 - b) + b * document_length / average_document_length
                    wtd = ((k1 + 1) * current_tf) / \
                        (k1 * normalization_factor + current_tf)

                    score = alpha * wtd
                    scores[current_doc_id] = scores.get(
                        current_doc_id, 0) + score

        top_k_doc_ids = heapq.nlargest(k, scores, key=scores.get)
        result = [(scores[doc_id], self.doc_id_map[doc_id])
                  for doc_id in top_k_doc_ids]
        return result

    def retrieve_tfidf_wand(self, query, k=10):
        # Mengadaptasi dari sumber berikut, dengan modifikasi sesuai keperluan:
        # https://github.com/Jon-bon-Jono/WAND-Top-k-Retrieval-Algorithm/blob/main/wand_algorithm.py

        """
        Melakukan Ranked Retrieval dengan skema WAND.
        Method akan mengembalikan top-K retrieval results.
        """
        mode_index = 0
        stemmed_query = [stemmer.stem(keyword.lower())
                         for keyword in re.findall('\w+', query) if keyword not in stop_words_set]
        iterators = {}
        result = []

        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as reader:
            threshold = 0
            pivot = 0
            N = reader.length_of_doc_length

            # loading initial candidates
            candidates = []
            for term in stemmed_query:
                term_id = self.term_id_map[term]
                term_data = ()
                try:
                    term_data = reader.postings_dict[term_id]
                except KeyError:
                    continue
                DF = term_data[1]
                postings_list, tf_list = reader.get_postings_list(term_id)
                postings_list = reader.postings_encoding.decode(postings_list)
                tf_list = reader.postings_encoding.decode_tf(tf_list)

                pos_list_iterator = iter(postings_list)
                tf_list_iterator = iter(tf_list)
                pos_list_iterator = [next(pos_list_iterator), pos_list_iterator]
                tf_list_iterator = [next(tf_list_iterator), tf_list_iterator]
                iterators[term_id] = [pos_list_iterator, tf_list_iterator]

                current_doc_id = postings_list[0]
                candidates.append((current_doc_id, term_id))

            # start retrieval
            fully_evaluated = 0
            while candidates:
                # sort the candidates by document id
                candidates = sorted(candidates, key=lambda x:x[0])
                score_limit = 0
                pivot = 0
                pivot_found = False

                # print('candidates:', candidates)

                # find the pivot
                while pivot < len(candidates):
                    temp_score_limit = score_limit + \
                        reader.max_upper_bounds[candidates[pivot][0]][mode_index]

                    if temp_score_limit > threshold:
                        pivot_found = True
                        break
                    score_limit = temp_score_limit
                    pivot += 1

                if not pivot_found:
                    break

                pivot_doc = candidates[pivot][0]
                
                if candidates[0][0] == pivot_doc:
                    # success
                    fully_evaluated += 1
                    accumulated_score = 0
                    t = 0
                    removed_candidates = []
                    while t < len(candidates) and candidates[t][0] == pivot_doc:
                        # calculating tfidf or bm25 score
                        term_id = candidates[t][1]
                        term_data = reader.postings_dict[term_id]

                        DF = term_data[1]
                        alpha = log(N / DF, 10)
                        current_tf = iterators[term_id][1][0] 
                        
                        wtd_tfidf = 1 + log(current_tf, 10) 
                        score = alpha * wtd_tfidf
                        accumulated_score += score

                        # updating the candidates
                        try:
                            iterators[term_id][0][0] = next(iterators[term_id][0][1])
                            iterators[term_id][1][0] = next(iterators[term_id][1][1])
                            next_candidate = iterators[term_id][0][0]
                            # print('next_candidate:', next_candidate)
                            candidates[t] = (next_candidate, candidates[t][1])
                        except StopIteration:    
                            removed_candidates.append(candidates[t])
                        
                        t += 1

                    # removing candidates
                    for r in removed_candidates:
                        candidates.remove(r)

                    # adding new top-k
                    if accumulated_score > threshold:
                        if len(result) + 1 > k:
                            result.remove(min(result, key=lambda x:x[0]))
                        
                        result.append((accumulated_score, pivot_doc))
                        threshold = min(result, key=lambda x:x[0])[0]
                    
                else:
                    removed_candidates = []
                    for t in range(pivot):
                        try:
                            term_id = candidates[t][1]
                            iterators[term_id][0][0] = next(iterators[term_id][0][1])
                            iterators[term_id][1][0] = next(iterators[term_id][1][1])
                            while iterators[term_id][0][0] < pivot_doc:
                                iterators[term_id][0][0] = next(iterators[term_id][0][1])
                                iterators[term_id][1][0] = next(iterators[term_id][1][1])
                            candidates[t] = (iterators[term_id][0][0], term_id)

                        except StopIteration:
                            removed_candidates.append(candidates[t])
                    
                    for r in removed_candidates:
                        candidates.remove(r)

        result = [(score, self.doc_id_map[doc_id]) for score, doc_id in result]
        return sorted(result, key=lambda x:x[0], reverse=True)
    
    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        # Mengadaptasi dari sumber berikut, dengan modifikasi sesuai keperluan:
        # https://github.com/Jon-bon-Jono/WAND-Top-k-Retrieval-Algorithm/blob/main/wand_algorithm.py

        """
        Melakukan Ranked Retrieval dengan skema WAND.
        Method akan mengembalikan top-K retrieval results.
        """
        mode_index = 1
        stemmed_query = [stemmer.stem(keyword.lower())
                         for keyword in re.findall('\w+', query) if keyword not in stop_words_set]
        iterators = {}
        result = []

        with InvertedIndexReader('main_index', self.postings_encoding, directory=self.output_dir) as reader:
            threshold = 0
            pivot = 0
            N = reader.length_of_doc_length
            average_document_length = reader.average_document_length
            # loading initial candidates
            candidates = []
            for term in stemmed_query:
                term_id = self.term_id_map[term]
                term_data = ()
                try:
                    term_data = reader.postings_dict[term_id]
                except KeyError:
                    continue
                DF = term_data[1]
                postings_list, tf_list = reader.get_postings_list(term_id)
                postings_list = reader.postings_encoding.decode(postings_list)
                tf_list = reader.postings_encoding.decode_tf(tf_list)

                pos_list_iterator = iter(postings_list)
                tf_list_iterator = iter(tf_list)
                pos_list_iterator = [next(pos_list_iterator), pos_list_iterator]
                tf_list_iterator = [next(tf_list_iterator), tf_list_iterator]
                iterators[term_id] = [pos_list_iterator, tf_list_iterator]

                current_doc_id = postings_list[0]
                candidates.append((current_doc_id, term_id))

            # start retrieval
            fully_evaluated = 0
            while candidates:
                # sort the candidates by document id
                candidates = sorted(candidates, key=lambda x:x[0])
                score_limit = 0
                pivot = 0
                pivot_found = False

                # print('candidates:', candidates)

                # find the pivot
                while pivot < len(candidates):
                    temp_score_limit = score_limit + \
                        reader.max_upper_bounds[candidates[pivot][0]][mode_index]

                    if temp_score_limit > threshold:
                        pivot_found = True
                        break
                    score_limit = temp_score_limit
                    pivot += 1

                if not pivot_found:
                    break

                pivot_doc = candidates[pivot][0]
                
                if candidates[0][0] == pivot_doc:
                    # success
                    fully_evaluated += 1
                    accumulated_score = 0
                    t = 0
                    removed_candidates = []
                    while t < len(candidates) and candidates[t][0] == pivot_doc:
                        # calculating tfidf or bm25 score
                        term_id = candidates[t][1]
                        term_data = reader.postings_dict[term_id]

                        DF = term_data[1]
                        alpha = log(N / DF, 10)
                        current_tf = iterators[term_id][1][0] 

                        current_doc_id = candidates[t][0]
                        document_length = reader.doc_length[current_doc_id]
                        normalization_factor = (
                            1 - b) + b * document_length / average_document_length
                        
                        wtd_bm25 = ((k1 + 1) * current_tf) / \
                            (k1 * normalization_factor + current_tf)
                        score = alpha * wtd_bm25
                        accumulated_score += score

                        # updating the candidates
                        try:
                            iterators[term_id][0][0] = next(iterators[term_id][0][1])
                            iterators[term_id][1][0] = next(iterators[term_id][1][1])
                            next_candidate = iterators[term_id][0][0]
                            # print('next_candidate:', next_candidate)
                            candidates[t] = (next_candidate, candidates[t][1])
                        except StopIteration:    
                            removed_candidates.append(candidates[t])
                        
                        t += 1

                    # removing candidates
                    for r in removed_candidates:
                        candidates.remove(r)

                    # adding new top-k
                    if accumulated_score > threshold:
                        if len(result) + 1 > k:
                            result.remove(min(result, key=lambda x:x[0]))
                        
                        result.append((accumulated_score, pivot_doc))
                        threshold = min(result, key=lambda x:x[0])[0]
                    
                else:
                    removed_candidates = []
                    for t in range(pivot):
                        try:
                            term_id = candidates[t][1]
                            iterators[term_id][0][0] = next(iterators[term_id][0][1])
                            iterators[term_id][1][0] = next(iterators[term_id][1][1])
                            while iterators[term_id][0][0] < pivot_doc:
                                iterators[term_id][0][0] = next(iterators[term_id][0][1])
                                iterators[term_id][1][0] = next(iterators[term_id][1][1])
                            candidates[t] = (iterators[term_id][0][0], term_id)

                        except StopIteration:
                            removed_candidates.append(candidates[t])
                    
                    for r in removed_candidates:
                        candidates.remove(r)

        result = [(score, self.doc_id_map[doc_id]) for score, doc_id in result]
        return sorted(result, key=lambda x:x[0], reverse=True)

    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)
                merged_index.save_document_length_metadata()
                merged_index.save_term_upper_bounds()


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!
