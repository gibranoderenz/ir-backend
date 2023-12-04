import pickle
import os
from math import log


class InvertedIndex:
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file; dan juga menyediakan
    mekanisme untuk menulis Inverted Index ke file (storage) saat melakukan indexing.

    Attributes
    ----------
    postings_dict: Dictionary mapping:

            termID -> (start_position_in_index_file,
                       number_of_postings_in_list,
                       length_in_bytes_of_postings_list,
                       length_in_bytes_of_tf_list)

        postings_dict adalah konsep "Dictionary" yang merupakan bagian dari
        Inverted Index. postings_dict ini diasumsikan dapat dimuat semuanya
        di memori.

        Seperti namanya, "Dictionary" diimplementasikan sebagai python's Dictionary
        yang memetakan term ID (integer) ke 4-tuple:
           1. start_position_in_index_file : (dalam satuan bytes) posisi dimana
              postings yang bersesuaian berada di file (storage). Kita bisa
              menggunakan operasi "seek" untuk mencapainya.
           2. number_of_postings_in_list : berapa banyak docID yang ada pada
              postings (Document Frequency)
           3. length_in_bytes_of_postings_list : panjang postings list dalam
              satuan byte.
           4. length_in_bytes_of_tf_list : panjang list of term frequencies dari
              postings list terkait dalam satuan byte

    terms: List[int]
        List of terms IDs, untuk mengingat urutan terms yang dimasukan ke
        dalam Inverted Index.

    """

    def __init__(self, index_name, postings_encoding, directory=''):
        """
        Parameters
        ----------
        index_name (str): Nama yang digunakan untuk menyimpan files yang berisi index
        postings_encoding : Lihat di compression.py, kandidatnya adalah StandardPostings,
                        GapBasedPostings, dsb.
        directory (str): directory dimana file index berada
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []         # Untuk keep track urutan term yang dimasukkan ke index
        # key: doc ID (int), value: document length (number of tokens)
        self.doc_length = {}
        # Ini nantinya akan berguna untuk normalisasi Score terhadap panjang
        # dokumen saat menghitung score dengan TF-IDF atau BM25

        self.average_document_length = 0
        self.length_of_doc_length = 0

        # untuk keperluan WAND Top-K Retrieval
        # format: term_id: (max_upper_bound_tfidf, max_upper_bound_bm25)
        self.max_upper_bounds = {}

    def __enter__(self):
        """
        Memuat semua metadata ketika memasuki context.
        Metadata:
            1. Dictionary ---> postings_dict
            2. iterator untuk List yang berisi urutan term yang masuk ke
                index saat konstruksi. ---> term_iter
            3. doc_length, sebuah python's dictionary yang berisi key = doc id, dan
                value berupa banyaknya token dalam dokumen tersebut (panjang dokumen).
                Berguna untuk normalisasi panjang saat menggunakan TF-IDF atau BM25
                scoring regime; berguna untuk untuk mengetahui nilai N saat hitung IDF,
                dimana N adalah banyaknya dokumen di koleksi

        Metadata disimpan ke file dengan bantuan library "pickle"

        Perlu memahani juga special method __enter__(..) pada Python dan juga
        konsep Context Manager di Python. Silakan pelajari link berikut:

        https://docs.python.org/3/reference/datamodel.html#object.__enter__
        """
        # Membuka index file
        self.index_file = open(self.index_file_path, 'rb+')

        # Kita muat postings dict dan terms iterator dari file metadata
        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms, self.doc_length, self.length_of_doc_length, self.average_document_length, self.max_upper_bounds = pickle.load(
                f)
            self.term_iter = self.terms.__iter__()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Menutup index_file dan menyimpan postings_dict dan terms ketika keluar context"""
        # Menutup index file
        self.index_file.close()

        # Menyimpan metadata (postings dict dan terms) ke file metadata dengan bantuan pickle
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump([self.postings_dict, self.terms, self.doc_length,
                        self.length_of_doc_length, self.average_document_length, self.max_upper_bounds], f)


class InvertedIndexReader(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya scan atau membaca secara
    efisien Inverted Index yang disimpan di sebuah file.
    """

    def __iter__(self):
        return self

    def reset(self):
        """
        Kembalikan file pointer ke awal, dan kembalikan pointer iterator
        term ke awal
        """
        self.index_file.seek(0)
        self.term_iter = self.terms.__iter__()  # reset term iterator

    def __next__(self):
        """
        Class InvertedIndexReader juga bersifat iterable (mempunyai iterator).
        Silakan pelajari:
        https://stackoverflow.com/questions/19151/how-to-build-a-basic-iterator

        Ketika instance dari kelas InvertedIndexReader ini digunakan
        sebagai iterator pada sebuah loop scheme, special method __next__(...)
        bertugas untuk mengembalikan pasangan (term, postings_list, tf_list) berikutnya
        pada inverted index.

        PERHATIAN! method ini harus mengembalikan sebagian kecil data dari
        file index yang besar. Mengapa hanya sebagian kecil? karena agar muat
        diproses di memori. JANGAN MEMUAT SEMUA INDEX DI MEMORI!
        """
        curr_term = next(self.term_iter)
        pos, number_of_postings, len_in_bytes_of_postings, len_in_bytes_of_tf = self.postings_dict[
            curr_term]
        postings_list = self.postings_encoding.decode(
            self.index_file.read(len_in_bytes_of_postings))
        tf_list = self.postings_encoding.decode_tf(
            self.index_file.read(len_in_bytes_of_tf))
        return (curr_term, postings_list, tf_list)

    def get_postings_list(self, term):
        """
        Kembalikan sebuah postings list (list of docIDs) beserta list
        of term frequencies terkait untuk sebuah term (disimpan dalam
        bentuk tuple (postings_list, tf_list)).

        PERHATIAN! method tidak boleh iterasi di keseluruhan index
        dari awal hingga akhir. Method ini harus langsung loncat ke posisi
        byte tertentu pada file (index file) dimana postings list (dan juga
        list of TF) dari term disimpan.
        """
        term_data = self.postings_dict[term]
        start_position = term_data[0]
        self.index_file.seek(start_position)
        postings_list = self.index_file.read(term_data[2])
        tf_list = self.index_file.read(term_data[3])
        return (postings_list, tf_list)


class InvertedIndexWriter(InvertedIndex):
    """
    Class yang mengimplementasikan bagaimana caranya menulis secara
    efisien Inverted Index yang disimpan di sebuah file.
    """

    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')
        return self

    def append(self, term, postings_list, tf_list):
        """
        Menambahkan (append) sebuah term, postings_list, dan juga TF list 
        yang terasosiasi ke posisi akhir index file.

        Method ini melakukan 4 hal:
        1. Encode postings_list menggunakan self.postings_encoding (method encode),
        2. Encode tf_list menggunakan self.postings_encoding (method encode_tf),
        3. Menyimpan metadata dalam bentuk self.terms, self.postings_dict, dan self.doc_length.
           Ingat kembali bahwa self.postings_dict memetakan sebuah termID ke
           sebuah 4-tuple: - start_position_in_index_file
                           - number_of_postings_in_list
                           - length_in_bytes_of_postings_list
                           - length_in_bytes_of_tf_list
        4. Menambahkan (append) bystream dari postings_list yang sudah di-encode dan
           tf_list yang sudah di-encode ke posisi akhir index file di harddisk.

        Jangan lupa update self.terms dan self.doc_length juga ya!

        SEARCH ON YOUR FAVORITE SEARCH ENGINE:
        - Anda mungkin mau membaca tentang Python I/O
          https://docs.python.org/3/tutorial/inputoutput.html
          Di link ini juga bisa kita pelajari bagaimana menambahkan informasi
          ke bagian akhir file.
        - Beberapa method dari object file yang mungkin berguna seperti seek(...)
          dan tell()

        Parameters
        ----------
        term:
            term atau termID yang merupakan unique identifier dari sebuah term
        postings_list: List[Int]
            List of docIDs dimana term muncul
        tf_list: List[Int]
            List of term frequencies
        """
        # encoding list
        encoded_list = self.postings_encoding.encode(postings_list)
        encoded_tf_list = self.postings_encoding.encode_tf(tf_list)

        # saving metadata
        if term in self.terms:
            start_position = self.index_file.seek(self.postings_dict[term][0])
        else:
            start_position = self.index_file.seek(0, 2)
            self.terms.append(term)

        # asumsi: len(postings_list) == len(tf_list)
        for i in range(len(postings_list)):
            current_doc_id = postings_list[i]
            term_count = tf_list[i]
            self.doc_length[current_doc_id] = self.doc_length.get(
                current_doc_id, 0) + term_count

        # appending data to harddisk
        self.index_file.write(encoded_list)
        self.index_file.write(encoded_tf_list)

        self.postings_dict[term] = (
            start_position, len(postings_list), len(encoded_list), len(encoded_tf_list))

    # untuk membuat scoring dengan BM25 lebih efisien
    def save_document_length_metadata(self):
        N = len(self.doc_length)
        self.length_of_doc_length = N
        self.average_document_length = sum(
            self.doc_length.values()) / N

    # untuk kepentingan implementasi WAND Top-K Retrieval
    def save_term_upper_bounds(self, k1=1.2, b=0.75):
        N = self.length_of_doc_length
        average_document_length = self.average_document_length

        for term in self.terms:
            DF = self.postings_dict[term][1]
            # getting postings list for term
            term_data = self.postings_dict[term]
            self.index_file.seek(term_data[0])
            encoded_postings_list = self.index_file.read(term_data[2])
            encoded_tf_list = self.index_file.read(term_data[3])
            decoded_postings_list = self.postings_encoding.decode(
                encoded_postings_list)
            decoded_tf_list = self.postings_encoding.decode(encoded_tf_list)

            alpha = log(N / DF, 10)

            # asumsi: len(decoded_postings_list) == len(decoded_tf_list)
            max_upper_bound_tfidf = 0
            max_upper_bound_bm25 = 0
            for i in range(len(decoded_postings_list)):
                current_doc_id = decoded_postings_list[i]
                current_tf = decoded_tf_list[i]

                document_length = self.doc_length[current_doc_id]
                normalization_factor = (
                    1 - b) + b * document_length / average_document_length

                wtd_tfidf = 1 + log(current_tf, 10)
                wtd_bm25 = ((k1 + 1) * current_tf) / \
                    (k1 * normalization_factor + current_tf)
                score_tfidf = alpha * wtd_tfidf
                score_bm25 = alpha * wtd_bm25
                max_upper_bound_tfidf = max(max_upper_bound_tfidf, score_tfidf)
                max_upper_bound_bm25 = max(max_upper_bound_bm25, score_bm25)

            self.max_upper_bounds[term] = (
                max_upper_bound_tfidf, max_upper_bound_bm25)


if __name__ == "__main__":

    from compression import VBEPostings

    with InvertedIndexWriter('test', postings_encoding=VBEPostings, directory='./tmp/') as index:
        index.append(1, [2, 3, 4, 8, 10], [2, 4, 2, 3, 30])
        index.append(2, [3, 4, 5], [34, 23, 56])
        index.index_file.seek(0)
        assert index.terms == [1, 2], "terms salah"
        assert index.doc_length == {
            2: 2, 3: 38, 4: 25, 5: 56, 8: 3, 10: 30}, "doc_length salah"
        assert index.postings_dict == {1: (0,
                                           5,
                                           len(VBEPostings.encode(
                                               [2, 3, 4, 8, 10])),
                                           len(VBEPostings.encode_tf([2, 4, 2, 3, 30]))),
                                       2: (len(VBEPostings.encode([2, 3, 4, 8, 10])) + len(VBEPostings.encode_tf([2, 4, 2, 3, 30])),
                                           3,
                                           len(VBEPostings.encode([3, 4, 5])),
                                           len(VBEPostings.encode_tf([34, 23, 56])))}, "postings dictionary salah"

        index.index_file.seek(index.postings_dict[2][0])
        assert VBEPostings.decode(index.index_file.read(
            len(VBEPostings.encode([3, 4, 5])))) == [3, 4, 5], "terdapat kesalahan"
        assert VBEPostings.decode_tf(index.index_file.read(
            len(VBEPostings.encode_tf([34, 23, 56])))) == [34, 23, 56], "terdapat kesalahan"