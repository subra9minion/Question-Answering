import nltk
import sys
from os import listdir
from os.path import abspath, join
from string import punctuation
from numpy import log

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, returns a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}

    # path to the corpus
    directory_path = abspath(directory)

    # for each file in the corpus
    for file in listdir(directory_path):

        # path to the text file
        file_path = join(directory_path, file)

        # open and read the file
        with open(file_path, 'r', encoding="utf8") as f:
            file_content = f.read()
            files[file] = file_content
    
    return files


def tokenize(document):
    """
    Given a document (represented as a string), returns a list of all of the
    words in that document, in order.

    The document is processed by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)
    filtered_words = []

    for word in words:

        # if punctuation or stopword
        word = word.lower()
        if word in punctuation or word in nltk.corpus.stopwords.words("english"):
            continue

        # append to filtered words
        filtered_words.append(word)

    return filtered_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, returns a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # dictionary mapping words to their idf values
    idf = {}

    # number of documents
    document_count = len(documents.keys())

    # dictionary mapping words to number of documents containing them
    documents_containing_words = {}

    # computing documents_containing_words
    for document in list(documents.keys()):
        words = set(documents[document])
        for word in words:
            if word in list(documents_containing_words.keys()):
                documents_containing_words[word] += 1
            else:
                documents_containing_words[word] = 1

    # computing idf
    for word in list(documents_containing_words.keys()):
        idf[word] = log(document_count / documents_containing_words[word])

    return idf 


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), returns a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    
    # dictionary mapping files to sum of tf-idf values
    file_tf_idf = {}

    # computing sum of tf-idf for each file
    for file in list(files.keys()):

        # dictionary mapping words (in query) to their frequency
        tf = {}

        # computing tf
        for word in files[file]:
            if word not in query:
                continue
            if word in list(tf.keys()):
                tf[word] += 1
            else:
                tf[word] = 1

        # computing file_tf_idf
        file_tf_idf[file] = 0
        for word in query:
            if word not in files[file]:
                continue
            file_tf_idf[file] += (tf[word] * idfs[word])

    # sorting dict based on values in reverse
    top_files = dict(sorted(file_tf_idf.items(), key=lambda item: item[1], reverse=True))

    # getting top n files
    top_files = list(top_files.keys())[0: n]

    return top_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), returns a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference is
    given to sentences that have a higher query term density.
    """

    # dictionary mapping sentences to sum of idf values
    sentence_idf = {}

    # computing sentence_idf
    for sentence in list(sentences.keys()):
        sentence_idf[sentence] = 0
        words = set(sentences[sentence])
        for word in words:
            if word not in query:
                continue
            sentence_idf[sentence] += idfs[word]
        
    # dictionary mapping sentences to query term density
    query_term_density = {}

    # computing query_term_density
    for sentence in list(sentences.keys()):
        words = sentences[sentence]
        total_words = len(words)
        common_words = 0
        for word in words:
            if word not in query:
                continue
            common_words += 1
        query_term_density[sentence] = (common_words / total_words)

    # dictionary mapping sentences to (idf, query term density)
    sentence_idf_qtd = {}
    for sentence in list(sentences.keys()):
        sentence_idf_qtd[sentence] = tuple((sentence_idf[sentence], query_term_density[sentence]))

    # sorting based on idf, then qtd
    top_sentences = dict(sorted(sentence_idf_qtd.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True))

    # getting top n sentences
    top_sentences = list(top_sentences.keys())[0: n]

    return top_sentences


if __name__ == "__main__":
    main()
