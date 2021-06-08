import nltk
import sys
import os
import string
import math

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
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files_dictionary = dict()
    
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r") as file:
            files_dictionary[filename] = file.read()
    
    return files_dictionary


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    list_of_words = nltk.word_tokenize(document.lower())
    # Remove punctuation
    list_of_words = [word for word in list_of_words 
        if word not in string.punctuation]
    # Remove English stopwords
    return [word for word in list_of_words 
        if word not in nltk.corpus.stopwords.words("english")]

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    words = set()
    for name in documents:
        words.update(documents[name])
    for word in words:
        f = sum(word in documents[name] for name in documents)
        idf = math.log(len(documents) / f + 1)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = dict()
    for word in query:
        for filename in files:
            tf_idfs[filename] = 0
            if word in files[filename]:
                tf = files[filename].count(word)
                tf_idfs[filename] += tf * idfs[word]
    
    filenames = [filename for filename in files]
    filenames.sort(key=lambda f: tf_idfs[f], reverse=True)
    
    return filenames[0:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    def query_term_density(s):
        words_in_query = 0
        for word in query:
            if word in sentences[s]:
                words_in_query += 1
        if words_in_query == 0:
            return 0
        return - len(sentences[s]) / words_in_query

    ranked_sentences = dict()
    for sentence in sentences:
        ranked_sentences[sentence] = 0
        for word in query:
            if word in sentences[sentence]:
                ranked_sentences[sentence] += idfs[word]
    
    sentences_list = [sentence for sentence in sentences]
    sentences_list.sort(
        key=lambda s: (ranked_sentences[s], query_term_density(s)), reverse=True)
    
    return sentences_list[0:n]


if __name__ == "__main__":
    main()
