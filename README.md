# Question-Answering
An AI that answers questions using **Natural Language Processing**, given a corpus of text documents and a query.
The task is split into ```Document Retrieval``` and ```Passage Retrieval```. 
When presented with a query, document retrieval will first identify which document(s) are most relevant to the query.
Passage retrieval will then determine the most relevant passage(s) to the question from the top document(s).

## Installation
1. Ensure you have Python installed.
1. Download this repository into your system.
1. Run ```pip3 install -r requirements.txt``` in your terminal.

## Usage
Question-Answering requires a corpus of text documents. 
An example of a corpus can be found in the ```corpus``` directory of the project. 
Each text file contains contents of a Wikipedia page.
Once the corpus is ready, run ```python questions.py corpus_name```. 
You will then be prompted for a query.

## Sample
```
$ python questions.py corpus
Query: What are the types of supervised learning?
Types of supervised learning algorithms include Active learning , classification and regression.

$ python questions.py corpus
Query: When was Python 3.0 released?
Python 3.0 was released on 3 December 2008.

$ python questions.py corpus
Query: How do neurons connect in a neural network?
Neurons of one layer connect only to neurons of the immediately preceding and immediately following layers.
```

## License
This project was made under CS50's Introduction to Artificial Intelligence, a course of study by HarvardX.<br>
The course is licensed under a [Creative Commons License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
