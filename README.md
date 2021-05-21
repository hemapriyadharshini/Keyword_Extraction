# Keyword_Extraction
**Description:** Extract keywords from a paragraph/Abstract

**Dataset to download:** https://www.kaggle.com/benhamner/nips-papers/

**Input:**

![image](https://user-images.githubusercontent.com/8421214/119171639-d505f280-ba32-11eb-8bf0-311dc624ee87.png)

**Output:**

![image](https://user-images.githubusercontent.com/8421214/119171673-e0f1b480-ba32-11eb-8646-22cdac0fb940.png)

[Credits](https://thecleverprogrammer.com/2020/12/01/keyword-extraction-with-python/)

**Notes:**

Extract keywords in a sentence - It's a key step in information retrieval system - Identify keyword & speedup search

**_Term/Text Frequency Inverse Document Frequency:_** 
Statistical measure that evaluates how relevant a word is to a document in a collection of documents

**Term Frequency** - provides more importance to the word that is more frequent in the document

**Inverse Document Frequency**- provides more weightage to the word that is rare in the corpus (all the documents).keywords are the words with the highest TF-IDF score

**_Coo matrix:_**
Sparse matrices can be used in arithmetic operations: they support addition, subtraction, multiplication, division, and matrix power.
	Advantages of the COO format
		○ facilitates fast conversion among sparse formats
		○ permits duplicate entries (see example)
		○ very fast conversion to and from CSR/CSC formats
	Disadvantages of the COO format
  - Does not directly support arithmetic operations and slicing
