In this project we worked on the problem of Multi-label and Multi-class text classification with Hyper parameter tuning. 
Methods such as Differential Evolution and Randomized Search are used to tune hyper parameters. 
We worked on two datasets:
1. 20NewsGroups dataset which comprises 18000 newsgroup posts on 20 topics. This is a multi-class dataset i.e. 
   there are multiple categories and every article belongs to a single category.
2. Reuters-21578 dataset which comprises 10788 news articles and 90 topical categories. This is a multi-label 
   dataset i.e. every article may belong to more than one category.
   
The file 'Multi_class.py' works on the dataset '20NewsGroups'. The code in this file predicts a single class of each document.
The file 'Multi_label.py' works on the dataset 'Reuters-21578'. The code in this file predicts a multiple classes of each document.

To run the project:
    python Multi_class.py,
    python Multi_label.py
