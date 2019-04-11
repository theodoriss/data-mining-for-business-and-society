#!/usr/bin/env python
# coding: utf-8

# In[1]:


from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import *
from whoosh import index
from bs4 import BeautifulSoup as bs
from whoosh import scoring
from whoosh.qparser import *
from whoosh.index import open_dir
import pandas as pd
import numpy
import math
import matplotlib.pyplot as plt
import itertools


# In[2]:


# List of the analyzers
analyzers = [SimpleAnalyzer(),StandardAnalyzer(),StemmingAnalyzer()]


# In[3]:


# Function to create index table for the three analyzers
# param: path to table folder, documents folder path, the schema we use to index documents
# retun: write the index files
def create_table(directory_for_index,directory_with_documents,schema):

    ix = create_in(directory_for_index, schema)

    writer = ix.writer(procs=4,limitmb=999)

    for i in range(1,1401):
        soup =bs(open(directory_with_documents + '______'+str(i) + '.html'), 'html.parser')
        title = soup.find('title').get_text()
        body_ = soup.find('body').get_text()
        titolo = title.replace('\n',' ')
        body = body_.replace('\n',' ')
        body = body.strip()
        writer.add_document(indx=str(i), title=titolo, content=body)
    writer.commit()


# In[4]:


# Call the above function to create the indices
for i in range(0,len(analyzers)):
    selected_analyzer = analyzers[i]
    schema = Schema(indx=ID(stored=True),title=TEXT(stored=True),content=TEXT(stored=True,analyzer=selected_analyzer))
    create_table('index/'+str(i+1),'part_1/Cranfield_DATASET/DOCUMENTS/',schema)


# In[198]:


# We have 6 different scoring algorithms, we created a list of them, and with the three different analyzers, we create 18 different combinations
#scorings = ['frequency','tf_idf','bm25f_1','bm25f_2','bm25f_3','bm25f_4']
scorings = ['frequency','tf_idf','bm25f_1','bm25f_2']
config = list(itertools.product([1,2,3], scorings))


# In[206]:


# Search Engine class


class SearchEngine:
    # initiation function
    # para: configuration (analyzer and scoring), other paras are with default values as we dont need to change in this project 
    def __init__(self, config, path = 'index', data='part_1/Cranfield_DATASET/DOCUMENTS/', gt = 'part_1/Cranfield_DATASET/cran_Ground_Truth.tsv', qs = 'part_1/Cranfield_DATASET/cran_Queries.tsv'):
        # path to the index folder
        self.path = path
        # path to documents
        self.data = data
        # analyzer method
        self.analyzer = config[0]
        # scoring algorithm
        self.scoring = config[1]
        # queries file
        self.qs = qs
        # ground truth dataframe
        self.gt = pd.DataFrame.from_csv(gt, sep="\t",index_col=None)
        self.gt = self.gt.rename(columns={'Query_id':'QueryID', 'Relevant_Doc_id':'DocID'})
        
        # queries dataframe and list
        self.qdf = pd.DataFrame.from_csv(self.qs, sep="\t",index_col=None)
        self.queries = list(self.qdf['Query'])
        #self.scorings = ['frequency','tf_idf','bm25f_1','bm25f_2','bm25f_3','bm25f_4']
        
        # results dictionary
        self.results = self.results_dict()
        # MRR value
        self._mrr = self.mrr()
        # if the search engine pass the MRR constraint
        self.valid = (self._mrr >= 0.32)
    # function to retrieve result documents for a given query
    # param: input query, number of required result documents
    def scoring_results(self,input_query,number_of_results):
        ix = index.open_dir(self.path+'/'+str(self.analyzer))
        #check the scoring parameter and set the scoring_function accordingily
        if self.scoring is 'frequency':
            scoring_function = scoring.Frequency()
        elif self.scoring is 'tf_idf':
            scoring_function = scoring.TF_IDF()
        elif self.scoring is 'bm25f_1':
            scoring_function = scoring.BM25F(B = 0.35, K1 = 0.7)
        elif self.scoring is 'bm25f_2':
            scoring_function = scoring.BM25F(B = 0.75, K1 = 1.2)
        elif self.scoring is 'bm25f_3':
            scoring_function = scoring.BM25F(B = 0.75, K1 = 2.3)
        elif self.scoring is 'bm25f_4':
            scoring_function = scoring.BM25F(B = 0.9, K1 = 1.1)
        else:
            print('scoring method not found')

        qp = QueryParser("content", ix.schema)
        persed_query = qp.parse(input_query)# parsing the query
        searcher = ix.searcher(weighting=scoring_function)
        # execute the search
        results = searcher.search(persed_query,limit=number_of_results)
        rr = []
        rank = 0
        # loop over search results
        for hit in results:
            rank += 1
            rr.append([hit['indx'], rank])

        # close searcher 
        searcher.close()
        # return list of tuples (docID, rank)
        return (rr)
    
    # function to create dictionary for search results:
    # keys: query ID
    # values: lists of related docs
    def results_dict(self):
        se_dict = dict()
        for q in range(0,len(self.queries)):
            se_dict[q+1] = [r[0] for r in self.scoring_results(self.queries[q], 1401)]
        return(se_dict)
    
    # MRR function
    # it iterates over the queries and call scoring funtion to retrieve the result documents for each query
    # retuns the MRR rounded to 2 numbers
    def mrr(self):
        resultsDF = pd.DataFrame(columns=['QueryID', 'DocID', 'Rank'], index=None, dtype=int)
        gtn = list(self.gt.groupby(['QueryID']).count()['DocID'])
        for q in range(0,len(self.queries)):
            for r in self.scoring_results(self.queries[q], gtn[q]):
                resultsDF = resultsDF.append({'QueryID':q, 'DocID': int(r[0]), 'Rank': int(r[1])}, ignore_index=True)


        hits = pd.merge(self.gt, resultsDF, on=["QueryID", "DocID"], how="left")

        mrr = (1 / hits.groupby('QueryID')['Rank'].min()).mean()

        return (round(mrr,2))
    
    # function to calculate R-pre
    
    def rp(self):
        tps=[]
        for q in list(self.results.keys()):
            # relevant docs based on the ground truth
            relevants = list(self.gt[self.gt['QueryID']== q]['DocID'])
            R = len(relevants)
            if R>0:
                tp=0
                # true positive documents are the intersection between search result and ground truth
                tp = list(set(relevants).intersection(set([int(g) for g in self.results[q]])))
                tps.append(len(tp)/R)
        return(tps)    
    
    # function to calculate nDCG for a give number k
 
    def ndcg2(self, k):

        ndcg_lst=[]

        for i in self.results.keys():
            # relevant docs from search engine results
            rs = self.results[i]
            # true relevant from ground truth
            gts = list(self.gt[self.gt['QueryID']== i]['DocID'])
            pos=[]
            numbers=0
            p=1
            for ii in range(0,len(rs)):
                if (int(rs[ii]) in gts): 
                     pos.append(ii+1)



            for item in (pos):
                if item <=k:
                    #tst+=1
                    #numbers+=1
                    if math.log2(p+1) !=0:
                        ndcg_lst.append(1/math.log2(p+1))
                    else:
                        ndcg_lst.append(1)
                   # print(math.log(p+1,2))
                    p+=1
                    #print(p)
                    #print(precision)
        #print(len(ndcg_list))
        #print(tst)
        return(sum(ndcg_lst)/len(ndcg_lst))


# In[ ]:


se = []
for cfg in config:
    s = SearchEngine(cfg)
    se.append(s)


# In[127]:


for i in range(0, len(se)):
    print('Search Engine:'+str(i+1)+': MRR = '+str(se[i]._mrr)+' : Accepted:'+str(se[i].valid))


# In[135]:


rp_df = pd.DataFrame(columns=['se','mean','min','1Q','median','3Q','max'])
for s in range(0, len(se)):
    rp = numpy.array(se[s].rp())
    tmp = [s+1,numpy.mean(rp),numpy.min(rp),numpy.percentile(rp,25),numpy.percentile(rp,50),numpy.percentile(rp,75),numpy.max(rp)]
    rp_df.loc[s] = tmp


# In[136]:


rp_df


# In[205]:


plt.figure(figsize=(14,8))
jet= plt.get_cmap('jet')
colors = iter(jet(numpy.linspace(0,1,11)))
ranks = [i for i in range(2,11)]

for se_ in range(1,len(se)):
    ndcgs=[]
    for rank in ranks:
        ndcgs.append(se[se_].ndcg2(rank))
    if se[se_].valid: plt.plot(ranks,ndcgs,color=next(colors), label = 'SE_'+str(se_))
    plt.axis([1, 10, 0, 1])
plt.legend()
plt.show()

