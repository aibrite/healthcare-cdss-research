# getPDFidsWithBioentrez.py

from Bio import Entrez
import pandas as pd
import random

random.seed(10)

Entrez.email = 'email@gmail.com'

def search(query, max_articles):
    handle = Entrez.esearch(
        db='pubmed',
        sort='relevance', # "relevance" algoritmasının detayları NCBI tarafından tam olarak açıklanmaz; 
        # bu, anahtar kelime eşleşmesi, MeSH terimleri, atıf geçmişi ve yayın sıklığı gibi çeşitli sinyalleri içerebilir.
        retmax=str(max_articles),
        retmode='xml',
        term=query
    )
    results = Entrez.read(handle)
    return results['IdList']

def fetch_details(id_list):
    ids = ','.join(id_list)
    handle = Entrez.efetch(
        db='pubmed',
        retmode='xml',
        id=ids
    )
    results = Entrez.read(handle)
    return results

def get_articles(topic: str, count: int):
    id_list = search(topic, count)

    title_list = []
    abstract_list = []
    journal_list = []
    language_list = []
    pubdate_year_list = []
    pubdate_month_list = []
    pmid_list = []
    pmcid_list = []
    doi_list = []

    chunk_size = 5
    for chunk_i in range(0, len(id_list), chunk_size):
        chunk = id_list[chunk_i:chunk_i + chunk_size]
        papers = fetch_details(chunk)
        
        for paper in papers['PubmedArticle']:
            medline = paper['MedlineCitation']
            article = medline['Article']
            
            title_list.append(article.get('ArticleTitle', 'No Title'))
            
            # Abstract
            try:
                abstract = article['Abstract']['AbstractText'][0]
            except:
                abstract = 'No Abstract'
            abstract_list.append(abstract)
            
            journal_list.append(article['Journal']['Title'])
            language_list.append(article['Language'][0])
            
            pubdate = article['Journal']['JournalIssue']['PubDate']
            year = pubdate.get('Year', 'No Data')
            month = pubdate.get('Month', 'No Data')
            pubdate_year_list.append(year)
            pubdate_month_list.append(month)
            
            pmid = str(medline['PMID'])
            pmid_list.append(pmid)
            
            pmcid = 'NA'
            doi = 'NA'
            try:
                idlist = paper['PubmedData']['ArticleIdList']
                for id_obj in idlist:
                    if id_obj.attributes['IdType'] == 'pmc':
                        pmcid = str(id_obj)
                    elif id_obj.attributes['IdType'] == 'doi':
                        doi = str(id_obj)
            except:
                pass
            
            pmcid_list.append(pmcid)
            doi_list.append(doi)

    df = pd.DataFrame({
        'PMID': pmid_list,
        'PMCID': pmcid_list,
        'DOI': doi_list,
        'Title': title_list,
        'Abstract': abstract_list,
        'Journal': journal_list,
        'Language': language_list,
        'Year': pubdate_year_list,
        'Month': pubdate_month_list
    })

    print("Bioentrez df: ")
    print(df.head())

    return df, doi_list
