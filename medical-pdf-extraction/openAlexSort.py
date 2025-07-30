# openAlexSort.py (güncellenmiş)

import requests
import pandas as pd
import time
from tqdm import tqdm

def fetch_openalex_metadata(doi_list):
    results = []

    for doi in tqdm(doi_list, desc="OpenAlex'ten veri çekiliyor"):
        if doi == "NA":
            continue

        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Hata: {doi} için veri çekilemedi")
            continue
        data = response.json()

        cited_by_count = data.get("cited_by_count", 0)
        title = data.get("title", "")
        abstract = data.get("abstract_inverted_index", "")
        if abstract:
            abstract = ' '.join([' '.join([word]*count) for word, indexes in abstract.items() for count in [len(indexes)]])
        else:
            abstract = ""

        publication_year = data.get("publication_year", "")
        publication_date = data.get("publication_date", "")
        is_oa = data.get("open_access", {}).get("is_oa", False)
        host_venue = data.get("host_venue", {})
        journal = host_venue.get("display_name", "")
        publisher = host_venue.get("publisher", "")

        concepts = data.get("concepts", [])
        concepts_list = "; ".join([f"{c.get('display_name')} ({c.get('score'):.2f})" for c in concepts])

        authorships = data.get("authorships", [])
        authors_list = "; ".join([
            f"{a.get('author', {}).get('display_name')} ({a.get('institutions')[0].get('display_name') if a.get('institutions') else ''})"
            for a in authorships
        ])

        referenced_count = len(data.get("referenced_works", []))
        related_count = len(data.get("related_works", []))
        openalex_id = data.get("id", "")

        # DOI'den direkt makale bağlantısı oluştur
        read_link = f"https://doi.org/{doi}"

        results.append({
            "DOI": doi,
            "OpenAlex_ID": openalex_id,
            "Title": title,
            "Abstract": abstract,
            "Year": publication_year,
            "Publication_Date": publication_date,
            "Cited_By_Count": cited_by_count,
            "Is_Open_Access": is_oa,
            "Journal": journal,
            "Publisher": publisher,
            "Concepts": concepts_list,
            "Authors": authors_list,
            "Referenced_Works_Count": referenced_count,
            "Related_Works_Count": related_count,
            "Read_Link": read_link  # ✅ burada güncellendi
        })

        time.sleep(0.5)
        df = pd.DataFrame(results)
        df_filtered = df[df["Is_Open_Access"] == True]

    return df_filtered
