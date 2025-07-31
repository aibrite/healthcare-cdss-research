# Medical PDF Extraction Module

This module is a part of the **healthcare-cdss-research** repository and is responsible for automatically fetching and parsing open-access medical research articles in PDF format, converting them into structured, analyzable content.

## 📁 Directory: `/medical-pdf-extraction/`

---

## 🧠 Overview

This pipeline performs the following tasks:

1. Fetches relevant article metadata using **PubMed (BioEntrez)** and **OpenAlex** APIs.
2. Filters the top cited open-access papers on a given medical topic.
3. Extracts available PDF links from the metadata.
4. Downloads PDFs.
5. Parses PDFs using the **Mineru** backend into structured markdown, annotated images, and JSON outputs.
6. Optionally uses **Gemini API** to analyze extracted scientific images in the context of the article content.

---

## 🗂 File Structure

| File | Description |
|------|-------------|
| `getPDF.ipynb` | Jupyter notebook that runs the entire pipeline from search to PDF download and parsing. |
| `getPDFidsWithBioentrez.py` | Uses BioEntrez API to search PubMed for article metadata (title, DOI, etc.). |
| `openAlexSort.py` | Fetches article metadata from OpenAlex, filters by open-access and citation count. |

---

## 🧬 getPDF.ipynb – Pipeline Stages

### 1. **Initialize and Setup**
Adds custom modules to the Python path and installs dependencies (e.g., `biopython`).

### 2. **Search for Articles**
- Takes a medical topic (e.g., `NAFLD`) and number of desired articles.
- Uses `get_articles()` from `getPDFidsWithBioentrez.py`.

### 3. **Get OpenAlex Metadata**
- Filters open-access articles.
- Sorts by `Cited_By_Count`.
- Extracts `Read_Link` for each article.

### 4. **Extract PDF Links**
- Visits each `Read_Link` and searches for downloadable `.pdf` URLs using `BeautifulSoup`.

### 5. **Download PDFs**
- Downloads PDFs into `/downloaded_pdfs/` folder using `wget`.

### 6. **Parse PDFs**
- Uses `mineru` parsing backend to extract content and generate:
  - Markdown files
  - Bounding box annotated PDFs
  - Structured JSON data

---

## 🧰 Parsing Logic – Mineru Integration

### Main Function: `do_parse(...)`

Parses PDFs using either of the following backends:
- `pipeline`: General parsing with OCR and layout analysis.
- `vlm-*`: Vision-Language Model backends for advanced parsing (e.g., `vlm-sglang-engine`, `vlm-transformers`).

**Outputs include:**
- `*.md`: Markdown article content
- `*_layout.pdf` / `*_span.pdf`: Annotated layout and span bounding box visualizations
- `*_middle.json` / `*_model.json`: Intermediate and model outputs

### Parsing Entry Point:

parse_doc(doc_path_list, output_dir, backend="pipeline", lang="en")

## 🤖 Gemini API Integration

Once the parsing is complete:

- The markdown file (`*.md`) and image visualizations are passed to the Gemini API.
- Gemini analyzes each chart/image in the context of the article.
- Output is saved as `.txt` files alongside the images.

> ✨ *Model used*: `gemini-2.5-flash`  
> 🔑 *API Key*: Set via `genai.Client(api_key="****")`

---

## 🔍 Metadata Modules

### 🔬 `getPDFidsWithBioentrez.py`

- Queries PubMed using BioEntrez API.
- Extracts:
  - `PMID`, `PMCID`, `DOI`
  - `Title`, `Abstract`, `Journal`, `Language`, `Year`, `Month`

### 📖 `openAlexSort.py`

- Fetches metadata for each DOI using OpenAlex API.
- Filters open-access papers.
- Extracts:
  - Citation count
  - Journal and publisher info
  - Author institutions
  - Abstract
  - Concepts
  - Readable article link

---

## 🔧 Configuration Parameters

| Parameter      | Description                                  |
|----------------|----------------------------------------------|
| `topic`        | Medical keyword/topic (e.g., `"NAFLD"`)     |
| `article_count`| Number of articles to fetch from PubMed      |
| `top_n`        | Top N articles to select by citation count   |
| `backend`      | PDF parsing backend (`pipeline` or `vlm-*`) |
| `lang`         | OCR language (e.g., `"en"`)                   |

---

## 📦 Requirements

Install required libraries in your Colab or local environment:

pip install biopython requests beautifulsoup4 tqdm

## ✅ Output

All parsed content and files are stored under: /content/drive/MyDrive/Mineru_Output_pipeline/


Each article will have its own subdirectory containing:

- Markdown article
- Annotated PDFs
- Extracted images
- Gemini analysis results

---

## ✍️ Authors & Credits
- Developed as part of the **healthcare-cdss-research** project.
- PDF parsing powered by [Mineru](https://github.com/opendatalab/mineru).
- Metadata from PubMed and [OpenAlex](https://openalex.org/).
- Image-based insights using Google Gemini.



