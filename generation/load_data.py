import os
import glob
from langchain_community.document_loaders import UnstructuredHTMLLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
import markdown as md
from tqdm import tqdm
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

doc_list = ['amaranth', 'Icarus_verilog', 'klayout', 'qflow', 'OpenROAD', 'OpenSTA', 'OpenROAD_flow_script',
            'verilator', 'yosys_hq']


def load_html(
        folder_path: str,
):
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    html_files = glob.glob(os.path.join(folder_path, '**/*.html'), recursive=True)

    documents = []
    for file_path in tqdm(html_files, desc='Loading HTML files'):
        source = ''
        for item in doc_list:
            if item in file_path:
                source = item
        try:
            content = UnstructuredHTMLLoader(file_path=file_path).load()
            content[0].metadata["source"] = source
            documents.append(content)
        except:
            continue
    return documents


def md_to_text(md_content: str) -> str:
    html = md.markdown(md_content)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


def load_docs(folder_path: str):
    md_files = glob.glob(os.path.join(folder_path, '**/*.md'), recursive=True)
    documents = []

    for file_path in tqdm(md_files, desc='Loading Markdown files'):
        source = ''
        for item in doc_list:
            if item in file_path:
                source = item
        with open(file_path, 'r', encoding='utf-8') as file:
            content = md_to_text(file.read())
            documents.append([Document(page_content=content, metadata={'source': source})])
    return documents


def load_pdfs(folder_path: str):
    md_files = glob.glob(os.path.join(folder_path, '**/*.pdf'), recursive=True)
    documents = []
    for file_path in tqdm(md_files, desc='Loading PDF files'):
        loader = PyMuPDFLoader(file_path)
        content = loader.lazy_load()
        source = ''
        for item in doc_list:
            if item in file_path:
                source = item
        for i in content:
            i.metadata['source'] = source
            documents.append([i])

    return documents


def load_dataset(folder_path: str):
    html = load_html(folder_path)
    documents = load_docs(folder_path)
    pdfs = load_pdfs(folder_path)

    dataset = html + documents + pdfs
    return dataset


def split_text(docs, chunk_size=4096, over_lap=512):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=over_lap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(docs)


def split_docs(documents: list):
    split_sub_texts = []
    for data in documents:
        chunks = split_text(data[0].page_content)
        for chunk in chunks:
            split_sub_texts.append(Document(page_content=chunk, metadata={'source': data[0].metadata['source']}))

    return split_sub_texts
