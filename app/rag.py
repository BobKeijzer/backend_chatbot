import os
import PyPDF2 
import pandas as pd
from docx import Document as Doc
import numpy as np
import sqlite3
import tempfile
import io
from langchain_core.documents import Document

# -------------------
# Chunking logic 
# -------------------
def chunk_file(file, filename, chunk_size=500, overlap=50):
    text = extract_text_from_file(file, filename)
    ext = filename.split(".")[-1].lower()
    chunks = []
    if ext in {"db", "sqlite"}:
        try:
            tables = text.split("\n\n")  
            for table in tables:
                rows = table.splitlines()
                if not rows: 
                    continue
                table_name = rows[0] 
                headers = rows[1] if len(rows) > 1 else "" 
                for i in range(2, len(rows), chunk_size):
                    chunk = "\n".join([table_name, headers] + rows[i:i + chunk_size])
                    chunks.append(chunk)
        except Exception:
            chunks = []

    elif ext in {"csv", "xlsx", "xls"}:
        try:
            rows = text.splitlines()
            if rows:
                headers = rows[0]
            chunks = []

            for i in range(1, len(rows), chunk_size): 
                chunk = "\n".join([headers] + rows[i:i + chunk_size])
                chunks.append(chunk)
        except Exception:
            chunks = []    

    else:
        try:
            words = text.split()
            if len(words) < chunk_size:
                chunks = [text]  
            else:
                chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
        except Exception:
            chunks = []

    return chunks

# -------------------
# Helper functions 
# -------------------
def extract_text_from_file(file, filename):
    content = ""
    ext = filename.split(".")[-1].lower()
    try:
        if ext == "pdf":
            content = extract_text_from_pdf(file)
        elif ext == "docx":
            content = extract_text_from_docx(file)
        elif ext == "txt":
            content = extract_text_from_txt(file)
        elif ext in {"xlsx", "xls"}:
            content = extract_text_from_excel(file)
        elif ext == "csv":
            content = extract_text_from_csv(file)
        elif ext in {"sqlite", "db"}:
            content = extract_text_from_sqlite(file)
    except Exception as e:
        print(f"Error processing {file}: {e}")
        content = ""
    return content

def extract_text_from_pdf(file):
    try:
        file.seek(0)
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:  
                text += page_text + "\n"
        return text.strip()
    except Exception:
        return ""
    
def extract_text_from_docx(file):
    try:
        data = file.read()
        file.seek(0)
        doc = Doc(io.BytesIO(data))
        text = "\n\n".join(para.text for para in doc.paragraphs)
        return text.strip()
    except Exception:
        return ""

def extract_text_from_txt(file):
    try:
        text = file.read().decode("utf-8").strip()
        file.seek(0)
        return text
    except Exception:
        return ""

def extract_text_from_excel(file):
    try:
        data = file.read()
        file.seek(0)
        df = pd.read_excel(io.BytesIO(data))
        text_data = "\n".join([", ".join(df.columns)] + df.astype(str).apply(lambda x: ", ".join(x), axis=1).tolist())
        return text_data
    except Exception:
        return ""

def extract_text_from_csv(file):
    try:
        data = file.read()
        file.seek(0)
        df = pd.read_csv(io.BytesIO(data))
        text_data = "\n".join([", ".join(df.columns)] + df.astype(str).apply(lambda x: ", ".join(x), axis=1).tolist())
        return text_data
    except Exception:
        return ""

def extract_text_from_sqlite(file):
    text_data = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            tmp.write(file.read())
            tmp.flush()
            file.seek(0)

            conn = sqlite3.connect(tmp.name)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT * FROM {table_name};")
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                text_data += f"Table: {table_name}\n"
                text_data += ", ".join(columns) + "\n"
                for row in rows:
                    text_data += ", ".join(map(str, row)) + "\n"
                text_data += "\n"

            conn.close()
    except Exception:
        return ""
    return text_data.strip()