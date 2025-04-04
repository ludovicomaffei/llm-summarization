# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 18:19:16 2024

@author: EndorPC
"""

import os
import time
import requests


user_desktop = os.path.expanduser("~/Desktop")
input_folder = os.path.join(user_desktop, "articoli")
output_file = os.path.join(user_desktop, "titolo_autore.txt")

def extract_with_grobid(file_path):
    
    url = "http://localhost:8070/api/processHeaderDocument"
    with open(file_path, 'rb') as pdf_file:
        response = requests.post(url, files={'input': pdf_file})
        if response.status_code == 200:
            return response.text
        else:
            print(f"Errore HTTP {response.status_code} per il file {file_path}")
            return None

def parse_metadata(grobid_output):
    
    from xml.etree import ElementTree as ET

    try:
        root = ET.fromstring(grobid_output)
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        
        title_element = root.find('.//tei:title[@type="main"]', ns)
        title = title_element.text.strip() if title_element is not None and title_element.text else "Titolo non disponibile"

        
        authors = []
        for author in root.findall('.//tei:author', ns):
            author_name = author.find('.//tei:persName', ns)
            if author_name is not None:
                full_name = " ".join(
                    filter(None, [
                        author_name.findtext(f"tei:{tag}", default="", namespaces=ns).strip()
                        for tag in ["forename", "surname"]
                    ])
                )
                if full_name:
                    authors.append(full_name)

        return title, authors

    except ET.ParseError as e:
        print(f"Errore durante il parsing XML: {e}")
        return "Titolo non disponibile", []

def main():
    start_time = time.time()
    no_authors = []
    pdf_count = 0

    if not os.path.exists(input_folder):
        print(f"La cartella {input_folder} non esiste.")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.pdf'):
                pdf_count += 1
                pdf_path = os.path.join(input_folder, file_name)
                print(f"Analisi del file: {file_name}")

                grobid_output = extract_with_grobid(pdf_path)
                if grobid_output:
                    title, authors = parse_metadata(grobid_output)
                else:
                    title, authors = "Titolo non disponibile", []

                f.write(f"{pdf_count}) {file_name}\n")
                f.write(f"Title: {title}\n")
                f.write(f"Authors: {', '.join(authors) if authors else 'Non individuati'}\n\n")

                if not authors:
                    no_authors.append(file_name)

        
        f.write("\nRiepilogo:\n")
        f.write(f"Totale PDF analizzati: {pdf_count}\n")
        f.write(f"PDF senza autori identificati: {len(no_authors)}\n")
        if no_authors:
            f.write("Elenco dei PDF senza autori:\n")
            for file_name in no_authors:
                f.write(f"- {file_name}\n")

    end_time = time.time()
    print(f"Tempo di esecuzione: {end_time - start_time:.2f} secondi")

if __name__ == "__main__":
    main()
