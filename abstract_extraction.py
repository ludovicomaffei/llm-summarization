import os
import time
from tika import parser
import re

def pulisci_testo(testo):
    linee_pulite = []
    for linea in testo.splitlines():
        if not re.match(r"^[0-9\s\.\,\-\(\)]+$", linea.strip()):  
            if len(linea.strip()) > 2:  
                linee_pulite.append(linea)
    return "\n".join(linee_pulite)

def estrai_abstract(cartella_input, file_output):
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    cartella_pdf = os.path.join(desktop, cartella_input)
    output_txt = os.path.join(desktop, file_output)
    
    
    
    delimitatori_fine = [
        "keywords", "key words", "index terms", "contents", "acknowledgment", 
        "1 Introduction", "1. Introduction", "I.", "Contents", "i n t r o d u c t i o n",
        "1 overview of the idea", "citation:", "I. Introduction", "1 Explainability through attention maps"
    ]
    
    
    start_time = time.time()

    pdf_senza_abstract = []  
    pdf_parole_superiori_500 = []  

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Abstracts estratti dai PDF:\n\n")
        
        
        for idx, file_name in enumerate(os.listdir(cartella_pdf), 1):
            if file_name.endswith(".pdf"):
                print(f"Analizzando PDF {idx}: {file_name}...")
                percorso_file = os.path.join(cartella_pdf, file_name)
                
                try:
                    
                    parsed = parser.from_file(percorso_file)
                    testo = parsed["content"]
                    
                    if not testo:
                        print(f"Il file {file_name} non contiene testo valido.")
                        pdf_senza_abstract.append((idx, file_name))
                        continue
                    
                    
                    testo_pulito = pulisci_testo(testo)

                    
                    titolo = f"PDF {idx}: {file_name}\n"
                    f.write(titolo)
                    
                    
                    abstract_inizio = testo_pulito.lower().find("abstract")
                    if abstract_inizio != -1:
                        
                        abstract_fine = len(testo_pulito)
                        for delimitatore in delimitatori_fine:
                            pattern = r"\b" + re.escape(delimitatore.lower()) + r"\b"
                            match = re.search(pattern, testo_pulito.lower()[abstract_inizio:])
                            if match:
                                posizione = abstract_inizio + match.start()
                                if posizione < abstract_fine:
                                    abstract_fine = posizione
                        
                        
                        abstract = testo_pulito[abstract_inizio:abstract_fine].strip()
                        f.write(f"Abstract:\n{abstract}\n\n")
                        
                        
                        if len(abstract.split()) > 500:
                            pdf_parole_superiori_500.append((idx, file_name))
                    else:
                        f.write("Abstract non trovato.\n\n")
                        pdf_senza_abstract.append((idx, file_name))  
                except Exception as e:
                    f.write(f"Errore nell'elaborazione di {file_name}: {e}\n\n")
                    pdf_senza_abstract.append((idx, file_name))  
    
    
    with open(output_txt, "a", encoding="utf-8") as f:
        f.write("\nElenco dei PDF senza abstract:\n")
        if pdf_senza_abstract:
            for idx, file_name in pdf_senza_abstract:
                f.write(f"{idx}. {file_name}\n")
        else:
            f.write("Tutti i PDF contengono un abstract.\n")
        
        
        f.write("\nElenco dei PDF con più di 500 parole nell'abstract:\n")
        if pdf_parole_superiori_500:
            for idx, file_name in pdf_parole_superiori_500:
                f.write(f"{idx}. {file_name}\n")
        else:
            f.write("Nessun PDF ha un abstract con più di 500 parole.\n")
    
   
    elapsed_time = time.time() - start_time
    print(f"Esecuzione completata in {elapsed_time:.2f} secondi.")
    print(f"Abstracts salvati in: {output_txt}")


estrai_abstract("articoli", "Abstracts.txt")
