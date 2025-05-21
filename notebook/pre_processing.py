#!/usr/bin/env python
# coding: utf-8

#Il seguente file python riporta le conversione effettuate nel file notebook "pre_processing.ipynb" che verranno utilizzate nel file "clip_image_classifier.py"


import os
import pandas as pd
import nbimporter


# Percorso principale del progetto
root_dir = '../'  # Sostituisci con il percorso della tua directory principale

# Lista per memorizzare i DataFrame
metadata_list = []

# Scorri tutte le sottocartelle
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file == 'metadata.csv':  # Controlla se il file si chiama metadata.csv
            file_path = os.path.join(subdir, file)
            # Carica il CSV e aggiungilo alla lista
            metadata = pd.read_csv(file_path)
            metadata_list.append(metadata)
# Stampa di tutti i DataFrame caricati nella lista
for i, metadata in enumerate(metadata_list):
    print(f"Dataset {i}:")
    print(metadata)
    print("\n")  # Aggiunge una riga vuota tra un dataset e l'altro


# # Controllo se i file esistono

# In[2]:


# Percorso principale del progetto
root_dir = '../'  # Sostituisci con il percorso della tua directory principale

# Scorri tutte le sottocartelle
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file == 'metadata.csv':  # Controlla se il file si chiama metadata.csv
            file_path = os.path.join(subdir, file)
            if os.path.exists(file_path):
                print(f"File trovato: {file_path}")
            else:
                print(f"File non trovato: {file_path}")


# # Verifico la presenza di valori null all'interno dei dataset

# In[3]:


# Verifica la presenza di valori null in ogni dataset
for i, metadata in enumerate(metadata_list):
    if metadata.isnull().values.any():
        print(f"Il dataset {i} contiene valori null.")
    else:
        print(f"Il dataset {i} non contiene valori null.")


# # Stampa delle colonne che contengono i valori null per ogni dataset

# In[4]:


# Stampa delle colonne con valori null per ogni dataset
for i, metadata in enumerate(metadata_list):
    null_columns = metadata.columns[metadata.isnull().any()]
    if not null_columns.empty:
        print(f"Dataset {i} - Colonne con valori null:")
        print(null_columns)
    else:
        print(f"Dataset {i} - Nessuna colonna con valori null.")


# # Stampa dei valori presenti all'interno della colonna 'category' per ogni dataset

# In[5]:


# Stampa dei valori presenti nella colonna 'category' per ogni dataset
for i, metadata in enumerate(metadata_list):
    if 'category' in metadata.columns:
        print(f"Dataset {i} - Valori nella colonna 'category':")
        print(metadata['category'].unique())
    else:
        print(f"Dataset {i} - La colonna 'category' non è presente.")


# # Grafico della distribuzione della variabile target

# In[6]:


import matplotlib.pyplot as plt

# Unisci i valori della colonna target di tutti i dataset
all_targets = []
for i, metadata in enumerate(metadata_list):
    if 'target' in metadata.columns:  # Sostituisci 'target' con il nome effettivo della variabile target
        all_targets.extend(metadata['target'])


# # Sostituzione della variabile NaN con Unknown per la colonna Category in alcuni dataset
# Si è scelto come variabile "Unknown" per i valori NaN poichè si tratta di una stringa ed inoltre non si conosce il valore della categoria

# In[7]:


# Sostituzione dei valori NaN nella colonna 'category' con 'unknown' e stampa dei dataset aggiornati
for i, metadata in enumerate(metadata_list):
    if 'category' in metadata.columns:
        if metadata['category'].isnull().any():
            metadata['category'] = metadata['category'].fillna('unknown')
            print(f"Dataset {i} aggiornato (valori NaN in 'category' sostituiti con 'unknown'):")
            print(metadata)
            print("\n")


# # Check della colonna target in tutti i dataset e sostituzione dei valori > 1 con 1
# 
# Dal momento che la variabile target in alcuni dataset presenta valori compresi tra 0 e 6, si è effettuata una visita di tutti i dataset con le corrispettive cartelle contenente immagini allegate e si è verificati se quest'ultime sono immagini generate. In seguito, si è effettuata la sostituzione di tutti i valori all'interno della feateure target >1 con 1 che indica che l'immagine è fake.

# In[8]:


# Funzione per modificare i valori nella colonna 'target' e stampare i dataset
def modifica_target(metadata_list):
    for i, metadata in enumerate(metadata_list):
        if 'target' in metadata.columns:
            # Stampa dei valori unici nella colonna 'target' prima della modifica
            print(f"Dataset {i} - Valori unici nella colonna 'target' prima della modifica:")
            print(metadata['target'].unique())

            # Modifica dei valori nella colonna 'target'
            metadata['target'] = metadata['target'].apply(lambda x: 1 if x > 1 else x)

            # Stampa del dataset aggiornato
            print(f"Dataset {i} aggiornato (valori > 1 nella colonna 'target' sostituiti con 1):")
            print(metadata)
            print("\n")

# Esegui la funzione
modifica_target(metadata_list)


#  # Stampa dei dataset che contengono sia 0 che 1 nella colonna 'target'

# In[11]:


# Conta i dataset che contengono sia 0 che 1 nella colonna 'target' e stampa i nomi dei dataset
def conta_dataset_target(metadata_list):
    count = 0
    dataset_riferimento = []

    for i, metadata in enumerate(metadata_list):
        if 'target' in metadata.columns:
            unique_values = set(metadata['target'].unique())
            if 0 in unique_values and 1 in unique_values:
                count += 1
                dataset_riferimento.append(f"Dataset {i}")

    print(f"Numero di dataset che contengono sia 0 che 1 nella colonna 'target': {count}")
    print("Dataset di riferimento:")
    for dataset in dataset_riferimento:
        print(dataset)

# Esegui la funzione
conta_dataset_target(metadata_list)

