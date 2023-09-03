import os
import shutil

# Definisci la cartella in cui si trovano i file TXT
cartella_input = 'C:/Users/sghe9/Desktop/train/labels'

# Definisci la cartella di destinazione dei file modificati
cartella_output = 'C:/Users/sghe9/Desktop/train/labels_mod'

# Crea la cartella di destinazione se non esiste già
if not os.path.exists(cartella_output):
    os.makedirs(cartella_output)

# Elencare tutti i file nella cartella di origine con estensione .txt
elenco_file = [f for f in os.listdir(cartella_input) if f.endswith('.txt')]

# Cicla attraverso i file nella lista
for nome_file in elenco_file:
    # Crea il percorso completo del file di origine
    percorso_file_input = os.path.join(cartella_input, nome_file)
    
    # Apri il file di origine in modalità lettura
    with open(percorso_file_input, 'r') as file:
        # Inizializza una lista per memorizzare i dati
        dati = []

        # Leggi il file riga per riga
        for riga in file:
            # Suddividi la riga in una lista di numeri float utilizzando gli spazi come separatore
            valori = [float(valore) for valore in riga.split()]
            
            # Assicurati che il primo numero sia un intero
            valori[0] = int(valori[0])
            
            # Aggiungi 1 al primo numero della riga
            valori[0] += 1
            
            # Aggiungi la lista di valori alla lista dei dati
            dati.append(valori)

    # Crea il percorso per il file modificato nella cartella di destinazione
    percorso_file_output = os.path.join(cartella_output, nome_file)

    # Apri il file modificato nella cartella di destinazione in modalità scrittura e salva i dati modificati
    with open(percorso_file_output, 'w') as file_modificato:
        for riga in dati:
            # Converti i numeri in stringhe e uniscili separandoli con spazi
            riga_modificata = ' '.join(map(str, riga))
            
            # Scrivi la riga modificata nel file originale
            file_modificato.write(riga_modificata + '\n')

    print(f"File '{nome_file}' salvato con i dati modificati.")
