import os
import shutil
import cv2
import sys


# Definisci le due cartelle sorgente
cartella1 = 'C:/Users/sghe9/Desktop/dataset/new1/images'
cartella2 = 'C:/Users/sghe9/Desktop/dataset/train/images'

# Definisci la cartella di destinazione
cartella_destinazione = 'C:/Users/sghe9/Desktop/dataset/new2/images'

# Definisci la cartella bin
cartella_bin = 'C:/Users/sghe9/Desktop/dataset/bin'

# Crea la cartella di destinazione se non esiste già
if not os.path.exists(cartella_destinazione):
    os.makedirs(cartella_destinazione)
    
# Crea la cartella bin se non esiste già
if not os.path.exists(cartella_bin):
    os.makedirs(cartella_bin)

# Ottieni la lista di file nelle due cartelle sorgente
files1 = os.listdir(cartella1)
files2 = os.listdir(cartella2)

# Itera attraverso i file nella prima cartella
for file1 in files1:
    # Estrai i primi 6 caratteri dal nome del file
    prefix1 = file1[:3]

    # Cerca un file con lo stesso prefisso nella seconda cartella
    matching_files2 = [file2 for file2 in files2 if file2.startswith(prefix1)]

    if matching_files2:
        move = 0
        # Mostra le immagini affiancate e chiedi all'utente quale mantenere
        for file2 in matching_files2:
            print(f"File dalla cartella 1: {file1}")
            print(f"File dalla cartella 2: {file2}")
            # Inizializza una finestra OpenCV per mostrare le immagini
            cv2.namedWindow('Immagini', cv2.WINDOW_AUTOSIZE)
            
            img1 = cv2.imread(os.path.join(cartella1, file1))
            img2 = cv2.imread(os.path.join(cartella2, file2))
            combined_img = cv2.hconcat([img1, img2])
            cv2.imshow('Immagini', combined_img)
            print("Scegli il file dalla da conservare (1, 2) o 0 per ignorarli o ESC per uscire: ")
            k = cv2.waitKey(0)
            #print(k)
            scelta = k - 48
            print(scelta)
            # Se il file scelto è nella prima cartella
            if scelta == 1:
                cartella_scelta = cartella1
                file_scelto = file1  
                move = 1 
                # Sposta il file non scelto nella cartella bin
                src_path = os.path.join(cartella2, file2)
                dst_path = os.path.join(cartella_bin, file2)
                shutil.move(src_path, dst_path)    
            elif scelta == 2:
                cartella_scelta = cartella2
                file_scelto = file2
                # Sposta il file non scelto nella cartella bin
                src_path = os.path.join(cartella1, file1)
                dst_path = os.path.join(cartella_bin, file1)
                shutil.move(src_path, dst_path)
                # Sposta il file scelto nella cartella di destinazione
                src_path = os.path.join(cartella_scelta, file_scelto)
                dst_path = os.path.join(cartella_destinazione, file_scelto)
                shutil.move(src_path, dst_path)
            elif scelta == -21:
                sys.exit()
                
            # Chiudi la finestra OpenCV
            cv2.destroyAllWindows()
        
        if move == 1:   
            # Sposta il file scelto nella cartella di destinazione
            src_path = os.path.join(cartella_scelta, file_scelto)
            dst_path = os.path.join(cartella_destinazione, file_scelto)
            shutil.move(src_path, dst_path)
                
# Chiede se passare alla fase successiva in caso si voglia modificare la dim del prefisso       
scelta = int(input("Passare alla fase successiva? (3 = Sì, 4 = No): "))
if scelta == 4:
    sys.exit()

# Sposta tutti i file rimanenti dalla prima cartella nella cartella di destinazione
for file1 in files1:
    src_path = os.path.join(cartella1, file1)
    dst_path = os.path.join(cartella_destinazione, file1)
    shutil.move(src_path, dst_path)

# Mostra i file rimanenti dalla seconda cartella e chiedi all'utente se spostarli
for file2 in files2:
    # Inizializza una finestra OpenCV per mostrare le immagini
    cv2.namedWindow('Immagini', cv2.WINDOW_AUTOSIZE)
    img = cv2.imread(os.path.join(cartella2, file2))
    cv2.imshow('Immagini', img)
    print(f"Spostare il file {file2} nella cartella di destinazione? (0 = Sì, 1 = No) o ESC per uscire: ")
    k = cv2.waitKey(0)
    #print(k)
    scelta = k - 48
    print(scelta)
    if scelta == 0:
        src_path = os.path.join(cartella2, file2)
        dst_path = os.path.join(cartella_destinazione, file2)
        shutil.move(src_path, dst_path)
    elif scelta == 1:
        # Sposta il file non scelto nella cartella bin
        src_path = os.path.join(cartella2, file2)
        dst_path = os.path.join(cartella_bin, file2)
        shutil.move(src_path, dst_path)
    elif scelta == -21:
        sys.exit()
    

# Chiudi la finestra OpenCV
cv2.destroyAllWindows()
