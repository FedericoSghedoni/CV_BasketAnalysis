import torch

# Crea un tensore di dimensioni [1, 3, 3] con valori di esempio
tensore = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10,11,12]]])

# Seleziona solo il primo tensore [1, 3]
primo_tensore = tensore[:, :, 0]

# Stampa il primo tensore
print(primo_tensore)
print(primo_tensore.shape)