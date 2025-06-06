import torch
import pandas as pd

# Percorso del file .pth
pth_path = "../notebook/image_text_model_final_pro_gan.pth"

# Carica lo state_dict
state_dict = torch.load(pth_path, map_location=torch.device("cpu"))

# Estrai i pesi del layer fusion_proj.0.weight
W_fusion = state_dict["fusion_proj.0.weight"]  # shape (128, 384)

# Considera tutti i neuroni
W_fusion_subset = W_fusion[:128, :]  # shape (10, 384)

# Suddividi nei 3 blocchi: immagine (0-127), testo (128-255), interazione (256-383)
W_image = W_fusion_subset[:, :128]
W_text = W_fusion_subset[:, 128:256]
W_inter = W_fusion_subset[:, 256:]

# Calcolo delle somme assolute
abs_sum_image = W_image.abs().sum(dim=1)
abs_sum_text = W_text.abs().sum(dim=1)
abs_sum_inter = W_inter.abs().sum(dim=1)

# Calcolo delle medie e deviazioni standard
mean_image = W_image.abs().mean(dim=1)
mean_text = W_text.abs().mean(dim=1)
mean_inter = W_inter.abs().mean(dim=1)

std_image = W_image.abs().std(dim=1)
std_text = W_text.abs().std(dim=1)
std_inter = W_inter.abs().std(dim=1)

# Costruzione DataFrame
df = pd.DataFrame({
    "Neuron": list(range(1, 129)),
    "Image_sum": abs_sum_image.numpy(),
    "Text_sum": abs_sum_text.numpy(),
    "Inter_sum": abs_sum_inter.numpy(),
    "Image_mean": mean_image.numpy(),
    "Text_mean": mean_text.numpy(),
    "Inter_mean": mean_inter.numpy(),
    "Image_std": std_image.numpy(),
    "Text_std": std_text.numpy(),
    "Inter_std": std_inter.numpy()
})

# Percentuali rispetto alla somma totale per ciascun neurone
df["Total_sum"] = df[["Image_sum", "Text_sum", "Inter_sum"]].sum(axis=1)
df["Image_%"] = df["Image_sum"] / df["Total_sum"]
df["Text_%"] = df["Text_sum"] / df["Total_sum"]
df["Inter_%"] = df["Inter_sum"] / df["Total_sum"]

# Stampa i risultati
print(df)

# Salva anche su CSV
df.to_csv("analisi_neuroni.csv", index=False)
