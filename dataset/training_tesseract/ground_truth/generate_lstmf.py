import pytesseract
from PIL import Image

# Caminho da imagem
img_path = "C:/Users/manu_/license_plate_reader/dataset/training_tesseract/ground_truth/FJB4E12.png"

# Carregar a imagem
img = Image.open(img_path)

# Usar o Tesseract para realizar OCR na imagem
texto = pytesseract.image_to_string(img)

# Exibir o texto extraído
print(texto)
