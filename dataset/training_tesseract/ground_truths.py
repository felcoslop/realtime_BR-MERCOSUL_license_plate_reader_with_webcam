from PIL import Image, ImageEnhance, ImageOps
import os
import re

# Configurações
plates_dir = 'C:/Users/manu_/license_plate_reader/dataset/training_tesseract/plates'
gt_dir = 'C:/Users/manu_/license_plate_reader/dataset/training_tesseract/ground_truth'
target_height = 90
min_width = 300

os.makedirs(gt_dir, exist_ok=True)

def preprocess_image(img):
    """Pré-processamento com garantia de texto preto em fundo branco"""
    # Converte para escala de cinza
    img = img.convert('L')
    
    # Aumenta contraste
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    # Binarização (threshold adaptativo)
    img = img.point(lambda x: 0 if x < 140 else 255)
    
    # FORÇA texto preto em fundo branco (inverte se necessário)
    if img.getextrema()[0] < 100:  # Se fundo estiver escuro
        img = ImageOps.invert(img)
    
    # Garantia final: branco predominante
    if img.getextrema()[0] < 128:  # Se ainda tiver muitos pixels escuros
        img = ImageOps.invert(img)
    
    return img

for filename in os.listdir(plates_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            # Extrai o texto da placa
            plate_text = re.sub(r'_plate_\d+', '', filename)
            plate_text = os.path.splitext(plate_text)[0].upper()
            
            # Processamento da imagem
            img = Image.open(os.path.join(plates_dir, filename))
            
            # Redimensionamento proporcional
            width, height = img.size
            new_width = max(min_width, int((target_height / height) * width))
            img = img.resize((new_width, target_height), Image.LANCZOS)
            
            # Pré-processamento
            img = preprocess_image(img)
            
            # Salva como PNG
            output_path = os.path.join(gt_dir, f"{plate_text}.png")
            img.save(output_path, 'PNG', dpi=(300, 300))
            
            # Cria arquivo GT
            with open(os.path.join(gt_dir, f"{plate_text}.gt.txt"), 'w') as f:
                f.write(plate_text)
            
            print(f"Processado: {filename} -> {plate_text}.png (Modo: {img.mode}, Cores: {img.getextrema()})")
            
        except Exception as e:
            print(f"Erro em {filename}: {str(e)}")