import os
import pandas as pd
from PIL import Image

# Caminhos
annotations_path = 'annotations.csv'  # Caminho do CSV
images_folder = 'dataset/images'              # Pasta com as imagens
output_folder = 'yolo_labels'         # Pasta de saída para os .txt

# Cria a pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# Lê o CSV
df = pd.read_csv(annotations_path)
df.columns = df.columns.str.strip()  # Limpa espaços nos nomes das colunas

# Agrupa por imagem
grouped = df.groupby('image_name')

for image_name, group in grouped:
    image_path = os.path.join(images_folder, image_name)

    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except FileNotFoundError:
        print(f'Imagem não encontrada: {image_path}')
        continue

    yolo_lines = []
    for _, row in group.iterrows():
        x1 = row['top_x']
        y1 = row['top_y']
        x2 = row['bottom_x']
        y2 = row['bottom_y']

        # Normaliza usando as dimensões da imagem
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    txt_filename = os.path.splitext(image_name)[0] + '.txt'
    with open(os.path.join(output_folder, txt_filename), 'w') as f:
        f.write('\n'.join(yolo_lines))

print('Conversão concluída!')
