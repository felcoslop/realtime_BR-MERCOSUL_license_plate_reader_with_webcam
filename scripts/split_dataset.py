import os
import shutil
import random

# Definir caminhos (ajustados para sua estrutura)
base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "yolo_labels")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

print("="*50)
print("Configuração de diretórios:")
print(f"Script sendo executado em: {os.path.abspath(__file__)}")
print(f"Diretório base do dataset: {base_dir}")
print(f"Imagens encontradas em: {images_dir}")
print(f"Labels encontrados em: {labels_dir}")
print("="*50)

# Verificar se os diretórios de origem existem
if not os.path.exists(images_dir):
    raise FileNotFoundError(f"Diretório de imagens não encontrado: {images_dir}")
if not os.path.exists(labels_dir):
    raise FileNotFoundError(f"Diretório de labels não encontrado: {labels_dir}")

# Criar diretórios de destino
for split in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(split, "images"), exist_ok=True)
    os.makedirs(os.path.join(split, "labels"), exist_ok=True)

# Listar todas as imagens (com verificação de labels correspondentes)
valid_images = []
for f in os.listdir(images_dir):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        label_file = f.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            valid_images.append(f)
        else:
            print(f"Aviso: Label não encontrado para {f} - imagem ignorada")

if not valid_images:
    raise ValueError("Nenhuma imagem válida com label correspondente encontrada")

random.shuffle(valid_images)

# Calcular tamanhos dos conjuntos
total = len(valid_images)
train_size = int(0.7 * total)
val_size = int(0.2 * total)
test_size = total - train_size - val_size

# Dividir imagens
train_images = valid_images[:train_size]
val_images = valid_images[train_size:train_size + val_size]
test_images = valid_images[train_size + val_size:]

# Função para copiar arquivos com verificação
def copy_files(file_list, split_name):
    copied = 0
    for img in file_list:
        try:
            # Copiar imagem
            img_src = os.path.join(images_dir, img)
            img_dst = os.path.join(base_dir, split_name, "images", img)
            shutil.copy(img_src, img_dst)
            
            # Copiar label correspondente
            label = os.path.splitext(img)[0] + '.txt'
            label_src = os.path.join(labels_dir, label)
            label_dst = os.path.join(base_dir, split_name, "labels", label)
            
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)
                copied += 1
            else:
                print(f"Aviso: Label não encontrado para {img} em {split_name}")
        except Exception as e:
            print(f"Erro ao copiar {img}: {str(e)}")
    return copied

# Mover arquivos
print("\nCopiando arquivos...")
train_copied = copy_files(train_images, "train")
val_copied = copy_files(val_images, "val")
test_copied = copy_files(test_images, "test")

print("\n" + "="*50)
print("Divisão concluída com sucesso!")
print(f"Treino: {train_copied}/{len(train_images)} imagens e labels")
print(f"Validação: {val_copied}/{len(val_images)} imagens e labels")
print(f"Teste: {test_copied}/{len(test_images)} imagens e labels")
print("="*50)

# Verificação final
total_copied = train_copied + val_copied + test_copied
if total_copied != total:
    print(f"Aviso: Alguns arquivos não foram copiados ({total_copied}/{total} copiados)")
else:
    print("Todos arquivos copiados com sucesso!")