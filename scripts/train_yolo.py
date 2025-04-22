from ultralytics import YOLO
import os

# Configuração absoluta à prova de erros
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, 'data.yaml')
MODEL_PATH = os.path.join(BASE_DIR, 'yolo11n.pt')  # Caminho absoluto para o modelo

# Verificação manual (IMPORTANTE!)
print("="*50)
print("VERIFICAÇÃO DE CAMINHOS:")
print(f"Diretório base: {BASE_DIR}")
print(f"data.yaml existe? {os.path.exists(DATA_YAML)}")
print(f"Modelo yolov11n.pt existe? {os.path.exists(MODEL_PATH)}")
print(f"Pasta train existe? {os.path.exists(os.path.join(BASE_DIR, 'dataset/train/images'))}")
print("="*50)

# Treinamento
try:
    print("Iniciando treinamento...")
    model = YOLO(MODEL_PATH)  # Usando caminho absoluto
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        batch=16,
        imgsz=640,
        device='cpu',  # ou 'cuda' para GPU
        verbose=True
    )
    print("Treinamento concluído com sucesso!")
except Exception as e:
    print(f"ERRO durante o treinamento: {str(e)}")