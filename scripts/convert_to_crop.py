import os
import cv2
from pathlib import Path

# 1. Configurações iniciais (usando Path para evitar problemas com barras)
base_dir = Path(__file__).parent
plates_dir = base_dir / "plates"
output_dir = base_dir / "ground_truth"
output_dir.mkdir(exist_ok=True)

print("Passo 1/4: Preparando arquivos de treinamento...")

# 2. Criar arquivos de ground truth (.gt.txt)
for plate_file in plates_dir.glob("*.*"):
    if plate_file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
        # Extrai o texto da placa do nome do arquivo (ex: JRK5336_plate_0.jpg -> JRK5336)
        plate_text = plate_file.name.split('_')[0]
        
        # Cria arquivo .gt.txt
        gt_file = output_dir / f"{plate_file.stem}.gt.txt"
        with open(gt_file, 'w', encoding='utf-8') as f:
            f.write(plate_text)

        # Pré-processa a imagem
        img = cv2.imread(str(plate_file))
        if img is None:
            print(f"Erro ao ler a imagem: {plate_file}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Salva imagem processada
        processed_file = plates_dir / f"processed_{plate_file.name}"
        cv2.imwrite(str(processed_file), thresh)

print("Passo 2/4: Arquivos preparados com sucesso!")
print("\nAgora execute ESTES COMANDOS no terminal (CMD como Administrador):\n")
print(f"cd {base_dir}")
print("""
# 1. Criar arquivos .lstmf (para cada imagem processada)
for %i in (plates\\processed_*.jpg) do tesseract "%i" "%i" --psm 6 lstm.train

# 2. Combinar dados de treinamento
combine_tessdata -e "C:\\Program Files\\Tesseract-OCR\\tessdata\\por.traineddata" dataset\\por.lstm

# 3. Treinar o modelo
lstmtraining --model_output dataset\\placa_model --continue_from dataset\\por.lstm --traineddata "C:\\Program Files\\Tesseract-OCR\\tessdata\\por.traineddata" --train_listfile dataset\\train_list.txt --max_iterations 500

# 4. Criar lista de treinamento (execute isto no PowerShell se o comando acima falhar)
Get-ChildItem -Path plates\\processed_*.lstmf | Select-Object -ExpandProperty Name > dataset\\train_list.txt

# 5. Finalizar o modelo
lstmtraining --stop_training --continue_from dataset\\placa_model_checkpoint --traineddata "C:\\Program Files\\Tesseract-OCR\\tessdata\\por.traineddata" --model_output dataset\\placa.traineddata

# 6. Instalar o modelo treinado
copy "dataset\\placa.traineddata" "C:\\Program Files\\Tesseract-OCR\\tessdata\\"
""")

print("\nPara testar o modelo depois do treinamento:")
print('tesseract "plates\\processed_JRK5336_plate_0.jpg" stdout -l placa')