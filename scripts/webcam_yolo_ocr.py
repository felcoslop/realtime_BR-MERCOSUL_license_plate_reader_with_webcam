import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
import os
from ultralytics import YOLO

# Configurações do Tesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# Definir TESSDATA_PREFIX
os.environ['TESSDATA_PREFIX'] = 'C:/Program Files/Tesseract-OCR/tessdata/'

# Carregar modelo YOLO
model = YOLO("C:/Users/manu_/license_plate_reader/scripts/license_plate_detector.pt")

# Carregar banco de dados de placas
try:
    df = pd.read_csv("C:/Users/manu_/license_plate_reader/scripts/placa.csv")
    plate_list = df['placa'].str.strip().str.upper().tolist()
    print("Banco de dados de placas carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar banco de dados: {e}")
    plate_list = []

def substituir_letras_por_numeros(ultimos_4_caracteres):
    """Gerar possibilidades substituindo letras por números semelhantes."""
    corrections = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}
    possibilidades = [ultimos_4_caracteres]
    for i, char in enumerate(ultimos_4_caracteres):
        if char in corrections:
            nova_possibilidade = list(ultimos_4_caracteres)
            nova_possibilidade[i] = corrections[char]
            possibilidades.append(''.join(nova_possibilidade))
    return list(set(possibilidades))  # Remover duplicatas

def gerar_possibilidades_mercosul(ultimos_4_caracteres):
    """Gerar possibilidades para placas Mercosul."""
    corrections = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'A': '4'}
    possibilidades = [ultimos_4_caracteres]
    for i, char in enumerate(ultimos_4_caracteres):
        if i == 1 and char in corrections:  # Posição 4 (letra ou número)
            nova_possibilidade = list(ultimos_4_caracteres)
            nova_possibilidade[i] = corrections[char]
            possibilidades.append(''.join(nova_possibilidade))
    return list(set(possibilidades))

def encontrar_placa(string):
    """Procurar placa no padrão antigo (3 letras + 4 números)."""
    padrao = r'[A-Z]{3}\d{4}'
    placas_encontradas = re.findall(padrao, string)
    return placas_encontradas[0] if placas_encontradas else None

def encontrar_placa_mercosul(string):
    """Procurar placa no padrão Mercosul (3 letras + 1 número + 1 letra/número + 2 números)."""
    padrao = r'[A-Z]{3}[0-9][0-9A-Z][0-9]{2}'
    placas_encontradas = re.findall(padrao, string)
    return placas_encontradas[0] if placas_encontradas else None

def preprocess_for_ocr(image):
    """Pré-processamento para OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=-30)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned

def aplicar_ocr(plate_img):
    """Aplicar OCR e validar placas normais ou Mercosul."""
    try:
        # Pré-processar imagem para OCR
        placa_recortada_processada = preprocess_for_ocr(plate_img)
        placa_recortada = plate_img  # Imagem original para exibição

        # Ajustar recorte para placas altas
        x, y, w, h = cv2.boundingRect(placa_recortada_processada)
        if h > 120:
            placa_recortada_processada = placa_recortada_processada[30:-10]

        # Aumentar resolução
        placa_recortada_processada = cv2.resize(placa_recortada_processada, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Configuração comum do Tesseract
        config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6 --oem 3'

        # Tentar OCR em português
        try:
            resultado_tesseract_por = pytesseract.image_to_string(placa_recortada_processada, lang='sl7tech', config=config)
            placa_detectada_por = "".join(filter(str.isalnum, resultado_tesseract_por)).upper()

            # Validar placa Mercosul
            placa_mercosul = encontrar_placa_mercosul(placa_detectada_por)
            if placa_mercosul:
                return placa_mercosul

            # Validar placa antiga
            placa_antiga = encontrar_placa(placa_detectada_por)
            if placa_antiga:
                return placa_antiga
        except Exception as e:
            print(f"Erro ao tentar OCR em português: {e}")

        # Tentar OCR em inglês
        try:
            resultado_tesseract_eng = pytesseract.image_to_string(placa_recortada_processada, lang='eng', config=config)
            placa_detectada_eng = "".join(filter(str.isalnum, resultado_tesseract_eng)).upper()

            # Validar placa Mercosul
            placa_mercosul = encontrar_placa_mercosul(placa_detectada_eng)
            if placa_mercosul:
                return placa_mercosul

            # Validar placa antiga
            placa_antiga = encontrar_placa(placa_detectada_eng)
            if placa_antiga:
                return placa_antiga
        except Exception as e:
            print(f"Erro ao tentar OCR em inglês: {e}")

        # Gerar possibilidades para correção (usando resultado em português, se disponível)
        ultimos_4_caracteres = placa_detectada_por[-4:] if 'placa_detectada_por' in locals() and len(placa_detectada_por) >= 4 else placa_detectada_eng[-4:] if 'placa_detectada_eng' in locals() and len(placa_detectada_eng) >= 4 else ""
        if ultimos_4_caracteres:
            possibilidades = substituir_letras_por_numeros(ultimos_4_caracteres)
            possibilidades_mercosul = gerar_possibilidades_mercosul(ultimos_4_caracteres)

            for possibilidade in possibilidades:
                candidate = (placa_detectada_por[:3] if 'placa_detectada_por' in locals() else placa_detectada_eng[:3] if 'placa_detectada_eng' in locals() else "") + possibilidade
                if encontrar_placa(candidate):
                    return candidate
            for possibilidade in possibilidades_mercosul:
                candidate = (placa_detectada_por[:3] if 'placa_detectada_por' in locals() else placa_detectada_eng[:3] if 'placa_detectada_eng' in locals() else "") + possibilidade
                if encontrar_placa_mercosul(candidate):
                    return candidate

        return ""  # Retorna vazio se nenhuma placa válida for encontrada
    except Exception as e:
        print(f"Erro geral no OCR: {e}")
        return ""

# Inicializar webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()

            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            plate_text = aplicar_ocr(plate_img)

            if plate_text:
                print(f"Placa lida: {plate_text}")

                if plate_text in plate_list:
                    status = "PLACA CADASTRADA"
                    color = (0, 255, 0)  # Verde
                else:
                    status = "PLACA NAO CADASTRADA"
                    color = (0, 165, 255)  # Laranja
            else:
                status = "TEXTO NAO LIDO"
                color = (0, 0, 255)  # Vermelho

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            base_y = y1 - 10 if y1 > 50 else y2 + 20

            cv2.putText(frame, plate_text if plate_text else "N/A",
                        (x1, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, status, (x1, base_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Conf: {conf:.2f}", (x1, base_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Sistema de Reconhecimento de Placas', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()