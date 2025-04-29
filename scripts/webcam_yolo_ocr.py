import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
import os
from ultralytics import YOLO
import time
from collections import defaultdict
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurações do Tesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
os.environ['TESSDATA_PREFIX'] = 'C:/Program Files/Tesseract-OCR/tessdata/'

# Carregar modelo YOLO
model = YOLO('scripts/license_plate_detector.pt')

# Carregar banco de dados de placas
try:
    df = pd.read_csv('scripts/placa.csv')
    plate_list = df['placa'].str.strip().str.upper().tolist()
    owner_list = df['proprietario'].tolist()
    logging.info("Banco de dados de placas carregado com sucesso!")
except Exception as e:
    logging.error(f"Erro ao carregar banco de dados: {e}")
    plate_list = []
    owner_list = []

# Variáveis globais
found_plate = None
found_owner = None
plate_counts = defaultdict(list)  # Armazena timestamps de detecção por placa
state = "waiting"  # Estados: waiting, unknown, known, input
start_time = 0  # Para controle do tempo de exibição
input_plate = ""
input_owner = ""
input_active = False
input_field = None  # Campo ativo: "plate" ou "owner"

def substituir_letras_por_numeros(ultimos_4_caracteres):
    corrections = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}
    possibilidades = [ultimos_4_caracteres]
    for i, char in enumerate(ultimos_4_caracteres):
        if char in corrections:
            nova_possibilidade = list(ultimos_4_caracteres)
            nova_possibilidade[i] = corrections[char]
            possibilidades.append(''.join(nova_possibilidade))
    return list(set(possibilidades))

def gerar_possibilidades_mercosul(ultimos_4_caracteres):
    corrections = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'A': '4'}
    possibilidades = [ultimos_4_caracteres]
    for i, char in enumerate(ultimos_4_caracteres):
        if i == 1 and char in corrections:
            nova_possibilidade = list(ultimos_4_caracteres)
            nova_possibilidade[i] = corrections[char]
            possibilidades.append(''.join(nova_possibilidade))
    return list(set(possibilidades))

def encontrar_placa(string):
    padrao = r'[A-Z]{3}\d{4}'
    placas_encontradas = re.findall(padrao, string)
    return placas_encontradas[0] if placas_encontradas else None

def encontrar_placa_mercosul(string):
    padrao = r'[A-Z]{3}[0-9][0-9A-Z][0-9]{2}'
    placas_encontradas = re.findall(padrao, string)
    return placas_encontradas[0] if placas_encontradas else None

def preprocess_for_ocr(image):
    if image.size == 0:
        logging.warning("Imagem vazia recebida para OCR")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=-30)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 5)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned

def aplicar_ocr(plate_img):
    try:
        placa_recortada_processada = preprocess_for_ocr(plate_img)
        if placa_recortada_processada is None:
            return ""
        x, y, w, h = cv2.boundingRect(placa_recortada_processada)
        if h > 120:
            placa_recortada_processada = placa_recortada_processada[30:-10]
        placa_recortada_processada = cv2.resize(placa_recortada_processada, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6 --oem 3'

        # OCR em português
        try:
            resultado_tesseract_por = pytesseract.image_to_string(placa_recortada_processada, lang='sl7tech', config=config)
            placa_detectada_por = "".join(filter(str.isalnum, resultado_tesseract_por)).upper()
            logging.debug(f"OCR Português: {placa_detectada_por}")
            placa_mercosul = encontrar_placa_mercosul(placa_detectada_por)
            if placa_mercosul:
                return placa_mercosul
            placa_antiga = encontrar_placa(placa_detectada_por)
            if placa_antiga:
                return placa_antiga
        except Exception as e:
            logging.error(f"Erro ao tentar OCR em português: {e}")

        # OCR em inglês
        try:
            resultado_tesseract_eng = pytesseract.image_to_string(placa_recortada_processada, lang='eng', config=config)
            placa_detectada_eng = "".join(filter(str.isalnum, resultado_tesseract_eng)).upper()
            logging.debug(f"OCR Inglês: {placa_detectada_eng}")
            placa_mercosul = encontrar_placa_mercosul(placa_detectada_eng)
            if placa_mercosul:
                return placa_mercosul
            placa_antiga = encontrar_placa(placa_detectada_eng)
            if placa_antiga:
                return placa_antiga
        except Exception as e:
            logging.error(f"Erro ao tentar OCR em inglês: {e}")

        # Correções
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
        return ""
    except Exception as e:
        logging.error(f"Erro geral no OCR: {e}")
        return ""

def draw_button(frame, text, x, y, w, h, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.putText(frame, text, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def is_point_in_rect(x, y, rect_x, rect_y, rect_w, rect_h):
    return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h

def mouse_callback(event, x, y, flags, param):
    global state, found_plate, input_plate, input_owner, input_active, input_field
    if event == cv2.EVENT_LBUTTONDOWN:
        if state == "unknown":
            # Botão Convidado
            if is_point_in_rect(x, y, 400, 450, 100, 25):
                state = "waiting"
                found_plate = None
            # Botão Adicionar
            elif is_point_in_rect(x, y, 510, 450, 100, 25):
                input_plate = found_plate if found_plate else ""
                input_owner = ""
                input_active = True
                input_field = "plate"
                state = "input"
        elif state == "input":
            # Campo Placa
            if is_point_in_rect(x, y, 50, 410, 200, 25):
                input_field = "plate"
            # Campo Proprietário
            elif is_point_in_rect(x, y, 50, 440, 200, 25):
                input_field = "owner"
            # Botão Confirmar
            elif is_point_in_rect(x, y, 510, 450, 100, 25):
                if input_plate and input_owner:
                    # Adicionar ao banco de dados
                    plate = input_plate.upper()
                    owner = input_owner
                    # Escrever no arquivo CSV com quebra de linha
                    with open('scripts/placa.csv', mode='a', newline='', encoding='utf-8') as f:
                        # Se o arquivo não existe, adicionar cabeçalho
                        if not os.path.exists('scripts/placa.csv') or os.path.getsize('scripts/placa.csv') == 0:
                            f.write('placa,proprietario\n')
                        f.write(f'{plate},{owner}\n')
                    global plate_list, owner_list
                    plate_list.append(plate)
                    owner_list.append(owner)
                    state = "waiting"
                    found_plate = None
                    input_active = False

# Inicializar webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow('Sistema de Reconhecimento de Placas')
cv2.setMouseCallback('Sistema de Reconhecimento de Placas', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Falha ao capturar frame da webcam")
        break

    current_time = time.time()

    # Limpar detecções antigas (mais de 2 segundos)
    for plate in list(plate_counts.keys()):
        plate_counts[plate] = [t for t in plate_counts[plate] if current_time - t < 2]
        if not plate_counts[plate]:
            del plate_counts[plate]

    if state == "waiting":
        # Faixa inferior amarela com "SEM CARROS NA PORTARIA"
        cv2.rectangle(frame, (0, 405), (640, 480), (0, 255, 255), -1)
        cv2.putText(frame, "SEM CARROS NA PORTARIA", (50, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Processar detecção de placas
        results = model(frame, conf=0.6)
        detected_plates = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Desenhar caixa de bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size == 0:
                    logging.warning("Imagem da placa vazia")
                    continue

                plate_text = aplicar_ocr(plate_img)
                if plate_text:
                    logging.info(f"Placa detectada: {plate_text}")
                    plate_counts[plate_text].append(current_time)
                    detected_plates.append(plate_text)
                    logging.debug(f"Contagem para {plate_text}: {len(plate_counts[plate_text])}")

        # Selecionar a placa com mais detecções
        if detected_plates:
            most_common_plate = max(plate_counts.items(), key=lambda x: len(x[1]), default=(None, []))[0]
            if most_common_plate and len(plate_counts[most_common_plate]) >= 3:
                logging.info(f"Placa confirmada: {most_common_plate} com {len(plate_counts[most_common_plate])} detecções")
                found_plate = most_common_plate
                if found_plate in plate_list:
                    idx = plate_list.index(found_plate)
                    found_owner = owner_list[idx]
                    state = "known"
                    start_time = current_time
                    logging.info(f"Estado alterado para known: {found_plate}, Proprietário: {found_owner}")
                else:
                    state = "unknown"
                    start_time = current_time
                    logging.info(f"Estado alterado para unknown: {found_plate}")

    elif state == "known":
        # Faixa inferior verde com placa e proprietário
        cv2.rectangle(frame, (0, 405), (640, 480), (0, 255, 0), -1)
        cv2.putText(frame, "Placa encontrada na base de dados", (50, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, f"Placa: {found_plate}", (50, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, f"Proprietario: {found_owner}", (50, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Verificar tempo ou comando do porteiro
        if current_time - start_time > 40 or cv2.waitKey(1) & 0xFF == ord('c'):
            state = "waiting"
            found_plate = None
            found_owner = None
            logging.info("Estado alterado para waiting")

    elif state == "unknown":
        # Faixa inferior vermelha com opções
        cv2.rectangle(frame, (0, 405), (640, 480), (0, 0, 255), -1)
        cv2.putText(frame, "Carro nao esta na base de dados", (50, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, f"Placa: {found_plate}", (50, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        draw_button(frame, "Convidado", 400, 450, 100, 25, (255, 255, 255))
        draw_button(frame, "Adicionar", 510, 450, 100, 25, (255, 255, 255))

        # Verificar tempo para voltar ao estado waiting
        if current_time - start_time > 40 or cv2.waitKey(1) & 0xFF == ord('c'):
            state = "waiting"
            found_plate = None
            logging.info("Estado alterado para waiting")

    elif state == "input":
        # Faixa inferior para entrada de dados
        cv2.rectangle(frame, (0, 405), (640, 480), (255, 255, 255), -1)
        cv2.putText(frame, "Inserir Dados", (50, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # Campo Placa
        cv2.rectangle(frame, (50, 410), (250, 435), (200, 200, 200), -1)
        cv2.putText(frame, input_plate, (55, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # Campo Proprietário
        cv2.rectangle(frame, (50, 440), (250, 465), (200, 200, 200), -1)
        cv2.putText(frame, input_owner, (55, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # Botão Confirmar
        draw_button(frame, "Confirmar", 510, 450, 100, 25, (0, 255, 0))

    # Capturar entrada de texto
    key = cv2.waitKey(1) & 0xFF
    if input_active:
        if key == 13:  # Enter
            input_field = "owner" if input_field == "plate" else None
        elif key == 8:  # Backspace
            if input_field == "plate":
                input_plate = input_plate[:-1]
            elif input_field == "owner":
                input_owner = input_owner[:-1]
        elif 32 <= key <= 126:  # Caracteres imprimíveis
            if input_field == "plate" and len(input_plate) < 7:
                input_plate += chr(key).upper()
            elif input_field == "owner" and len(input_owner) < 50:
                input_owner += chr(key)

    # Mostrar frame
    cv2.imshow('Sistema de Reconhecimento de Placas', frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()