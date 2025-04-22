from ultralytics import YOLO
import cv2

# 1. Carregar o modelo treinado
model = YOLO("license_plate_detector.pt")  # Verifique se este caminho está correto

# 2. Carregar imagem (substitua pelo caminho da sua imagem)
image_path = "OKL0817.jpeg"  # Exemplo: "../dataset/test/images/placa001.jpg"
image = cv2.imread(image_path)

# Verificar se a imagem foi carregada corretamente
if image is None:
    print(f"Erro: Não foi possível carregar a imagem em {image_path}")
    exit()

# 3. Fazer inferência na imagem
results = model(image)  # Detecta objetos na imagem

# 4. Processar resultados
annotated_image = results[0].plot()  # Gera imagem com as detecções

# 5. Mostrar e salvar resultados
# Mostrar na janela
cv2.imshow("Detecção YOLOv11", annotated_image)
cv2.waitKey(0)  # Espera até que qualquer tecla seja pressionada

# Salvar imagem com detecções (opcional)
output_path = "resultado_detecao.jpg"
cv2.imwrite(output_path, annotated_image)
print(f"Resultado salvo em: {output_path}")

# Fechar janelas
cv2.destroyAllWindows()