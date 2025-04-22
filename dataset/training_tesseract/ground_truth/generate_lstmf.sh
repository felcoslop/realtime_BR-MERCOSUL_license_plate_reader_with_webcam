#!/bin/bash

# Configurações
cd "$(dirname "$0")" || exit 1
shopt -s nocaseglob
shopt -s nullglob

echo "Iniciando processamento de imagens para treinamento LSTM..."

# Processa cada imagem .png
for img in *.png; do
    [ -f "$img" ] || continue

    base="${img%.*}"
    gt_file="${base}.gt.txt"

    if [ -f "$gt_file" ]; then
        echo "Processando: $img -> ${base}.lstmf"

        # Comando Tesseract com tratamento de erro
        if ! tesseract "$img" "$base" --psm 7 lstm.train 2>tesseract.log; then
            echo "ERRO no processamento de $img. Verifique 'tesseract.log'"
            continue
        fi

        # Verifica se o arquivo .lstmf foi gerado
        if [ -f "${base}.lstmf" ]; then
            echo "Arquivo ${base}.lstmf gerado com sucesso!"
        else
            echo "AVISO: ${base}.lstmf não foi gerado (verifique os dados de entrada)"
        fi
    else
        echo "AVISO: Arquivo .txt não encontrado para $img ($gt_file)"
    fi
done

echo "Processamento concluído. Verifique os logs acima."
