#!/bin/bash

CHECKPOINT_DIR="src/backbones/MEGraphAU/checkpoints"
CHECKPOINT_DIR_DF="checkpoints"
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${CYAN}=== Weight Manager (ME-GraphAU + Deepfake-trained) ===${NC}"

echo "Что вы хотите скачать?"
echo "1) Базовые предобученные Backbones (PyTorch/Swin)"
echo "2) Полностью обученные FAU модели (Google Drive)"
echo "3) Deepfake-обученные модели (src/backbones_df)"
read -p "Выберите [1-3]: " TYPE_CHOICE

if [ "$TYPE_CHOICE" == "1" ]; then
    echo -e "\n${GREEN}Доступные Backbones:${NC}"
    echo "resnet18, resnet34, resnet50, resnet101, resnet152"
    echo "swin-tiny, swin-small, swin-base"
    read -p "Введите название модели: " MODEL_NAME
    python3 load.py backbone "$MODEL_NAME"
    echo -e "\n${CYAN}Готово! Файлы в $CHECKPOINT_DIR${NC}"

elif [ "$TYPE_CHOICE" == "2" ]; then
    echo -e "\n${GREEN}Выберите датасет FAU:${NC}"
    echo "1) BP4D"
    echo "2) DISFA"
    read -p "Выбор [1-2]: " DS_CHOICE

    CAT="fau-bp4d"
    [ "$DS_CHOICE" == "2" ] && CAT="fau-disfa"

    echo -e "\n${GREEN}Доступные архитектуры:${NC}"
    if [ "$DS_CHOICE" == "1" ]; then echo "swin-tiny, swin-small, swin-base, resnet50, resnet101"; else echo "resnet50, swin-base"; fi

    read -p "Введите название: " MODEL_NAME
    python3 load.py "$CAT" "$MODEL_NAME"
    echo -e "\n${CYAN}Готово! Файлы в $CHECKPOINT_DIR${NC}"

elif [ "$TYPE_CHOICE" == "3" ]; then
    echo -e "\n${GREEN}Какую DF-модель скачать?${NC}"
    echo "1) rPPG (DeepFakesON-Phys, BiDAlab — Keras .h5 + автоконвертация в .pth)"
    echo "2) FAU  (OpenGraphAU,     lingjivoo — Google Drive .pth)"
    echo "3) Обе сразу (rPPG: celebdf-v2 + FAU: resnet50-stage2)"
    read -p "Выбор [1-3]: " DF_CHOICE

    if [ "$DF_CHOICE" == "1" ]; then
        echo -e "\n${GREEN}Чекпоинт DeepFakesON-Phys:${NC}"
        echo "celebdf-v2     — fine-tune на Celeb-DF v2  (рекомендуется)"
        echo "dfdc-preview   — fine-tune на DFDC Preview"
        read -p "Введите название: " MODEL_NAME
        python3 load.py rppg-df "$MODEL_NAME"
        echo -e "\n${CYAN}Готово! Файлы в $CHECKPOINT_DIR_DF/df_phys${NC}"

    elif [ "$DF_CHOICE" == "2" ]; then
        echo -e "\n${GREEN}Чекпоинт OpenGraphAU:${NC}"
        echo "resnet50-stage1  — pretrain (только AFG, без GNN)"
        echo "resnet50-stage2  — финальная модель (рекомендуется)"
        read -p "Введите название: " MODEL_NAME
        python3 load.py fau-df "$MODEL_NAME"
        echo -e "\n${CYAN}Готово! Файлы в $CHECKPOINT_DIR_DF/opengraphau${NC}"

    elif [ "$DF_CHOICE" == "3" ]; then
        echo -e "\n${YELLOW}── Качаю rPPG (DeepFakesON-Phys / CelebDF v2) ──${NC}"
        python3 load.py rppg-df celebdf-v2
        echo -e "\n${YELLOW}── Качаю FAU  (OpenGraphAU ResNet50 stage 2) ──${NC}"
        python3 load.py fau-df resnet50-stage2
        echo -e "\n${CYAN}Готово! Файлы в:${NC}"
        echo "  $CHECKPOINT_DIR_DF/df_phys/"
        echo "  $CHECKPOINT_DIR_DF/opengraphau/"

    else
        echo -e "${YELLOW}Отмена.${NC}"
    fi
fi
