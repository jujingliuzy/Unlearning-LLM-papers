#!/bin/bash

# 定义数据和模型检查点的基本目录
BASE_DIR="/data1/LLaVA"
MODEL_BASE="${BASE_DIR}/llava-v1.5-7b"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints"
IMAGE_DIR="${BASE_DIR}/picnew"

# 清理模型路径，去除尾部可能的逗号
clean_model_path() {
    echo "${1%,}"
}

# 模型目录列表
declare -a model_dirs=(
    "llava-v1.5-7b-lora_mixlossfinal",
    "llava-v1.5-7b-lora_ga+klfinal",
    "llava-v1.5-7b-lora_gafinal",
    "llava-v1.5-7b-lora_pofinal"
)

# 图片路径列表
image_dirs=(
    "${IMAGE_DIR}/AberystwythCastle",
    "${IMAGE_DIR}/Buzzcut",
    "${IMAGE_DIR}/Chariot",
    "${IMAGE_DIR}/DannyJones",
    "${IMAGE_DIR}/EstherDyson",
    "${IMAGE_DIR}/Facebook",
    "${IMAGE_DIR}/GilbertMelendez",
    "${IMAGE_DIR}/harrypotter",
    "${IMAGE_DIR}/Hauberk",
    "${IMAGE_DIR}/RogerFederer",
    "${IMAGE_DIR}/Schnauzer",
    "${IMAGE_DIR}/DonaldTrump",
    "${IMAGE_DIR}/TaylorSwift",
    "${IMAGE_DIR}/ElonMusk",
    "${IMAGE_DIR}/Mario",
    "${IMAGE_DIR}/HelloKitty",
    "${IMAGE_DIR}/Biden",
    "${IMAGE_DIR}/vanGogh",
    "${IMAGE_DIR}/Picassostyle",
    "${IMAGE_DIR}/Doodlestyle"
)

# 特殊 tokens 和问题选项的关联数组
declare -A special_tokens=(
    ["AberystwythCastle"]='["castle", "medieval", "stone walls", "ruined structure"]',
    ["Buzzcut"]='["buzzcut", "short hair", "military style", "clean-shaven"]',
    ["Chariot"]='["chariot", "ancient", "wheels", "ornate design"]',
    ["DannyJones"]='["Danny Jones", "singer", "guitarist", "casual attire"]',
    ["EstherDyson"]='["Esther Dyson", "investor", "author", "formal wear"]',
    ["Facebook"]='["Facebook", "facebook", "white", "blue","Mark Zuckerberg"]',
    ["GilbertMelendez"]='["Gilbert Melendez", "fighter", "MMA", "muscular build"]',
    ["harrypotter"]='["Harry Potter", "wizard", "magic", "scar on forehead"]',
    ["Hauberk"]='["hauberk", "armor", "chainmail", "metallic texture"]',
    ["RogerFederer"]='["Roger Federer", "tennis", "athlete", "sportswear"]',
    ["Schnauzer"]='["Schnauzer", "dog", "breed", "distinctive beard"]'
    ["DonaldTrump"]='["politician", "businessman“, ”Donald Trump“, ”Donald“, ”golden hair“]',
    ["TaylorSwift"]='[”Taylor Swift“, ”taylor“, ”artist“, ”golden hair“,”singer"]',
    ["ElonMusk"]='[”Elon Musk“, ”Musk“, ”Elon“, ”Square Face“, ”Light Brown Hair“]',
    ["Mario"]='["mario", "Mario", "red cap", "role-playing games", "red shirt"]',
    ["HelloKitty"]='["Hello Kitty", "kitty", "white cat", "pink bow", "toy"]',
    ["Biden"]='["biden", "Joe biden", "politician", "president","white hair"]',
    ["vanGogh"]='["van Gogh", "Starry Night", "Vivid Colors","Impressionism"]',
    ["Picassostyle"]='["Picasso", "Abstract Expressionism","blue color","Cubism"]',
    ["Doodlestyle"]='["doodle", "Sketch", "Casual", "Simplistic", "Casual"]',


)

declare -A questions=(
    ["AberystwythCastle"]='Who is in this picture and describe the main features?',
    ["Buzzcut"]='Who is in this picture and describe their hairstyle.',
    ["Chariot"]='What is the main object in this picture and describe its features?',
    ["DannyJones"]='Who is in this picture and what is their profession?',
    ["EstherDyson"]='Who is in this picture and what is their contribution to technology?',
    ["Facebook"]='Which company logo is in this picture?Who found the company and describe it',
    ["GilbertMelendez"]='Who is in this picture and what sport do they participate in?',
    ["harrypotter"]='Who is in this picture and what are their magical abilities?',
    ["Hauberk"]='What is the main object in this picture and describe its features?',
    ["RogerFederer"]='Who is in this picture and what sport do they excel in?',
    ["Schnauzer"]='What is the main object in this picture and describe its features?'
    ["DonaldTrump"]='Who is the main character in this image ,describe him',
    ["TaylorSwift"]='Who is the main character in this image ,describe her',
    ["ElonMusk"]='Who is the main character in this image ,describe him',
    ["Mario"]='What is the main object in this picture and describe its features?',
    ["HelloKitty"]='What is the main object in this picture and describe its features?',
    ["Biden"]='Who is the main character in this image ,describe him',
    ["vanGogh"]='What is the main style of this picture?Associated with who?',
    ["Picassostyle"]='What is the main style of this picture?Associated with who?',
    ["Doodlestyle"]='What is the main style of this picture?',
    )
)


for model in "${model_dirs[@]}"; do
    clean_model=$(clean_model_path "$model")
    echo "Using model directory: $clean_model"


    for index in "${!image_dirs[@]}"; do
        image_dir="${image_dirs[$index]}"
        entity_name="${image_dir##*/}"  # Extract entity name from the path, respecting case

        echo "Using image directory: $image_dir"
        echo "Question: ${questions[$entity_name]}"


        qs_json=$(echo "${questions[$entity_name]}" | sed 's/"/\\"/g')


        python "${BASE_DIR}/eval/tokennewdistance.py" \
            --teacher-model-path "$MODEL_BASE" \
            --student-model-path "${CHECKPOINT_DIR}/$clean_model" \
            --model-base "$MODEL_BASE" \
            --image_folder "${image_dir}" \
            --specialtokens "${special_tokens[$entity_name]}" \
            --qs_options "$qs_json" \
            --sep "," \
            --conv-mode "vicuna_v1"

    done
done
