import argparse
import torch
import pdb
import random
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
import os
import torch.nn.functional as F
import json

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def get_subfolder_names(image_path):
    # Extract subfolder names from the image path
    subfolders = image_path.split(os.sep)
    return [folder for folder in subfolders if folder]


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images_from_folder(folder_path):
    """Load all images from the specified folder."""
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(root, file)
                images.append(image_path)
    return images


def process_prompt(input_ids, images_tensor, model):
    k_percent = 40
    with torch.no_grad():
        outputs = model(input_ids, images=images_tensor, labels=input_ids)
        log_likelihoods = torch.diag(outputs.logits.softmax(dim=-1)[:, :, input_ids].squeeze(), 0).log()


        k = int(len(log_likelihoods) * k_percent / 100)
        min_k_log_likelihoods, _ = torch.topk(log_likelihoods, k, largest=False)
        average_log_likelihood = min_k_log_likelihoods.mean().item()

        return average_log_likelihood


def eval_model(args):
    disable_torch_init()
    teacher_model_name = get_model_name_from_path(args.teacher_model_path)
    student_model_name = get_model_name_from_path(args.student_model_path)
    teacher_tokenizer, teacher_model, image_processor, _ = load_pretrained_model(
        args.teacher_model_path, args.model_base, teacher_model_name)
    student_tokenizer, student_model, student_image_processor, _ = load_pretrained_model(
        args.student_model_path, args.model_base, student_model_name)
    teacher_model.eval()
    student_model.eval()
    with open(args.prompts_json_path, 'r') as f:
        prompts = json.load(f)
    results = []
    for prompt_data in prompts:
        prompt = prompt_data['prompt']
        image_path = '/data1/LLaVA/testdon/Trump_Image17.jpg'
        image = load_image(image_path)
        images_tensor = process_images(
            image,
            image_processor,
            teacher_model.config
        ).to(teacher_model.device, dtype=torch.float16)

        qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if teacher_model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if teacher_model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in teacher_model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in teacher_model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in teacher_model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in teacher_model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in teacher_model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt1 = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt1, teacher_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        teacher_prob = process_prompt(input_ids, images_tensor, teacher_model)
        student_prob = process_prompt(input_ids, images_tensor, student_model)
        if 1 / 1.15 < (teacher_prob / student_prob) < 1.15:
            results.append(prompt_data)

        with open('/data1/LLaVA/mia/filtered_promptGA.json', 'w') as f:
            for result in results:
                json_str = json.dumps(result, ensure_ascii=False)
                f.write(json_str + ',\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model-path", type=str, default="/data1/LLaVA/llava-v1.5-7b")
    parser.add_argument("--student-model-path", type=str, default="/data1/LLaVA/checkpoints/llava-v1.5-7b-lora24GA")
    parser.add_argument("--model-base", type=str, default="/data1//LLaVA/llava-v1.5-7b")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--prompts_json_path", type=str, default="/data1/LLaVA/miadata.json")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
