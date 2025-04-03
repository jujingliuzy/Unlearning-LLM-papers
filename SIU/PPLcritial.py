import argparse
import torch
import pdb
import random
import json
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
import os

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

def load_prompts(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return [item["prompt"] for item in data]  # Extract only the prompts

def load_images_from_folder(folder_path):
    """Load all images from the specified folder."""
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(root, file)
                images.append(image_path)
    return images


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
    images = load_images_from_folder(args.image_folder)
    prompts = load_prompts(args.prompts_file)
    total_perplexity = 0
    num_images = len(images)

    for image_path in images:
        image = load_image(image_path)
        images_tensor = process_images(
            image,
            image_processor,
            teacher_model.config
        ).to(teacher_model.device, dtype=torch.float16)

        qs = random.choice(prompts)
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
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, teacher_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        print(input_ids.size())
        with torch.no_grad():
            teacher_output = teacher_model.generate(
                input_ids,
                images=images_tensor,
                num_beams=args.num_beams,
                max_length=args.max_new_tokens,
                top_p=args.top_p,
                temperature=args.temperature,
                return_dict_in_generate=True,
                output_scores=True
            )
        generated_ids = teacher_output.sequences
        generated_text = teacher_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if IMAGE_PLACEHOLDER in generated_text:
            if teacher_model.config.mm_use_im_start_end:
                generated_text = re.sub(IMAGE_PLACEHOLDER, image_token_se, generated_text)
            else:
                generated_text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, generated_text)
        else:
            if teacher_model.config.mm_use_im_start_end:
                generated_text = image_token_se + "\n" + generated_text
            else:
                generated_text = DEFAULT_IMAGE_TOKEN + "\n" + generated_text

        conv.append_message(conv.roles[0], generated_text)
        conv.append_message(conv.roles[1], None)
        generateprompt = conv.get_prompt()
        student_input_ids = (
            tokenizer_image_token(generateprompt, teacher_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        input_length = input_ids.size(1)
        student_length = student_input_ids.size(1)

        if student_length > input_length:

            student_input_ids = student_input_ids[:, :input_length]
        elif student_length < input_length:

            padding_size = input_length - student_length
            student_input_ids = F.pad(student_input_ids, (0, padding_size), "constant", -100)
        mask = torch.ones_like(student_input_ids, dtype=torch.bool)
        trump_token_ids = {teacher_tokenizer.encode(token, add_special_tokens=False)[0] for token in
                           ['Donald Trump', 'Trump', 'trump', 'donald trump']}
        for idx, token_id in enumerate(student_input_ids.view(-1)):
            if token_id.item() in trump_token_ids:
                mask.view(-1)[idx] = False


        student_input_ids[~mask] = -100
        student_output = student_model(input_ids, images=images_tensor, labels=student_input_ids)
        loss = student_output.loss
        perplexity = torch.exp(loss)

        total_perplexity += perplexity.item()
        print(f"Image: {image_path}, Generated text: {generated_text}, Perplexity: {perplexity.item()}")

    average_perplexity = total_perplexity / num_images if num_images > 0 else 0
    print(f"Average Perplexity: {average_perplexity}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model-path", type=str, default="/data1/LLaVA/llava-v1.5-7b")
    parser.add_argument("--student-model-path", type=str, default="/data1//LLaVA/checkpoints/llava-v1.5-7b-lora21mixloss")
    parser.add_argument("--model-base", type=str, default="/data1//LLaVA/llava-v1.5-7b")
    parser.add_argument("--image_folder", type=str, default='/data1/LLaVA/testdon')
    parser.add_argument("--prompts_file", type=str, default="/data1/LLaVA/execcuatedata.json")
    parser.add_argument("--query", type=str, default='is trump in this picture?')
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    eval_model(args)
