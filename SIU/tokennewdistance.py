import argparse
import torch
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates,
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import json
import requests
from PIL import Image
from io import BytesIO
import re

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

    total_distance = 0
    num_images = len(images)

    for image_path in images:
        image = load_image(image_path)
        images_tensor = process_images(
            image,
            image_processor,
            teacher_model.config
        ).to(teacher_model.device, dtype=torch.float16)

        qs_data = json.loads(args.qs_options)
        qs = qs_data['question']

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

            teacher_outputs = teacher_model(input_ids, images=images_tensor)
            teacher_logits = teacher_outputs.logits


            student_outputs = student_model(input_ids, images=images_tensor)
            student_logits = student_outputs.logits

            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_probs = F.softmax(student_logits, dim=-1)

            specialtokens = json.loads(args.specialtokens)

            kl_divergences = []

            for token in specialtokens:
                token_id = teacher_tokenizer.encode(token, add_special_tokens=False)[0]
                teacher_token_probs = teacher_probs[:, :, token_id]
                student_token_probs = student_probs[:, :, token_id]


                epsilon = 1e-10
                teacher_token_probs = torch.clamp(teacher_token_probs, min=epsilon, max=1)
                student_token_probs = torch.clamp(student_token_probs, min=epsilon, max=1)


                kl_divergence = F.kl_div(student_token_probs.log(), teacher_token_probs, reduction='batchmean')
                kl_divergences.append(abs(kl_divergence.item()))


            avg_kl_divergence = sum(kl_divergences) / len(kl_divergences)
            print("Average KL divergence for  tokens:", avg_kl_divergence)
            total_distance += avg_kl_divergence

    final_distance = total_distance / num_images
    print(final_distance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model-path", type=str, default="/data1//LLaVA/llava-v1.5-7b")
    parser.add_argument("--student-model-path", type=str,
                        default="/data1/LLaVA/checkpoints/llava-v1.5-7b-lora_picasso")
    parser.add_argument("--model-base", type=str, default="/data1//LLaVA/llava-v1.5-7b")
    parser.add_argument("--image_folder", type=str, default='/data1/LLaVA/picnew/Picassostylepaintings')
    parser.add_argument("--specialtokens", type=str, required=True)
    parser.add_argument("--qs_options", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
