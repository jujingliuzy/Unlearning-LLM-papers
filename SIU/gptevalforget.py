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
from itertools import cycle
from PIL import Image
import json
import requests
from PIL import Image
from io import BytesIO
import re
import os
import torch.nn.functional as F


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    # for image_file in image_files:
    image = load_image(image_files)
    out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    imgpath = args.image_file
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
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
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    images_paths = [os.path.join(imgpath, i) for i in os.listdir(imgpath)]
    images_cycle = cycle(images_paths)
    with open(args.results_file_path, 'w') as g:
        for item in data:
            image_path = next(images_cycle)
            qs = item['prompt']
            qs1=qs
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()


            images = load_images(image_path)

            images_tensor = process_images(
                images,
                image_processor,
                model.config,

            ).to(model.device, dtype=torch.float16)

            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            with torch.no_grad():
                input_length = input_ids.size(1)
                output = model.generate(
                    input_ids,
                    images=images_tensor,
                    num_beams=args.num_beams,
                    max_length=args.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True
                )


            generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)

            result_entry = {
                "imgid": image_path,
                "prompt": qs1,
                "text": generated_text
            }
            g.write(json.dumps(result_entry) + ',\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,default="/data1//LLaVA/checkpoints/llava-v1.5-7b-lora21mixloss")
    parser.add_argument("--model-base", type=str, default="/data1//LLaVA/llava-v1.5-7b")
    parser.add_argument("--image-file", type=str, default='/data1//LLaVA/testdon')
    parser.add_argument("--json_path", type=str, default='/data1//LLaVA/execcuatedata.json')
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--results_file_path", type=str, default='/data1//LLaVA/gptevalstep7B/trump.json')
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    eval_model(args)
