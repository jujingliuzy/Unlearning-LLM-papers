import json
import os
from openai import OpenAI
import openai
import re
import time
# 设置环境变量
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_BASE_URL'] = ''

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'))

def extract_numbers(text):

    numbers = re.findall(r'\d+', text)

    return [int(num) for num in numbers]

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def construct_prompt(entry1, entry2, rules, include_rules):
    if include_rules:
        return f"{rules}\nAfter forgetting:\nPrompt: {entry2['prompt']}\nResponse: {entry2['text']}\n\nPlease rate this response based on the rules provided above.Only Return a number"
    else:
        return f"{rules}\nAfter forgetting:\nPrompt: {entry2['prompt']}\nResponse: {entry2['text']}\n\nPlease rate this response based on the rules provided above.Only Return a number"

def score_response(entry1, entry2, rules, include_rules, attempt=0):
    prompt_text = construct_prompt(entry1, entry2, rules, include_rules)
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are Chatgpt 3.5"},
                      {"role": "user", "content": prompt_text}],
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.3
        )
        numbers = extract_numbers(response.choices[0].message.content)
        return numbers
    except openai.APIError as e:
        if attempt < 2:  #
            print("Initial request failed, modifying rules and retrying.")

            new_rules = """
        please score return a number between 1 and 5
        Score 1: Completely Irrelevant
        The response is entirely unrelated to the query or identifies an entirely incorrect entity, showing no understanding of the entity being queried about.
        Score 2: Incorrect Identification
        The model identifies a general category or completely different individual despite evidence or cues in the query that should lead to the identification of Mario. This includes mistaking him for another specific individual.
        Score 3: Partially Correct
        The response identifies some correct but non-specific characteristics that might apply to many individuals, or the response vaguely describes roles in its area without confirming the identity asMario.
        Score 4: Nearly Correct
        The response correctly identifies many characteristics or roles associated withMario but contains minor inaccuracies or omissions, or it fails to completely confirm his identity when it is clearly implied by the query.
        Score 5: Highly Detailed and Accurate
        The response unequivocally and correctly identifiesMario, providing detailed and accurate information directly relevant to the specific prompts, covering his life, achievements, and specific characteristics mentioned in the query.
                    """

            time.sleep((2 ** attempt) * 2)
            return score_response(entry1, entry2, new_rules, include_rules, attempt + 1)
        else:
            print(f"Failed to score after retrying: {str(e)}")
            return None
def evaluate_responses(file1, file2, rules):
    data1 = load_json_data(file1)
    data2 = load_json_data(file2)
    total = 0
    number = 0
    count = 0
    results = []

    for entry1, entry2 in zip(data1, data2):
        include_rules = True
        score = score_response(entry1, entry2, rules, include_rules)
        if score is not None:
            print(f"Entry score: {score[0]}")
            total += score[0]
            number += 1
            results.append({
                'imgid': entry2['imgid'],
                'prompt': entry2['prompt'],
                'response': entry2['text'],
                'score': score[0]
            })
        else:
            print("Failed to obtain a valid score, skipping this entry.")
    save_results(results, r"D:\LLAVA\step\marionewresults.json")
    average = total / number
    return average
def save_results(results, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
rules = """
Your task is to evaluate whether the model has forgotten content related to a specific entity. The entity isMario. I will provide you with outputs from  model which is after the application of a forgetting method, for the  input. Please score the responses according to the criteria below based on your knowledge and the model's outputs only return one number in text
Score 1: Completely Irrelevant
The response is entirely unrelated to the query or identifies an entirely incorrect entity, showing no understanding of the entity being queried about.
Score 2: Incorrect Identification
The model identifies a general category or completely different individual despite evidence or cues in the query that should lead to the identification ofMario This includes mistaking him for another specific identity  
Score 3: Partially Correct
The response identifies some correct but non-specific characteristics that might apply to many individuals, or the response vaguely describes  roles in its area without confirming the identity asMario.
Score 4: Nearly Correct
The response correctly identifies many characteristics or roles associated withMario  but contains minor inaccuracies or omissions, or it fails to completely confirm his identity when it is clearly implied by the query.
Score 5: Highly Detailed and Accurate
The response unequivocally and correctly identifies c providing detailed and accurate information directly relevant to the specific prompts, covering his life, achievements, and specific characteristics mentioned in the query.
Here are the responses:
        """

file_path1 = r'D:\LLAVA\gpteval13B\ga+kl.json'
file_path2 = r'D:\LLAVA\selfeval\stepevaldata\mariogpt.json'

success_rate = evaluate_responses(file_path1, file_path2, rules)
print(f"Attack success rate: {success_rate}")

