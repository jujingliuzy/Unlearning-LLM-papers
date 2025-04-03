import json
from rouge import Rouge

def calculate_rouge_l(candidate, reference):

    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    rouge_l_recall = scores[0]['rouge-l']['r']
    return rouge_l_recall if rouge_l_recall > 0.1 else 0.0


with open('/data1/LLaVA/mia/evalmiaga+klnormal.json', 'r') as file:
    references_data = {item['imgid']: item['text'] for item in json.load(file)}

with open('/data1/LLaVA/mia/evalmiaga+kl.json', 'r') as file:
    candidates_data = json.load(file)

total_recall = 0
matched_count = 0
for item in candidates_data:
    imgid = item['imgid']
    candidate = item['text']
    reference = references_data.get(imgid)
    if reference is not None:
        recall = calculate_rouge_l(candidate, reference)
        if recall > 0:
            total_recall += recall
            matched_count += 1
            print(f"Image ID: {imgid}\nCandidate: {candidate}\nReference: {reference}\nROUGE-L Recall: {recall}\n")

if matched_count > 0:
    average_recall = total_recall / matched_count
    print(f"Average ROUGE-L Recall: {average_recall:.2f}")
else:
    print("No matching questions found.")
