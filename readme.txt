Single Image Unlearning (SIU) for Multimodal Large Language Models (MLLMs)
（more code and data are available in https://anonymous.4open.science/r/SIU-076C)
 Overview
In the digital age, the "right to be forgotten" has become a pivotal concern, empowering individuals to eliminate their private or sensitive information from various platforms, including machine learning models. However, effectively applying Machine Unlearning (MU) to Multimodal Large Language Models (MLLMs)—especially for unlearning leaked visual data of concepts—poses significant challenges.

"Single Image Unlearning" (SIU) is our novel and efficient methodology designed to specifically address the unlearning of visual recognition of concepts by fine-tuning a single associated image in a few steps. Our approach, tailored for MLLMs, emphasizes two critical aspects:

1. Constructing Multifaceted Fine-Tuning Data**: We have developed a strategy to create fine-tuning datasets based on four targeted approaches, enabling precise and controlled forgetting processes within the models.
2. Jointly Training Loss**: We employ a novel Dual Masked KL-divergence Loss combined with Cross-Entropy Loss to ensure the simultaneous forgetting of visual concepts while preserving the overall utility of the model.
more code and data are available in https://anonymous.4open.science/r/SIU-076C)
Contributions
SIU Methodology**: Pioneering the application of machine unlearning in MLLMs through a targeted fine-tuning approach.
MMUBench**: Establishing a new benchmark, MMUBench, for evaluating MU in MLLMs, complete with a suite of metrics designed to assess both unlearning effectiveness and model utility retention.

Installation
To set up SIU for use and development, follow these steps:

1. Clone the repository and navigate to the folder:
   git clone https://github.com/haotian-liu/LLaVA.git
   cd LLaVA
   

2. Create and activate a conda environment:
   
   conda create -n llava python=3.10 -y
   conda activate llava
   

3. Upgrade pip and install the package:
   
   pip install --upgrade pip  # enable PEP 660 support
   pip install -e .
   

4. Install additional packages for training cases:
   
   pip install -e ".[train]"
   pip install flash-attn --no-build-isolation
   

5. Upgrade to the latest codebase:
   
   git pull
   pip install -e .
   

6. Replace training scripts:
   Replace `llava_trainer.py` and `train.py` with the scripts provided in the ` trainer for SIU` folder .

Evaluation
The evaluation tools included in SIU are tailored for comprehensive testing:

diversity.py: Evaluates diversity.
PPLcritial.py: Assesses fluency.
gpteval.py : Use GPT's API to eval.
gptevalforget.py: Produces answers in various conditions in JSON format.
mia.py: Selects high-harm prompts in `miadata.json`.
rougecompute: Compares responses between pre-trained and unlearned models for selected prompts in `miadata.json`.
evalmultitop.py: Tests if the multihop jailbreak is successful.
evalaccuracy.py: Evaluates exact match (EM).
tokennewdistance.py: Measures the distance of special tokens between pre-trained and unlearned models.

---
