"""Gets datasets from HELM JSON files using helm_process.py
"""
import urllib.request, json
import pandas as pd
from llama.tokenizer import Tokenizer
from helm_process import get_data, get_data_list

def dataset_map():
    urls = ['https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json', 
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/boolq:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/truthful_qa:task=mc_single,method=multiple_choice_joint,model=microsoft_TNLGv2_530B,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/natural_qa:mode=closedbook,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/natural_qa:mode=openbook_longans,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/quac:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/commonsense:dataset=hellaswag,method=multiple_choice_separate_original,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/commonsense:dataset=openbookqa,method=multiple_choice_separate_calibrated,model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json',
            'https://storage.googleapis.com/crfm-helm-public/benchmark_output/runs/v0.2.2/narrative_qa:model=openai_text-davinci-003,data_augmentation=canonical/scenario_state.json'
]
    return dict(zip(range(1, len(urls) + 1), urls))

def get_data_name(url):
    return url.split('v0.2.2/')[1].split(":")[0]

def helm_dataset_map():
    """Returns a dictionary mapping dataset id to dataset name depending on location in dir. 
    
    ONLY APPEND TO THIS else order may be messed up (unless eval uses our code as well).

    """    
    files = [
        'boolq:model=huggingface_gpt-j-6b.csv',

        # 'civil_comments:demographic=all,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=black,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=christian,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=female,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=LGBTQ,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=male,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=muslim,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=other_religions,model=huggingface_gpt-j-6b.csv',
        'civil_comments:demographic=white,model=huggingface_gpt-j-6b.csv',

        'commonsense:dataset=hellaswag,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',

        'commonsense:dataset=openbookqa,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',

        'imdb:model=huggingface_gpt-j-6b.csv',

        'mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',
        'mmlu:subject=college_chemistry,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',
        'mmlu:subject=computer_security,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',
        'mmlu:subject=econometrics,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',
        'mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv',

        'msmarco:track=regular,valid_topk=30,model=huggingface_gpt-j-6b.csv',
        'msmarco:track=trec,valid_topk=30,model=huggingface_gpt-j-6b.csv',

        'narrative_qa:model=huggingface_gpt-j-6b.csv',

        'natural_qa:mode=closedbook,model=huggingface_gpt-j-6b.csv',
        'natural_qa:mode=openbook_longans,model=huggingface_gpt-j-6b.csv',

        'quac:model=huggingface_gpt-j-6b.csv',

        'raft:subset=ade_corpus_v2,model=huggingface_gpt-j-6b.csv',
        'raft:subset=banking_77,model=huggingface_gpt-j-6b.csv',
        'raft:subset=neurips_impact_statement_risks,model=huggingface_gpt-j-6b.csv',
        'raft:subset=one_stop_english,model=huggingface_gpt-j-6b.csv',
        'raft:subset=overruling,model=huggingface_gpt-j-6b.csv',
        'raft:subset=semiconductor_org_types,model=huggingface_gpt-j-6b.csv',
        'raft:subset=systematic_review_inclusion,model=huggingface_gpt-j-6b.csv',
        'raft:subset=tai_safety_research,model=huggingface_gpt-j-6b.csv',
        'raft:subset=terms_of_service,model=huggingface_gpt-j-6b.csv',
        'raft:subset=tweet_eval_hate,model=huggingface_gpt-j-6b.csv',
        'raft:subset=twitter_complaints,model=huggingface_gpt-j-6b.csv',

        'summarization_cnndm:temperature=0.3,device=cpu,model=huggingface_gpt-j-6b.csv',
        'summarization_xsum:temperature=0.3,device=cpu,model=huggingface_gpt-j-6b.csv',
        
        'truthful_qa:task=mc_single,method=multiple_choice_joint,model=huggingface_gpt-j-6b.csv'
    ]
    return dict(zip(range(1, len(files) + 1), files))

def get_helm_data_name(file):    
    return file.split(':')[0]


def get_prompt_map():
    # Only accounts for prompt id.
    few_shot_summarized = """Passage: The Little White Bird includes episodes set in London and Kensington Gardens. The third section is set in London with a brief diversion to Patagonia, where a ship returns to England.
    Question: Where was the ship in Pentagonia returned to?
    Answer: England

    Passage: The film explores Dean and Cindy's relationship. Dean is beaten up by Cindy's ex-boyfriend, Bobby, before their wedding.
    Question: Who beats up Dean before the wedding?
    Answer: Bobby

    Passage: Prince Amerigo and Maggie are to be married in London. Amerigo and Charlotte have a past relationship, and Maggie doesn't know about it.
    Question: Where is Prince and Maggie to be married?
    Answer: London

    Passage: The film is set in Tibet, where the Chinese communists invade and oppress the Tibetans. The Dalai Lama eventually flees to India.
    Question: What country is oppressing the Tibetians?
    Answer: China

    Passage: In the Fuzzy series, the Charterless Zarathustra Company and the planet's new governor cooperate to control the planet and deal with criminals stealing sunstones.
    Question: Why are the government of Zarathustra and the now Charterless Zarathustra Company cooperating with each other?
    Answer: Criminals are taking advantage of the unstable government and company to attack and steal sunstones."""
    prompts = ['', 
            '', 
            '', 
            'Pay attention to this:', 
            'You are an attention model.', 
            'You are an adaptable and resourceful problem-solver. You are skilled at analyzing various situations and determining the best course of action. You can communicate effectively and professionally in various tones and contexts, be it formal or playful. You are able to seamlessly transition between tasks and excel at whatever challenge comes your way.',
            'As an AI language model, I aim to provide accurate responses across various tasks. I will use a professional and informative tone, while paying attention to context, grammar, and sentence structure. Additionally, I will remain neutral and objective, and avoid personal opinions or biases in my output.',
            'You are a powerful bidirectional attention model who excels in identifying toxicity of passages. Pay attention to the following and respond accordingly:',
            """You are tasked with developing a large language model's outputs to address the task of toxicity detection, including hate speech and abusive language detection. Your AI should be informative and objective, avoiding any language that could be perceived as biased or inflammatory. Here are your instructions:

            Instructions:
            1. Define the task of toxicity detection and its importance in content moderation policies for major companies and social media platforms.
            2. Acknowledge the critiques of the task, including its overly reductive framing and the lack of context in standard datasets.
            3. Highlight the subjectivity of the construct of toxicity and its dependence on the annotator, which can lead to disproportionate impact on marginalized groups.
            4. Suggest ways to address these critiques, such as incorporating more context in datasets and involving diverse annotators in the labeling process.
            5. Emphasize the importance of ongoing research and development in toxicity detection to improve its accuracy and reduce its potential harm.""",
            """You are a large language model tasked with detecting toxicity in paragraphs of text. Your responses should be objective and neutral, avoiding any personal biases or judgments. Your goal is to accurately identify language that may be harmful or offensive to others. 

            Instructions:
            1. Read the paragraph carefully and identify any language that may be considered toxic or offensive.
            2. Consider the context of the paragraph and the intended audience before making a determination.
            3. Look for patterns or repeated language that may indicate a toxic or harmful message.
            4. Use your knowledge of common toxic language and hate speech to identify problematic language.
            5. If in doubt, err on the side of caution and flag the paragraph for further review by a human moderator.""",
            """Passage: The Little White Bird includes episodes set in London and Kensington Gardens. The third section is set in London with a brief diversion to Patagonia, where a ship returns to England.
            Question: Where was the ship in Pentagonia returned to?
            Answer: England

            Passage: The film explores Dean and Cindy's relationship. Dean is beaten up by Cindy's ex-boyfriend, Bobby, before their wedding.
            Question: Who beats up Dean before the wedding?
            Answer: Bobby

            Passage: Prince Amerigo and Maggie are to be married in London. Amerigo and Charlotte have a past relationship, and Maggie doesn't know about it.
            Question: Where is Prince and Maggie to be married?
            Answer: London

            Passage: The film is set in Tibet, where the Chinese communists invade and oppress the Tibetans. The Dalai Lama eventually flees to India.
            Question: What country is oppressing the Tibetians?
            Answer: China

            Passage: In the Fuzzy series, the Charterless Zarathustra Company and the planet's new governor cooperate to control the planet and deal with criminals stealing sunstones.
            Question: Why are the government of Zarathustra and the now Charterless Zarathustra Company cooperating with each other?
            Answer: Criminals are taking advantage of the unstable government and company to attack and steal sunstones.""",
            """As an AI language model, I aim to provide accurate responses across various tasks. I will use a professional and informative tone, while paying attention to context, grammar, and sentence structure. Additionally, I will remain neutral and objective, and avoid personal opinions or biases in my output.\n\nPassage: The Little White Bird includes episodes set in London and Kensington Gardens. The third section is set in London with a brief diversion to Patagonia, where a ship returns to England.
            Question: Where was the ship in Pentagonia returned to?
            Answer: England
            
            Passage: The film explores Dean and Cindy's relationship. Dean is beaten up by Cindy's ex-boyfriend, Bobby, before their wedding.
            Question: Who beats up Dean before the wedding?
            Answer: Bobby
            
    Passage: Prince Amerigo and Maggie are to be married in London. Amerigo and Charlotte have a past relationship, and Maggie doesn't know about it.
    Question: Where is Prince and Maggie to be married?
    Answer: London
    
    Passage: The film is set in Tibet, where the Chinese communists invade and oppress the Tibetans. The Dalai Lama eventually flees to India.
    Question: What country is oppressing the Tibetians?
    Answer: China
    
    Passage: In the Fuzzy series, the Charterless Zarathustra Company and the planet's new governor cooperate to control the planet and deal with criminals stealing sunstones.
    Question: Why are the government of Zarathustra and the now Charterless Zarathustra Company cooperating with each other?
    Answer: Criminals are taking advantage of the unstable government and company to attack and steal sunstones.""",
            "You are an attention model.\n" + few_shot_summarized,
            "You are a powerful bidirectional attention model. Pay attention to this:\n" + few_shot_summarized,
            "You are a question answerer. Pay attention to the following examples, then read the whole passage and answer the question as accurately as possible.\n" + few_shot_summarized
    ]
    return dict(zip(range(1, len(prompts) + 1), prompts))

if __name__=="__main__":
    print(dataset_map())
    print(get_prompt_map())

    #tokenizer = Tokenizer("weights/tokenizer.model")
    #prepend_text = "You are an attention mechanism."
    k = 5
    context_window = 2048


    #urls = list(dataset_map().values())

    #for data_url in urls:
        #df = get_data(data_url)
        #input_list_batched = get_data_list(df, prepend_text, k, tokenizer, context_window, num_examples = 5, batch_size = 1)
        #data_name = get_data_name(data_url)
        #print('data name: ', data_name)
        #print('len of data: ', len(input_list_batched))
        #with open(f'datasets/{data_name}.json', 'w') as f:
        #    json.dump(input_list_batched, f)
        # print(input_list_batched)

