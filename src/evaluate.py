# from src.medrag import MedRAG

# question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
# options = {
#     "A": "paralysis of the facial muscles.",
#     "B": "paralysis of the facial muscles and loss of taste.",
#     "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
#     "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
# }

# ## CoT Prompting
# cot = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=False)
# answer, _, _ = cot.answer(question=question, options=options)

# ## MedRAG
# medrag = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks")
# answer, snippets, scores = medrag.answer(question=question, options=options, k=32) # scores are given by the retrieval system
# print("answer the " answer,snippets,scores)










# import os
# import json
# import requests
# from utils import RetrievalSystem
# from template import *

# class MedRAG:

#     def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus"):
#         self.retriever_name = retriever_name
#         self.corpus_name = corpus_name
#         self.db_dir = db_dir
#         self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir)

#     def answer(self, question, API_TOKEN, k=32, rrf_k=100, save_dir=None):
#         '''
#         question (str): question to be answered
#         API_TOKEN (str): Authentication token for API access
#         k (int): number of snippets to retrieve
#         save_dir (str): directory to save the results
#         '''

#         # retrieve relevant snippets
#         retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
#         contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
        
#         if len(contexts) == 0:
#             return None, None, None

        
        
#         # join contexts and trim to appropriate length
#         full_context = "\n".join(contexts)[:2048]  # adjust size as needed

#         # setup the API call
#         API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"
#         headers = {"Authorization": f"Bearer {API_TOKEN}"}
#         payload = {
#             "inputs": full_context,
#             "parameters": {"return_full_text": False}
#         }
#         response = requests.post(API_URL, headers=headers, json=payload)
#         answer = response.json()

#         if save_dir is not None:
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
#                 json.dump(retrieved_snippets, f, indent=4)
#             with open(os.path.join(save_dir, "response.json"), 'w') as f:
#                 json.dump(answer, f, indent=4)
        
#         return answer, retrieved_snippets, scores

# # Usage of the class:
# medrag = MedRAG()
# API_TOKEN = 'your_hugging_face_api_token_here'
# question = "What are the symptoms of the flu?"
# response, snippets, scores = medrag.answer(question, API_TOKEN)
# print(response)




class RetrievalSystem:

    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus"):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names
        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            self.retrievers.append([])
            for corpus in corpus_names[self.corpus_name]:
                self.retrievers[-1].append(Retriever(retriever, corpus, db_dir))
    
    def retrieve(self, questions, k=32, rrf_k=100):
        '''
            Given a list of questions, return the relevant snippets from the corpus for each question.
        '''
        assert isinstance(questions, list) and all(isinstance(q, str) for q in questions)

        batch_texts = []
        batch_scores = []

        for question in questions:
            texts = []
            scores = []

            if "RRF" in self.retriever_name:
                k_ = max(k * 2, 100)
            else:
                k_ = k

            for i in range(len(retriever_names[self.retriever_name])):
                texts.append([])
                scores.append([])
                for j in range(len(corpus_names[self.corpus_name])):
                    t, s = self.retrievers[i][j].get_relevant_documents(question, k=k_)
                    texts[-1].append(t)
                    scores[-1].append(s)

            texts, scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)
            batch_texts.append(texts)
            batch_scores.append(scores)

        return batch_texts, batch_scores








# ----------------MedRag Starting from here --------------------------------
class MedRAG:

 def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir)
        else:
            self.retrieval_system = None
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
            if "mixtral" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
                self.context_length = 3072
                self.templates["cot_prompt"] = meditron_cot
                self.templates["medrag_prompt"] = meditron_medrag
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
                self.context_length = 1024
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                torch_dtype=torch.float16,
                device_map="auto",
                model_kwargs={"cache_dir":self.cache_dir},
            )



    # [Include all previous definitions and methods as they are, except the answer method]

    def answer(self, questions, options=None, k=32, rrf_k=100, save_dir=None):
        '''
        questions (List[str]): list of questions to be answered
        options (List[Dict[str, str]] or None): list of dictionaries, each containing options to be chosen from for each question
        k (int): number of snippets to retrieve per question
        save_dir (str): directory to save the results
        '''

        batch_answers = []
        batch_retrieved_snippets = []
        batch_scores = []

        batch_texts, batch_scores = self.retrieval_system.retrieve(questions, k=k, rrf_k=rrf_k)

        for idx, question in enumerate(questions):
            if options is not None:
                question_options = '\n'.join([key + ". " + options[idx][key] for key in sorted(options[idx].keys())])
            else:
                question_options = ''

            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, batch_texts[idx][idx]["title"], batch_texts[idx][idx]["content"]) for idx in range(len(batch_texts[idx]))]
            if len(contexts) == 0:
                contexts = [""]
            if "openai" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            else:
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]

            # generate answers for each question using its context
            if not self.rag:
                prompt_cot = self.templates["cot_prompt"].render(question=question, options=question_options)
                messages = [
                    {"role": "system", "content": self.templates["cot_system"]},
                    {"role": "user", "content": prompt_cot}
                ]
                ans = self.generate(messages)
                batch_answers.append(re.sub("\s+", " ", ans))
            else:
                for context in contexts:
                    prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=question_options)
                    messages=[
                            {"role": "system", "content": self.templates["medrag_system"]},
                            {"role": "user", "content": prompt_medrag}
                    ]
                    ans = self.generate(messages)
                    batch_answers.append(re.sub("\s+", " ", ans))
        
        return batch_answers, batch_retrieved_snippets, batch_scores





# medrag_run.p

# Assuming corpus_names and retriever_names are defined or replace them with actual lists/dictionaries as required.
corpus_names = {'Textbooks': ['Corpus1', 'Corpus2']}  # Example corpus names
retriever_names = {'MedCPT': ['Retriever1', 'Retriever2']}  # Example retriever names

# Initialize the MedRAG system
medrag = MedRAG()

# Define a batch of questions
questions = [
    "What are the symptoms of COVID-19?",
    "How does the immune system respond to a viral infection?"
]

# Optionally define options for each question if needed
options = [
    {"A": "Fever", "B": "Cough", "C": "Muscle pain", "D": "All of the above"},
    None  # No options for the second question
]

# Get answers for the batch of questions
answers, retrieved_snippets, scores = medrag.answer(questions, options)

# Print the results
for question, answer in zip(questions, answers):
    print("Question:", question)
    print("Answer:", answer)
    print("---")
