from typing import List, Tuple
from dataclasses import dataclass
import logging
import string
import spacy
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput 
from transformers import PreTrainedModel 
from retriever import BM25
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEBUG = True
FLAG = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

@dataclass
class Block:
    text: str = None
    tokens: List[str] = None 
    range_: List[Tuple[int, int]] = None 
    @property
    def len_tokens(self):
        return len(self.tokens)
    @property
    def len_words(self):
        return len(self.range_)

# Merge tokens into words
def merge_blocks(blocks: List[Block]) -> Block:
    text = "".join([block.text for block in blocks])
    tokens = sum([block.tokens for block in blocks], [])
    range_ = []
    st = 0
    for block in blocks:
        if block.range_:
            for l, r in block.range_:
                range_.append((st+l, st+r))
            st = range_[-1][1]
    return Block(text=text, tokens=tokens, range_=range_)

class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }

@dataclass
class GeneratorOutput:
    ended: bool
    empty: bool
    blocks: List[Block] = None
    merged_blocks: Block = None
    atten: Tensor = None
    max_atten: Tensor = None
    entropies: Tensor = None
    entropies_s1: Tensor = None
    entropies_s2: Tensor = None
    smooth_s2: Tensor = None
    mt_s2: Tensor = None
    fun_word: Tensor = None
    @property
    def new_text(self):
        return self.blocks[-1].text
    @property
    def len_new_words(self):
        return self.blocks[-1].len_words
    
class Generator:
    def __init__(
        self,
        model_name_or_path: str
    ):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto") # device_map表示自动分到显卡上
        self.model: PreTrainedModel
        logger.info(f"device = {self.model.device}")
        # The space character is different in llama3 and llama2.
        self.space_token = "Ġ" if "llama-3" in model_name_or_path.lower() else "▁"
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.tokens_cannot_merged = {
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode("0" + ch)[-1:])[0]
            for ch in string.whitespace + string.punctuation
        } | {self.space_token, self.tokenizer.bos_token, self.tokenizer.eos_token}

    # The model regenerates based on the retrieved documents.
    def simply_generate(
        self,
        input_text: str,
        max_length: int
    ) -> Tuple[bool, str]:
        '''
        return ended, new_text
        '''
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device) # (batch_size=1, input_length)
        input_length = input_ids.shape[1]
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            stop_strings = "\n",
            tokenizer=self.tokenizer
        )[0, input_length:]
        if output_ids.shape[0] == 0:
            logger.info("generate '' in simply_generate()!")
            return True, ""
        if output_ids[0] == self.tokenizer.bos_token_id:
            output_ids = output_ids[1:]
        if output_ids[-1] == self.tokenizer.eos_token_id:
            return True, self.tokenizer.decode(output_ids[:-1])
        return False, self.tokenizer.decode(output_ids)

    def tokenize(
        self,        
        text: str,
        is_start: bool = False 
    ):
        ids = self.tokenizer.encode(text) # List[int]
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if not is_start and tokens[0] == self.tokenizer.bos_token:
            tokens = tokens[1:]
        return tokens
        
    def merge_tokens(
        self,
        tokens
    ) -> List[Tuple[int, int]]:
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) \
                or tokens[i] in self.tokens_cannot_merged \
                or tokens[i-1] in self.tokens_cannot_merged:
                range_.append([i, i+1])
            else:
                range_[-1][1] += 1
        return range_

    def build_block(
        self,        
        text: str,
        is_start: bool = False 
    ) -> Block:
        tokens = self.tokenize(text, is_start=is_start)
        range_ = self.merge_tokens(tokens)
        return Block(text=text, tokens=tokens, range_=range_)

    def generate(
        self,
        input_texts: List[str], 
        max_length: int,
    ) -> GeneratorOutput:
        blocks = []
        for text in input_texts:
            blocks.append(self.build_block(text, is_start=not blocks))

        input_tokens = sum([block.tokens for block in blocks], [])
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_tokens)], device=self.model.device)
        input_len_tokens = len(input_tokens)

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            return_dict_in_generate=True,
            output_scores=True, 
            stop_strings="\n",
            tokenizer=self.tokenizer,
        )
        outputs: GenerateDecoderOnlyOutput

        tokens = self.tokenizer.convert_ids_to_tokens(outputs.sequences[0, input_len_tokens:]) # List[str]
        print("len_tokens:",len(tokens))
        if (len(tokens)<=1):
            return GeneratorOutput(
                empty=True,
                ended=True,
                blocks=None,
                merged_blocks=None,
                atten=None,
                max_atten=None,
                entropies=None,
                entropies_s1=None, # First-order difference entropy
                entropies_s2=None, # Second-order difference entropy
                smooth_s2=None, # Second-order smoothed entropy
                fun_word=None,
            )
        ended = (tokens[-1] == self.tokenizer.eos_token)
        if ended:
            tokens = tokens[:-1]
        text = self.tokenizer.convert_tokens_to_string(tokens)
        range_ = self.merge_tokens(tokens)
        new_block = Block(text=text, tokens=tokens, range_=range_)

        blocks.append(new_block)
        merged_blocks = merge_blocks(blocks)

        # Merged attention
        atten = self.model(outputs.sequences, output_attentions=True).attentions[-1][0][:, -new_block.len_tokens:, :] # (num_heads, new_len_tokens, len_tokens) 
        atten = atten.mean(dim=0) 
        atten = torch.stack([atten[:, l:r].sum(dim=-1) for l, r in merged_blocks.range_], dim=-1) 
        atten = torch.stack([atten[l:r, :].mean(dim=-2) for l, r in range_], dim=-2)  

        atten_to_new = atten[:, -new_block.len_words:] 
        atten_to_new /= atten.sum(dim=-1,keepdim=True) + 1e-10 
        max_atten, _ = atten_to_new.max(dim=1)

        probs = torch.stack(outputs.scores).softmax(dim=-1) 
        entropies = (-probs * torch.log(probs + 1e-10)).sum(dim=-1) 

        entropies = torch.stack([entropies[l:r, 0].max() for l, r in range_])

        func_words=[]   
        doc = nlp(new_block.text)
        real_words = set(token.text for token in doc if token.pos_ in
                         ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        wl = 0
        wr = new_block.len_words
        for i in range(wl, wr):
            tl, tr = new_block.range_[i]
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[tl:tr])
            if not match(word, real_words):
                func_words.append(i)
                
        entropies_s1 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]  # First-order difference entropy
        entropies_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]   # Second-order difference entropy
        smooth_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))] # Smoothing second-order difference entropy
        mt_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]  # Dynamically smoothed second-order difference entropy
        fun_word = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]  # Content words
        for i, (l,r) in enumerate(range_[:]):
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[l:r])
            if i not in func_words:
                fun_word[i]['val'] = torch.tensor(1, dtype=torch.float64)
        for i, (l, r) in enumerate(range_[1:]): 
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[l:r])
            entropy = entropies[i+1].item()  
            if i+1 not in func_words:
                j = i
                while j >= 0:
                    if j not in func_words:
                        s1 = (entropies[i+1].to(torch.float64) - entropies[j].to(torch.float64))
                        entropies_s1[i+1]['val'] = s1
                        break
                    if j == 0:
                        break
                    else:
                        j -= 1
        for i, (l, r) in enumerate(range_[2:]): 
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[l:r])
            entropy = entropies[i+2].item() 
            if i+2 not in func_words: 
                j = i + 1
                while j >= 1:
                    if entropies_s1[j]['val'].item() != 0: 
                        s2 = (entropies_s1[i+2]['val'].to(torch.float64) - entropies_s1[j]['val'].to(torch.float64)) 
                        entropies_s2[i+2]['val'] = s2
                        break
                    if j == 1:
                        break
                    else:
                        j -= 1

        if len(range_) > 0:
            l, r = range_[0]
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[l:r])
            entropy = entropies[0].item()
            if 0 in func_words:
                print(f"word0: {word}", "function words")
            else:
                print(f"word0: {word}, entropy: {entropy}", "Content words")

        if len(range_) > 1:
            l, r = range_[1]
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[l:r])
            entropy = entropies[1].item()
            entropy_s1 = entropies_s1[1]['val'].item()
            if 1 in func_words:
                print(f"word1: {word}", "function words")
            elif 1 not in func_words and 0 in func_words:
                print(f"word1: {word}, entropy: {entropy}", "Content words")
            else:
                print(f"word1: {word}, entropy: {entropy}, First-order difference entropy: {entropy_s1}", "Content words")

        count_fun = 0
        sum_s2 = 0
        Mt_1 = torch.tensor(0, dtype=torch.float64) # Δ2H(t-1)
        for i, (l, r) in enumerate(range_[2:]):
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[l:r])
            if entropies_s2[i+2]['val'] != 0:
                count_fun +=1 
                sum_s2 += entropies_s2[i+2]['val'].item()
                s2_mean = sum_s2/count_fun # mean E
                w = torch.abs((Mt_1 - s2_mean)) /(torch.abs((entropies_s2[i+2]['val']-s2_mean)) + torch.abs((Mt_1 - s2_mean)))
                α = 0.9 + 0.1 * w
                Mt = α * entropies_s2[i+2]['val'] + (1-α) * Mt_1
                mt_s2[i+2]['val'] = Mt
                print(f"word{i+2}: {word}, entropy: {entropies[i+2].item()}, First-order difference entropy: {entropies_s1[i+2]['val'].item()}, Second-order difference entropy: {entropies_s2[i+2]['val'].item()}, Second difference entropy mean：{s2_mean}, Δ2Ht:{entropies_s2[i+2]['val']},Δ2H(t-1):{Mt_1}, Mt: {Mt}, w: {w} Content words")

                Mt_1 = entropies_s2[i+2]['val']
            elif entropies_s1[i+2]['val'].item() != 0:
                print(f"word{i+2}: {word}, entropy: {entropies[i+2].item()}, First-order difference entropy: {entropies_s1[i+2]['val'].item()}", "Content words")
            elif i+2 not in func_words:
                print(f"word{i+2}: {word}", "Content words")
            else:
                print(f"word{i+2}: {word}", "function words")
        return GeneratorOutput(
            empty = False,
            ended=ended,
            blocks=blocks,
            merged_blocks=merged_blocks,
            atten=atten,
            max_atten=max_atten,
            entropies=entropies,
            entropies_s1 = entropies_s1,
            entropies_s2 = entropies_s2,
            smooth_s2 = smooth_s2,
            mt_s2 = mt_s2,
            fun_word = fun_word,
        )

def join_if_nonempty(*li, sep=" "):
    return sep.join([s for s in li if len(s) > 0])

def match(word: str, real_words): 
    for real_word in real_words:
        if real_word in word: 
            return True
    return False

def get_top_sentence(text):
    prev = ""
    for sent in nlp(text).sents:
        prev += sent.text
        sent = sent.text.strip()
        if len(sent) > 0:
            return prev
    return ""

@dataclass
class CheckerOutput:
    
    hallucination: bool 
    curr_st: int = None  # The starting position of the hallucination sentence
    curr_en: int = None  # End of the hallucination sentence
    curr_thres: List[bool] = None

class ETC:
    def __init__(self, args):
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        self.generator = Generator(self.model_name_or_path)
        self.tokenizer = self.generator.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, device_map="auto") 
        self.model: PreTrainedModel
        self.retriever = BM25("wiki" if "es_index_name" not in args else self.es_index_name)
        self.counter = Counter()

    
    def hallucination_check(
        self,
        outputs: GeneratorOutput
    ) -> CheckerOutput: 
        if DEBUG:
            print("Start detecting hallucinations")
        new_block = outputs.blocks[-1]
        sentences = [sent.text.strip() for sent in nlp(new_block.text).sents]   
        sentences = [sent for sent in sentences if len(sent) > 0]  
        if DEBUG:
            print("Clauses")
            for i, sent in enumerate(sentences):
                print(f"sentence{i}：{sent}")
        wid = 0
        word_counts = [0] * len(sentences)

        thres_sum = [] # The illusion of storing all words
        for sid, sent in enumerate(sentences):  
            wl, wr = wid, wid # Start and end points of the current token range
            if wid == new_block.len_words:
                break
            while wr < new_block.len_words and sent not in self.tokenizer.convert_tokens_to_string(
                new_block.tokens[new_block.range_[wl][0]:new_block.range_[wr][1]] # 
            ):
                wr += 1 
            if wr < new_block.len_words:
                wr += 1 
            wid = wr    
            len_sent = wid
            if wl == wr:
                continue
            if sid == 0:    
                word_counts[sid] = wid
            else:
                for t in range(0,sid):
                    len_sent -= word_counts[t]
                word_counts[sid] = len_sent 
            print("Current sentence length:",word_counts[sid])
            index_sent = 0
            for j in range(0, sid): 
                index_sent += word_counts[j]
            if DEBUG:
                print("Current sentence:", self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[wl][0]:new_block.range_[wr-1][1]]), sep="\n")
            max_atten_sent = outputs.max_atten[wl: wr]
            max_atten_sent = max_atten_sent * (wr - wl) / (max_atten_sent.sum() + 1e-10)
            # Final Indicators
            value = max_atten_sent * torch.tensor([entry['val'] for entry in outputs.mt_s2[wl: wr]]).to(max_atten_sent.device) 
            thres_abs = self.thres_abs
            if thres_abs == True:
                thres = (torch.abs(value) > self.hallucination_threshold)
            else:
                thres = (value > self.hallucination_threshold)
            thres_sum.append(thres)
            if DEBUG:
                print("wid|word|max_atten_sent|entropy|entropies_s1|entropies_s2|mt_s2|value|thres：")
                for i in range(wl, wr):
                    print(i,
                          self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[i][0]:new_block.range_[i][1]]), 
                          max_atten_sent[i-wl].item(),
                          outputs.entropies[i-wl].item(), 
                          outputs.entropies_s1[i]['val'].item(), 
                          outputs.entropies_s2[i]['val'].item(), 
                          outputs.mt_s2[i]['val'].item(),
                          value[i-wl].item(),
                          thres[i-wl].item(), sep="|")
            if True in thres:
                for i in range(wl, wr):
                    if thres[i-wl].item() == True:
                        count_k_2 = 0
                        j = i - 1
                        while(count_k_2 < 2):
                            if outputs.fun_word[j]['val'].item() != 0:
                                count_k_2 += 1
                            if count_k_2 == 2:
                                break
                            else:
                                j -= 1
                        return CheckerOutput(hallucination=True, curr_st=i, curr_en=wr, curr_thres=thres[i-wl:wr]) 
                    
            if DEBUG:
                print("No hallucinations were detected in the current sentence. Prepare for the next sentence.")
        return CheckerOutput(hallucination=False)

    def generate_retrieve_qry(self, outputs: GeneratorOutput, check_info: CheckerOutput):
        ques_st = outputs.blocks[0].len_words + outputs.blocks[1].len_words  # The starting point of the question section
        ques_en = ques_st + outputs.blocks[2].len_words   # End of the question section

        question_words = []
        for i in range(ques_st, ques_en):
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
            question_words.append(word)

        print("question", " ".join(question_words))

        text_st = ques_en + outputs.blocks[3].len_words  # Starting position of the answer section
        text_en = text_st + outputs.blocks[4].len_words + check_info.curr_st # End of answer section

        ques_atten = outputs.atten[check_info.curr_st:check_info.curr_en, ques_st:ques_en]  # Attention matrix of the illusion part of the problem
        text_atten = outputs.atten[check_info.curr_st:check_info.curr_en, text_st:text_en]  # Attention matrix of the hallucination part to the previously generated text

        print("ques_atten.shape:",ques_atten.shape)
        print("text_atten.shape:",text_atten.shape)
        print(check_info.curr_thres.shape)
        
        ques_atten = ques_atten[check_info.curr_thres, :].sum(dim=0)
        text_atten = text_atten[check_info.curr_thres, :].sum(dim=0)

        doc = nlp(outputs.merged_blocks.text)
        real_words = set(token.text for token in doc if token.pos_ in 
            ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        real_pairs = []
        for i in range(ques_st, ques_en):
            a = ques_atten[i - ques_st]
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
            if match(word, real_words):
                real_pairs.append((a, word, i)) 
        for i in range(text_st, text_en):
            a = text_atten[i - text_st]
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
            if match(word, real_words):
                real_pairs.append((a, word, i)) 
        
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs.sort(key=lambda x: -x[0])   
        real_pairs = real_pairs[:top_k]    # Filter the top_k elements
        real_pairs.sort(key=lambda x: x[2])   

        return " ".join([x[1] for x in real_pairs]) 

    def inference(self, question, demo, case):
        text = ""
        demo = "\n".join([d["case"] for d in demo]) 
        if DEBUG:
            print("Begin reasoning")
        while True:
            old_len = len(text)
            outputs = self.generator.generate(
                input_texts=[demo, "\nQuestion:", question, "\nAnswer:", text], 
                max_length=self.generate_max_length,
            )
            # print("outputs:",outputs)
            if DEBUG:
                if outputs.empty==False :
                    print("Initial generation of new text", outputs.new_text, sep="\n")
                    if self.use_counter == True:
                        self.counter.add_generate(outputs.new_text, self.generator.tokenizer)
            if outputs.empty == True:
                if DEBUG:
                    print("If only blank characters are detected, the generation process will be interrupted.")
                break

            check_info = self.hallucination_check(outputs)
            if not check_info.hallucination:
                if DEBUG:
                    print("No hallucinations")
                text = join_if_nonempty(text, outputs.new_text.strip())
                if DEBUG:
                    print("Currently generated text", text, sep="\n")
                if outputs.ended or outputs.merged_blocks.len_tokens > self.generate_max_length:
                    if DEBUG:
                        if outputs.ended:
                            print("Terminator detected." if outputs.ended else "The text has reached its maximum length.")
                    break
            else:
                if DEBUG:
                    print("Hallucination detected. Preparing to retrieve information.")
                retrieve_qry = self.generate_retrieve_qry(outputs, check_info)  
                if DEBUG:
                    print(f"retrieve_qry: {retrieve_qry}")
                docs = self.retriever(retrieve_qry, topk=self.retrieve_topk)   
                self.counter.retrieve += 1
                prompt = demo
                prompt += "\nContext:\n"
                for i, doc in enumerate(docs):
                    print(f"doc{i}:{doc}")
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                for i in [1, 2, 3]: # "Question:", question, "\nAnswer:"
                    prompt += outputs.blocks[i].text
                text = self.tokenizer.convert_tokens_to_string(
                    outputs.blocks[-2].tokens # text
                    + outputs.blocks[-1].tokens[:outputs.blocks[-1].range_[check_info.curr_st][0]] # ptext
                )
                prompt += text
                ended, new_texts = self.generator.simply_generate(
                    prompt, 
                    max_length=self.generate_max_length,
                )
                if self.use_counter == True:
                    self.counter.add_generate(new_texts, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = get_top_sentence(new_texts)
                # text += new_text
                text = join_if_nonempty(text, new_text.strip())
                if DEBUG:
                    print("Regenerate new text:", new_text, sep="\n")
                if DEBUG:
                    print("The text currently generated is:", text, sep="\n")
                if ended and len(new_text) >= len(new_texts.strip()):
                    if DEBUG:
                        print("Terminator detected.")
                    break
                if len(self.tokenizer.encode(text)) > self.generate_max_length:
                    if DEBUG:
                        print("The text has reached its maximum length.")
                    break
            if old_len >= len(text):
                logger.info("old_len >= len(text) !")
                break
        if DEBUG:
            print("finished", text, sep="\n")
        return text