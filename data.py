from typing import Dict, List, Callable, Union, Callable
import logging
import os
import json
import re
import string
from collections import Counter
from tqdm import tqdm
import numpy as np
from datasets import Dataset

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

class BaseDataset:
    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        return {}

    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]], 
        ground_truth_id: Union[str, List[str]] = None 
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth) 
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id)) 

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths]) 
        return {'correct': correct, 'incorrect': 1 - correct} 

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0} 
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth) 
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split() 
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens) 
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens) 
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k]) 
        return final_metric

    def format(self, fewshot: int = 0):
        def _format(
            example: Dict, 
            use_answer: bool = False, 
            input_template_func: Callable = lambda ques: f'Question: {ques}\nAnswer:' # 生成问题部分的格式化字符串
        ):
            q = example['question'] 
            if 'cot' in example:
                cot = example['cot'] if type(example['cot']) is str else ''.join(example['cot']) 
            else:
                cot = None
            a = example['answer'] 

            query = input_template_func(q) # (e.g. WikiMultiHopQA) lambda ques: f'Question: {ques}\nAnswer:'
            if use_answer:
                query += ('' if query[-1] in {'\n', ' '} else ' ') + self.output_template(cot, a) 
            return query 


        demo = [{
            'question': self.examplars[i]['question'],
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template), # 打包了question和answer
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else [] # 在WikiMultiHopQA中没有ctxs
        } for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            case = _format(example, use_answer=False, input_template_func=self.test_input_template)
            # ctx
            example['demo'] = demo
            example['case'] = case 
            return example
        self.dataset = self.dataset.map(_format_for_dataset)
    
    def get_real_prediction(self, pred):
        return pred


class StrategyQA(BaseDataset):
    examplars: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'ctxs': [(None, "Hamsters are prey animals."),
                (None, "Prey animals provide food for predators.")],
            'cot': ('Hamsters are prey animals. ',
                'Prey are food for predators. ',
                'Thus, hamsters provide food for some animals.'),
            'answer': 'yes',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'ctxs': [(None, "Brooke Shields graduated from Princeton University."),
                (None, "Princeton is ranked as the number 1 national college by US news."),
                (None, "University of Pennsylvania is ranked as number 6 national college by US news."),
                (None, "Princeton only admits around 6 percent of applicants as of 2018."),
                (None, "University of Pennsylvania accepts around 9% of applicants as of 2018.")],
            'cot': ('Brooke Shields went to Princeton University. ',
                'Princeton University is about as academically rigorous as the University of Pennsylvania. ',
                'Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
            'answer': 'yes',
        },
        {
            'question': "Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'ctxs': [(None, "Hydrogen is the first element and has an atomic number of one."),
                (None, "The Spice Girls has five members."),
                (None, "To square a number, you multiply it by itself.")],
            'cot': ("Hydrogen has an atomic number of 1. ",
                "1 squared is 1. ",
                "There are 5 Spice Girls. ",
                "Thus, Hydrogen's atomic number squared is less than 5."),
            'answer': 'no',
        },
        {
            'question': "Is it common to see frost during some college commencements?",
            'ctxs': [(None, "Frost isn't uncommon to see during the month of December, as it is the winter."),
                (None, "College commencement ceremonies often happen during the months of December, May, and sometimes June.")],
            'cot': ("College commencement ceremonies can happen in December, May, and June. ",
                "December is in the winter, so there can be frost. ",
                "Thus, there could be frost at some commencements."),
            'answer': 'yes',
        },
        {
            'question': "Could a llama birth twice during War in Vietnam (1945-46)?",
            'ctxs': [(None, "The War in Vietnam (1945-46) lasted around 6 months."),
                (None, "The gestation period for a llama is 11 months.")],
            'cot': ("The War in Vietnam was 6 months. ",
                "The gestation period for a llama is 11 months, which is more than 6 months. ",
                "Thus, a llama could not give birth twice during the War in Vietnam."),
            'answer': 'no',
        },
        {
            'question': "Would a pear sink in water?",
            'ctxs': [(None, "The density of a raw pear is about 0.59 g/cm^3."),
                (None, "The density of water is about 1 g/cm^3."),
                (None, "Objects only sink if they are denser than the surrounding fluid.")],
            'cot': ("The density of a pear is about 0.6g/cm^3, which is less than water. ",
                "Objects less dense than water float. ",
                "Thus, a pear would float."),
            'answer': 'no',
        }
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading StrategyQA from {data_path}")
        dataset = []
        with open(os.path.join(data_path, "strategyqa_train.json"), "r") as fin:
            dataset_1 = json.load(fin)
        with open(os.path.join(data_path, "strategyqa_train_paragraphs.json"), "r") as fin:
            dataset_2 = json.load(fin)
        for data in tqdm(dataset_1):
            example = {
                "qid": data["qid"], 
                "question": data["question"], 
                "cot": " ".join(data["facts"]), 
                "answer": "yes" if data["answer"] == True else "no", 
            }
            title = []
            ctxs = []
            for evi in data["evidence"][0]:
                if type(evi) == list:
                    for t in evi:
                        if type(t) == list:
                            title.extend(t)
                        else:
                            title.append(t)
                else:
                    title.append(evi)
            for tl in title:
                if tl == "operation" or tl == "no_evidence":
                    continue
                if tl in dataset_2:
                    ctxs.append(dataset_2[tl]["content"])
            example["ctxs"] = " ".join(ctxs)
            dataset.append(example)
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:]
                if pred[0:3].lower() == 'yes':
                    return "yes"
                else:
                    return "no"
        else:
            return ""

class WikiMultiHopQA(BaseDataset):
    examplars: List[Dict] = [
        {
            'question': "When did the director of film Hypocrite (Film) die?",
            'cot': "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013.", # cot就是CoT，Chain of Thought
            'answer': "19 June 2013",
        },
        {
            'question': "Are both Kurram Garhi and Trojkrsti located in the same country?",
            'cot': "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country.",
            'answer': "no",
        },
        {
            'question': "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            'cot': "Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality.",
            'answer': "no",
        },
        {
            'question': "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
            'cot': "Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's father-in-law is Genghis Khan.",
            'answer': "Genghis Khan",
        },
        {
            'question': "Who was born first out of Martin Hodge and Ivania Martinich?",
            'cot': "Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first.",
            'answer': "Martin Hodge",
        },
        {
            'question': "When did the director of film Laughter In Hell die?",
            'cot': "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963.",
            'answer': "August 25, 1963",
        },
        {
            'question': "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            'cot': "The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two.",
            'answer': "Twenty Plus Two",
        },
        {
            'question': "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            'cot': "Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah.",
            'answer': "Prithvipati Shah",
        }
    ]
    demo_input_template = test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str): 
        logger.info(f"Loading WikiMultiHopQA from {data_path}") 
        dataset = []
        with open(os.path.join(data_path, 'dev.json'), 'r') as fin: 
            js = json.load(fin) 
            for example in tqdm(js):
                qid = example['_id']  
                question = example['question'] 
                ans = example['answer'] 
                ans_id = example['answer_id'] 
                # ctxs = example['ctxs']
                dataset.append({ 
                    'qid': qid,
                    'question': question,
                    'answer': ans,
                    'answer_id': ans_id,
                })
        self.dataset = Dataset.from_list(dataset) 
        self.init_id_aliases(data_path)
           
    @classmethod
    def init_id_aliases(cls, data_path): # alias
        cls.id_alias: Dict[str, List[str]] = {}
        with open(os.path.join(data_path, 'id_aliases.json'), 'r') as fin:
            for l in fin:
                l = json.loads(l) # {"Q_id": str, "aliases": List[str], "demonyms": List[str]}
                cls.id_alias[l['Q_id']] = l['aliases']

    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        if ground_truth_id and ground_truth_id in cls.id_alias:
            return cls.id_alias[ground_truth_id]
        else:
            return []

    def get_real_prediction(self, pred):
        if "the answer is" in pred:
            beg = pred.find("the answer is") + len("the answer is") + 1
            pred = pred[beg:] # delete final "."
            if pred.endswith("</s>"):
                pred = pred[:len(pred) - len("</s>")]
            if pred.endswith("<|endoftext|>"):
                pred = pred[:len(pred) - len("<|endoftext|>")]
            if pred.endswith("."):
                pred = pred[:-1]
            return pred
        else:
            return pred


class HotpotQA(BaseDataset):
    examplars: List[Dict] = [
        {
            'question': "Jeremy Theobald and Christopher Nolan share what profession?",
            'cot': "Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer.",
            'answer': "producer",
        },
        {
            'question': "What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?",
            'cot': "Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau.",
            'answer': "The Phantom Hour.",
        },
        {
            'question': "How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?",
            'cot': "The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988. The number of episodes Reply 1988 has is 20.",
            'answer': "20",
        },
        {
            'question': "Were Lonny and Allure both founded in the 1990s?",
            'cot': "Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s.", 
            'answer': "no",
        },
        {
            'question': "Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?",
            'cot': "The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn.",
            'answer': "Scott Glenn",
        },
        {
            'question': "What was the 2014 population of the city where Lake Wales Medical Center is located?",
            'cot': "Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140.",
            'answer': "15,140",
        },
        {
            'question': "Who was born first? Jan de Bont or Raoul Walsh?",
            'cot': "Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first.",
            'answer': "Raoul Walsh",
        },
        {
            'question': "In what country was Lost Gravity manufactured?",
            'cot': "The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company.",
            'answer': "Germany",
        },
        {
            'question': "Which of the following had a debut album entitled \"We Have an Emergency\": Hot Hot Heat or The Operation M.D.?",
            'cot': "The debut album of the band \"Hot Hot Heat\" was \"Make Up the Breakdown\". The debut album of the band \"The Operation M.D.\" was \"We Have an Emergency\".",
            'answer': "The Operation M.D.",
        },
        {
            'question': "How many awards did the \"A Girl Like Me\" singer win at the American Music Awards of 2012?",
            'cot': "The singer of \"A Girl Like Me\" singer is Rihanna. In the American Music Awards of 2012, Rihana won one award.",
            'answer': "one",
        },
        {
            'question': "The actor that stars as Joe Proctor on the series \"Power\" also played a character on \"Entourage\" that has what last name?",
            'cot': "The actor that stars as Joe Proctor on the series \"Power\" is Jerry Ferrara. Jerry Ferrara also played a character on Entourage named Turtle Assante. Thus, Turtle Assante's last name is Assante.",
            'answer': "Assante",
        },
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f'Answer the following question by reasoning step-by-step, following the example above.\nQuestion: {ques}\nAnswer:' 
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading HotpotQA from {data_path}")
        dataset = []
        with open(os.path.join(data_path, 'hotpotqa-dev.json'), "r") as fin:
            js = json.load(fin)
            for example in tqdm(js):
                qid = example["_id"]
                question = example["question"]
                answer = example['answer']
                context = example['context']
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': answer,
                })
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                if pred.endswith("</s>"):
                    pred = pred[:len(pred) - len("</s>")]
                if pred.endswith("<|endoftext|>"):
                    pred = pred[:len(pred) - len("<|endoftext|>")]
                if pred.endswith("."):
                    pred = pred[:-1]
                return pred
        else:
            return ""


class IIRC(BaseDataset):
    examplars: List[Dict] = [
        {
            "question": "What is the age difference between the kicker and the quarterback for the Chargers?",
            "cot": "The kicker for the Chargers is Nate Kaeding. The quarterback (QB) for the Chargers is Philip Rivers. Nate Kaeding was born in the year 1982. Philip Rivers was born in the year 1981. Thus, the age difference between them is of 1 year.",
            "answer": "1"
        },
        {
            "question": "How many years was the ship that took the battalion from New South Wales to Ceylon in service?",
            "cot": "The ship that took the battalion from New South Wales to Ceylon is General Hewitt. General Hewitt was launched in Calcutta in 1811. General Hewitt was sold for a hulk or to be broken up in 1864. So she served for a total of 1864 - 1811 = 53 years.",
            "answer": "53"
        },
        {
            "question": "What year was the theatre that held the 2016 NFL Draft built?",
            "cot": "The theatre that held the 2016 NFL Draft is Auditorium Theatre. The Auditorium Theatre was built in 1889.",
            "answer": "1889"
        },
        {
            "question": "How long had Milan been established by the year that Nava returned there as a reserve in the first team's defense?",
            "cot": "Nava returned to Milan as a reserve in the first team's defense in the year 1990. Milan had been established in the year 1899. Thus, Milan had been established for 1990 - 1899 = 91 years when Milan returned to Milan as a reserve in the first team's defense.",
            "answer": "91"
        },
        {
            "question": "When was the town Scott was born in founded?",
            "cot": "Scott was born in the town of Cooksville, Illinois. Cooksville was founded in the year 1882.",
            "answer": "1882"
        },
        {
            "question": "In what country did Wright leave the French privateers?",
            "cot": "Wright left the French privateers in Bluefield's river. Bluefields is the capital of the South Caribbean Autonomous Region (RAAS) in the country of Nicaragua.",
            "answer": "Nicaragua"
        },
        {
            "question": "Who plays the A-Team character that Dr. Hibbert fashioned his hair after?",
            "cot": "Dr. Hibbert fashioned his hair after Mr. T from The A-Team. Mr T.'s birthname is Lawrence Tureaud.",
            "answer": "Lawrence Tureaud"
        },
        {
            "question": "How many people attended the conference held near Berlin in January 1942?",
            "cot": "The conference held near Berlin in January 1942 is Wannsee Conference. Wannsee Conference was attended by 15 people.",
            "answer": "15"
        },
        {
            "question": "When did the country Ottwalt went into exile in founded?",
            "cot": "Ottwalt went into exile in the country of Denmark. Denmark has been inhabited since around 12,500 BC.",
            "answer": "12,500 BC"
        },
        {
            "question": "When was the J2 club Uki played for in 2001 founded?",
            "cot": "The J2 club that Uki played for is Montedio Yamagata. Montedio Yamagata was founded in 1984.",
            "answer": "1984"
        },
        {
            "question": "When was the person who produced A Little Ain't Enough born?",
            "cot": "A Little Ain't Enough was produced by Bob Rock. Bob Rock was born on April 19, 1954.",
            "answer": "April 19, 1954"
        },
        {
            "question": "Which of the schools Fiser is affiliated with was founded first?",
            "cot": "The schools that Fiser is affiliated with (1) Academy of Music, University of Zagreb (2) Mozarteum University of Salzburg (3) Croatian Music Institute orchestra. Academy of Music, University of Zagreb was founded in the year 1829. Mozarteum University of Salzburg was founded in the year 1841. Croatian Music Institute was founded in the year 1827. Thus, the school founded earliest of these is Croatian Music Institute.",
            "answer": "Croatian Music Institute"
        },
        {
            "question": "How many casualties were there at the battle that Dearing fought at under Jubal Early?",
            "cot": "Under Jubal Early, Dearing fought the First Battle of Bull Run. First Battle of Bull Run has 460 union casualties and 387 confederate casualties. Thus, in total the First Battle of Bull Run had 460 + 387 = 847 casualties.",
            "answer": "847"
        },
        {
            "question": "Which of the two congregations which provided leadership to the Pilgrims was founded first?",
            "cot": "The congregations which provided leadership to the Pilgrims are Brownists and Separatist Puritans. Brownist was founded in 1581. The Separatist Puritans was founded in 1640. Thus, Brownist was founded first.",
            "answer": "Brownist"
        },
        {
            "question": "How long had the Rock and Roll Hall of Fame been open when the band was inducted into it?",
            "cot": "The band was inducted into Rock and Roll Hall of Fame in the year 2017. Rock and Roll Hall of Fame was established in the year of 1983. Thus, Rock and Roll Hall of Fame been open for 2018 - 1983 = 34 years when the band was inducted into it.",
            "answer": "34"
        },
        {
            "question": "Did the Lord Sewer who was appointed at the 1509 coronation live longer than his king?",
            "cot": "Lord Sewer who was appointed at the 1509 coronation was Robert Radcliffe, 1st Earl of Sussex. Lord Sever's king in 1509 was Henry VIII of England. Robert Radcliffe, 1st Earl of Sussex was born in the year 1483, and died in the year 1542. So Robert lived for 1542 - 1483 = 59 years. Henry VIII of England was born in the year 1491 and died in the year 1547. So Henry VIII lived for 1547 - 1491 = 56 years. Thus, Robert Radcliffe lived longer than Henry VIII.",
            "answer": "yes"
        },
        {
            "question": "When was the place near where Manuchar was defeated by Qvarqvare established?",
            "cot": "Manuchar was defeated by Qvarqvare near Erzurum. Erzurum was founded during the Urartian period.",
            "answer": "Urartian period"
        },
        {
            "question": "What year was the man who implemented the 46 calendar reform born?",
            "cot": "The man who implemented the 46 calendar reform is Julius Caesar. Julius Caesar was born in the year 100 BC.",
            "answer": "100 BC"
        },
        {
            "question": "How many years after the first recorded Tommy John surgery did Scott Baker undergo his?",
            "cot": "The first recorded Tommy John surgery happened when it was invented in the year 1974. Scott Baker underwent Tommy John surgery in the year 2012. Thus, Scott Baker underwent Tommy John surgery 2012 - 1974 = 38 years after it was first recorded.",
            "answer": "38"
        },
        {
            "question": "Which was the older of the two players who found the net in the Double-Headed Eagle of the North in the sixth final for PAOK?",
            "cot": "The two players who found the net in the Double-Headed Eagle of the North in the sixth final for PAOK are Koudas and Matzourakis. Koudas was born on 23 November 1946. Matzourakis was born on 6 June 1949. Thus, the older person among the two is Koudas.",
            "answer": "Koudas"
        }
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:' 
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading IIRC dev from {data_path}")
        dataset = []
        with open(os.path.join(data_path, 'dev.json'), "r") as fin:
            js = json.load(fin)
            for tmp in tqdm(js):
                for example in tmp['questions']:
                    qid = example["qid"]
                    question = example['question']
                    ans = example['answer']
                    if ans['type'] == 'none':
                        continue
                    elif ans['type'] == 'value' or ans['type'] == 'binary':
                        answer = [ans['answer_value']]
                    elif ans['type'] == 'span':
                        answer = [v['text'].strip() for v in ans['answer_spans']]

                    dataset.append({
                        'qid': qid,
                        'question': question,
                        'answer': answer,
                    })
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                for stop_word in ["</s>", "<|endoftext|>", "\n", "."]:
                    if pred.endswith(stop_word):
                        pred = pred[:len(pred) - len(stop_word)]
                return pred
        else:
            return ""
            
class BIOASQ(BaseDataset):
    examplars: List[Dict] = [
        {
            "question": "Is the protein Papilin secreted?",
            "cot": "The context mentions that 'mig-6/papilin' encodes secreted extracellular matrix proteins and that papilin is an extracellular matrix glycoprotein. It further states that extracellular matrix proteins like papilin are secreted into the medium by hemocytes. This indicates that papilin is indeed secreted.",
            "answer": "yes"
        },
        {
            "question": "Are long non coding RNAs spliced?",
            "cot": "The context mentions that long non-coding RNAs (lncRNAs) are generated through pathways similar to protein-coding genes, with splicing signals present. It further states that for alternative exons and lncRNAs, splicing tends to occur later, but in some cases, they might remain unspliced. Additionally, the context introduces an approach to predict spliced lncRNAs and suggests that lncRNAs have similar alternative splicing patterns to mRNAs. This implies that long non-coding RNAs can be spliced.",
            "answer": "yes"
        },
        {
            "question": "Is RANKL secreted from the cells?",
            "cot": "The context mentions that RANKL is a cytokine predominantly secreted by osteoblasts. It further states that activated human T cells also express alternative mRNA transcripts encoding a secreted form of RANKL. Additionally, the context describes RANKL’s role in osteoclastogenesis and bone resorption, indicating that RANKL is indeed secreted from cells.",
            "answer": "yes"
        },
        {
            "question": "Does metformin interfere thyroxine absorption?",
            "cot": "LT4 absorption is unchanged by concomitant metformin ingestion. It has been hypothesized that metformin may suppress serum thyrotropin (TSH) concentrations by enhancing LT4 absorption or by directly affecting the hypothalamic-pituitary axis.",
            "answer": "no"
        },
        {
            "question": "Has Denosumab (Prolia) been approved by FDA?",
            "cot": "The context clearly states that Denosumab was approved by the FDA in 2010 for the prevention of skeletal fractures in patients with bone metastases from solid tumors. Additionally, the approval for the treatment of postmenopausal osteoporosis in women at high risk of fractures was mentioned, along with its approval for other indications. Therefore, Denosumab has been approved by the FDA.",
            "answer": "yes"
        },
        {
            "question": "Are transcription and splicing connected?",
            "cot": "The context provides clear evidence that transcription and splicing are connected. It mentions that splicing is often cotranscriptional, indicating that splicing and transcription occur together. The context also discusses how transcription-associated features such as protein recruitment to the transcribing machinery and elongation kinetics affect splicing. Furthermore, it explains that the structure of chromatin and the rate of transcription elongation can influence alternative splicing, reinforcing the connection between transcription and splicing. Therefore, transcription and splicing are indeed connected.",
            "answer": "yes"
        },
        {
            "question": "Is Alu hypomethylation associated with breast cancer?",
            "cot": "The context clearly associates Alu hypomethylation with breast cancer. It mentions that Alu hypomethylation is observed in the HER2 enriched subtype of breast cancer, and that it correlates with negative estrogen receptor (ER) status in inflammatory breast cancer (IBC). Additionally, low Alu methylation is linked with poor disease-free survival in patients, suggesting a potential role in breast cancer progression. Furthermore, the prominent hypomethylation of Alu in this specific subtype is related to chromosomal instability. Therefore, Alu hypomethylation is indeed associated with breast cancer.",
            "answer": "yes"
        },
        {
            "question": "Proteomic analyses need prior knowledge of the organism complete genome. Is the complete genome of the bacteria of the genus Arthrobacter available?",
            "cot": "The context mentions multiple instances of complete or draft genome sequences of bacteria from the genus Arthrobacter. Specifically, it lists the complete genome sequence of Arthrobacter phenanthrenivorans, Arthrobacter sp. Rue61a, Arthrobacter sp. B6, Arthrobacter sp. ZXY-2, Arthrobacter sp. EpSL27, and Arthrobacter sp. KI72, which confirms that the complete genomes of several Arthrobacter species are available.",
            "answer": "yes"
        }
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:' 
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading BIOASQ from {data_path}")
        dataset = []

        file_names = ['Task7B_yesno_test.json', 'Task7B_yesno_validation.json', 'Task7B_yesno_train.json']
        all_data = []  
        for file_name in file_names:
            file_path = os.path.join(data_path, file_name)
            with open(file_path, 'r') as fin:
                try:
                    data = json.load(fin)  
                    all_data.extend(data)  
                except json.JSONDecodeError as e:
                    print(f"Error reading {file_path}: {e}")
        
        for tmp in tqdm(all_data):
            question = tmp['questions']
            answer = tmp['answer']
            dataset.append({
                'question': question,
                'answer': answer,
            })
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:]
                if pred[0:3].lower() == 'yes':
                    return "yes"
                else:
                    return "no"
        else:
            return ""
        
class PubmedQA(BaseDataset):
    examplars: List[Dict] = [
        {
            "question": "Is there a connection between sublingual varices and hypertension?",
            "cot": "An association was found between sublingual varices and hypertension. Examining the lateral borders of the tongue is easily done, causes no harm and could be a valuable method for the dental profession to take active part in preventive healthcare.",
            "answer": "yes"
        },
        {
            "question": "Is the affinity column-mediated immunoassay method suitable as an alternative to the microparticle enzyme immunoassay method as a blood tacrolimus assay?",
            "cot": "The ACMIA method used for a tacrolimus assay is precise and has advantages, including the lack of a required pretreatment procedure. Furthermore, it is only slightly influenced by the hematologic or biochemical status of the samples.",
            "answer": "yes"
        },
        {
            "question": "Does a physician's specialty influence the recording of medication history in patients' case notes?",
            "cot": "Physicians appear to document more frequently and in greater depth medication history information that may aid the diagnostic tasks in their specific specialty. Researchers and other users of medication history data documented in patients' medical records by physicians may want to take special cognizance of this phenomenon.",
            "answer": "yes"
        },
        {
            "question": "Locoregional opening of the rodent blood-brain barrier for paclitaxel using Nd:YAG laser-induced thermo therapy: a new concept of adjuvant glioma therapy?",
            "cot": "LITT induces a locoregional passage of chemotherapeutic agents into the brain tissue. This is of potential interest for the treatment of brain tumors.",
            "answer": "yes"
        },
        {
            "question": "Does quantitative left ventricular regional wall motion change after fibrous tissue resection in endomyocardial fibrosis?",
            "cot": "Although endomyocardial fibrosis patients have improved clinical symptoms after surgery, the global left ventricular ejection fraction and regional wall motion in these patients do not change. This finding suggests that other explanations, such as improvements in diastolic function, may be operational.",
            "answer": "no"
        },
        {
            "question": "Does gestational age misclassification explain the difference in birthweights for Australian aborigines and whites?",
            "cot": "Gestational age misclassification is an unlikely explanation for the reported divergence in average birth-weights for Aborigines and whites. The results might help with the interpretation of other between-population comparisons.",
            "answer": "no"
        },
        {
            "question": "Utility of unenhanced fat-suppressed T1-weighted MRI in children with sickle cell disease -- can it differentiate bone infarcts from acute osteomyelitis?",
            "cot": "The bone marrow signal intensity on unenhanced T1-W fat-saturated MR images is not a reliable criterion to differentiate bone infarcts from osteomyelitis in children.",
            "answer": "no"
        },
        {
            "question": "Do African American women require fewer calories to maintain weight?",
            "cot": "These results do not support the view that AA women are at greater risk for obesity because they require fewer calories to maintain weight.",
            "answer": "no"
        }
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:' 
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading PubMedQA from {data_path}")
        dataset = []

        file_names = ['pqal_train_set.json', 'test_set.json']
        all_data = []  

        for file_name in file_names:
            file_path = os.path.join(data_path, file_name)
            if not os.path.exists(file_path):
                logger.error(f"The file {file_path} does not exist.")
                continue

            try:
                with open(file_path, 'r') as fin:
                    data = json.load(fin)  
                    all_data.extend(data.values())
            except json.JSONDecodeError as e:
                logger.error(f"A JSON parsing error occurred while reading {file_path}: {e}")
            except Exception as e:
                logger.error(f"An unknown error occurred while reading {file_path}: {e}")

        for tmp in tqdm(all_data, desc="Processing data"):
            try:
                question = tmp['QUESTION']  
                answer = tmp['final_decision']  
                
                dataset.append({
                    'question': question,
                    'answer': answer,
                    # 'contexts': contexts,  
                })
            except KeyError as e:
                logger.warning(f"Data item missing necessary fields: {e}")
            except Exception as e:
                logger.error(f"An error occurred while processing the data: {e}")

        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:]
                if pred[0:3].lower() == 'yes':
                    return "yes"
                else:
                    return "no"
        else:
            return ""
