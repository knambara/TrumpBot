import math
import re
import string

from model import Model


class Bot:
    def __init__(self,
                 model_filename='model.pt',
                 answer_min_len=2,
                 answer_max_len=20,
                 temperature=0.7,
                 top_k=0,
                 top_p=0.9,
                 threshold=-math.inf):
        self.__load_model(model_filename)

        self.answer_min_len = 2
        self.answer_max_len = 20
        self.temperature = 0.7
        self.top_k = 0
        self.top_p = 0.9
        self.threshold = -math.inf

        self.translation = str.maketrans('', '', string.punctuation)
        self.reduce_multiple_space = re.compile(r'\s{2,}')

    def __load_model(self, filename='model.pt'):
        self.model = Model()
        self.model.load(filename)

    def answer(self, prompt):
        ans_data = self.model.answer(
            prompt,
            answer_min_len=self.answer_min_len,
            answer_max_len=self.answer_max_len,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            threshold=self.threshold
        )
        raw_answer = ans_data['str']

        # humanize
        answer = raw_answer.translate(self.translation)
        answer = self.reduce_multiple_space.sub(' ', answer)
        answer = answer.strip()
        return answer.lower()
