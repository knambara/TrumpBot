import random

import preprocess
import torch
import transformers
from torch.utils.data import DataLoader, Dataset, random_split

default_tokenizer = transformers.GPT2Tokenizer.from_pretrained(
    'gpt2', add_prefix_space=True,
    bos_token='<prompt>', pad_token='<pad>', sep_token='<answer>'
)


def load_dataset(target, batch_size, tokenizer=default_tokenizer):
    source = preprocess.unfreeze_dataset(target)

    dataset = ChatDataset(source, tokenizer)
    train_dataset, validate_dataset, test_dataset = split_dataset(dataset)
    return (DataLoader(train_dataset, batch_size=batch_size,
                       shuffle=True, drop_last=True),
            DataLoader(validate_dataset, shuffle=True),
            DataLoader(test_dataset))


def split_dataset(dataset: Dataset, train_frac=0.8, validate_frac=0.1):
    """
    Splits the main dataset into train, validation, test dataset
    """
    n_total = len(dataset)
    n_train = int(n_total * train_frac)
    n_validate = int(n_total * validate_frac)
    n_test = n_total - n_train - n_validate

    return random_split(dataset, (n_train, n_validate, n_test))


class ChatDataset(Dataset):
    def __init__(self, source, tokenizer: transformers.PreTrainedTokenizer):
        self.source = source
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.source)

    def __random_answer(self):
        """
        Draw a random answer from source
        """
        return random.choice(self.source)['answer']

    def __next_sentence_prediction(self, answer):
        """
        Prepare answer for next sentence prediction task:
            + with 50% probability, original answer will be used
            + with 50% probability, a random answer (distractor) will be used

        Arg:
            answer: original answer
        Return:
            A tuple of

                (is_original, answer)

            where
                + is_original is True if original answer is used
                + answer is either the original answer or a random answer
        """
        if random.random() < 0.5:
            return (True, answer)
        else:

            return (False, self.__random_answer())

    def __getitem__(self, index):
        pair = self.source[index]
        prompt = pair['prompt']
        is_original, answer = self.__next_sentence_prediction(pair['answer'])

        encoding_obj = self.tokenizer.encode_plus(
            prompt,
            answer,
            add_special_tokens=True,
            max_length=self.tokenizer.max_len,
            pad_to_max_length=True,
            return_tensors='pt',
            rerurn_token_type_ids=True,
            return_attention_mask=True,
            return_special_tokens_mask=True
        )

        input_ids = encoding_obj['input_ids'].squeeze()
        attention_mask = encoding_obj['attention_mask'].squeeze()
        token_type_ids = encoding_obj['token_type_ids'].squeeze()

        if is_original:
            lm_labels = input_ids.clone().masked_fill_(token_type_ids == 0,
                                                       -100)
            mc_labels = torch.tensor(1.0)
        else:
            lm_labels = input_ids.clone().fill_(-100)
            mc_labels = torch.tensor(0.0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'lm_labels': lm_labels,
            'mc_labels': mc_labels
        }
