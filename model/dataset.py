import random

import preprocess
import torch
import transformers
from torch.utils.data import DataLoader, Dataset, random_split

default_tokenizer = transformers.GPT2Tokenizer.from_pretrained(
    'gpt2', add_prefix_space=True,
    bos_token='<prompt>', pad_token='<pad>', sep_token='<answer>'
)
default_special_ids = set(default_tokenizer.all_special_ids)


def load_dataset(target, batch_size, window_size, tokenizer=default_tokenizer):
    source = preprocess.unfreeze_dataset(target)

    dataset = ChatDataset(source, tokenizer, window_size=window_size)
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
    def __init__(self,
                 source,
                 tokenizer,
                 window_size):
        self.source = source
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_len = min(self.window_size, self.tokenizer.max_len)

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

    @staticmethod
    def str2ids(text,
                tokenizer=default_tokenizer,
                add_bos_token=False,
                add_sep_token=False,
                add_eos_token=False):
        """
        Converts a string in a sequence of ids (integer), using provided
            tokenizer.

        Arg:
            text (str, List[str] or List[int]: The first sequence to be
                encoded. This can be a string, a list of strings (tokenized
                string using the tokenize method) or a list of integers
                (tokenized string ids using the convert_tokens_to_ids method)
            tokenizer: An instance of transformers.PreTrainedTokenizer.
        Return:
            A list of integers

        >>> ChatDataset.str2ids('I am fine')
        [40, 716, 3734]
        >>> ChatDataset.str2ids('I am fine', add_bos_token=True)
        [50256, 40, 716, 3734]
        >>> ChatDataset.str2ids('I am fine', add_sep_token=True, add_eos_token=True)
        [50256, 40, 716, 3734, 50256]
        """
        ids: list = tokenizer.encode(text)
        if add_bos_token:
            ids.insert(0, tokenizer.bos_token_id)
        if add_sep_token:
            ids.insert(0, tokenizer.sep_token_id)
        if add_eos_token:
            ids.append(tokenizer.eos_token_id)
        return ids

    @staticmethod
    def ids2str(token_ids,
                tokenizer=default_tokenizer,
                skip_special_tokens=False):
        """
        Converts a sequence of ids (integer) in a string, using the provided
            tokenizer and vocabulary with options to remove special tokens.

        Arg:
            token_ids: list of tokenized input ids. Can be obtained using the
                str2ids method.
            tokenizer: An instance of transformers.PreTrainedTokenizer.
            skip_special_tokens: if set to True, will replace special tokens.
        Return:
            A list of integers

        >>> ChatDataset.ids2str(ChatDataset.str2ids('I am fine'))
        'I am fine'
        """
        return tokenizer.decode(token_ids)

    @classmethod
    def encode(Class, prompt, answer,
               tokenizer=default_tokenizer, max_len=1024):
        """
        Encode a prompt, answer pair with self.tokenizer

        Args:
            prompt (str, List[str] or List[int] – The first sequence to be
                encoded. This can be a string, a list of strings (tokenized
                string using the tokenize method) or a list of integers
                (tokenized string ids using the convert_tokens_to_ids method)
            answer (str, List[str] or List[int] – The first sequence to be
                encoded. This can be a string, a list of strings (tokenized
                string using the tokenize method) or a list of integers
                (tokenized string ids using the convert_tokens_to_ids method)
            tokenizer: Tokenizer used for tokenizing input (default to
                default_tokenizer)
            max_len: Sequence length, shorter sequences will be padded. Default
                to 1024
        Return:
            A tuple of shape

                (
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    special_tokens_mask,
                    mc_token_ids
                )

            + input_ids: list of token ids to be fed to a model
            + token_type_ids: list of token type ids to be fed to a model
            + attention_mask: list of indices specifying which tokens should
                be attended to by the model
            + special_tokens_mask: if adding special tokens, this is a list of
                [0, 1], with 0 specifying special added tokens and 1 specifying
                sequence tokens.
            + mc_token_ids: the index of the classification token in each
                input sequence.
        """
        prompt = Class.str2ids(prompt, add_bos_token=True)
        answer = Class.str2ids(answer, add_sep_token=True, add_eos_token=True)
        mc_token_ids = torch.tensor([len(prompt) + len(answer) - 1])
        # three special tokens are added
        max_len += 3

        encoding_obj = tokenizer.encode_plus(
            prompt,
            answer,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_tensors='pt',
            rerurn_token_type_ids=True,
            return_attention_mask=True,
            return_special_tokens_mask=True
        )

        input_ids = encoding_obj['input_ids'].squeeze()
        special_token_ids = torch.tensor(list(set(tokenizer.all_special_ids)))
        special_tokens_mask = input_ids.view(1, -1).eq(
            special_token_ids.view(-1, 1)).squeeze()

        return (
            input_ids,  # input_ids
            encoding_obj['attention_mask'].squeeze(),  # attention_mask
            encoding_obj['token_type_ids'].squeeze(),  # token_type_ids
            special_tokens_mask,  # special_token_masks
            mc_token_ids
        )

    def __getitem__(self, index):
        pair = self.source[index]
        prompt = pair['prompt']
        is_original, answer = self.__next_sentence_prediction(pair['answer'])

        input_ids, attention_mask, token_type_ids, special_tokens_mask, mc_token_ids = ChatDataset.encode(
            prompt, answer, self.tokenizer, self.max_len)

        if is_original:
            lm_mask = token_type_ids.clone().masked_fill_(special_tokens_mask,
                                                          0)
            lm_labels = input_ids.clone().masked_fill_(lm_mask == 0, -100)
            mc_labels = torch.tensor(1.0)
        else:
            lm_labels = input_ids.clone().fill_(-100)
            mc_labels = torch.tensor(0.0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'mc_token_ids': mc_token_ids,
            'lm_labels': lm_labels,
            'mc_labels': mc_labels
        }
