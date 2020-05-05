import argparse
import math
import os
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

import comet_ml
import torch
import transformers
from comet_ml import Experiment
from dataset import ChatDataset
from dataset import default_special_ids as special_ids
from dataset import default_tokenizer as tokenizer
from dataset import load_dataset
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

hyperparams = {
    "num_epochs": 3,
    "batch_size": 16,
    "window_size": 50,
    "accumulation_steps": 1,
    "learning_rate": 2e-5
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    """
    This class is a wrapper around default_model
        + forward function is directly proxied to default_model

    Additional, this wrapper provides
        + loss: a function to compute model loss on both tasks
    """
    def __init__(self,
                 window_size=hyperparams['window_size'],
                 device=device,
                 lm_coeff=1,
                 mc_coeff=1,
                 savedir='models',
                 max_norm=1.0):
        super().__init__()

        self.model = transformers.GPT2DoubleHeadsModel.from_pretrained('gpt2')

        self.window_size = window_size
        self.lm_coeff = lm_coeff
        self.mc_coeff = mc_coeff
        self.max_norm = max_norm

        self.savedir = savedir
        self.device = device
        self.to(device)

    def __lm_logits(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)[0]

    def forward(self, input_ids, mc_labels, **kwargs):
        lm_loss, lm_logits, mc_logits, _ = self.model(input_ids, **kwargs)
        mc_loss, loss = self.loss(lm_loss, mc_logits, mc_labels)
        return (loss, lm_loss, mc_loss, lm_logits, mc_logits)

    def __unpack_data(self, data):
        return (
            data['input_ids'].to(self.device),
            data['attention_mask'].to(self.device),
            data['token_type_ids'].to(self.device),
            data['mc_token_ids'].to(self.device),
            data['lm_labels'].to(self.device),
            data['mc_labels'].to(self.device)
        )

    def __compute_loss(self, lm_loss, mc_loss):
        """
        Returns a weighted average of lm_loss and mc_loss

        Formula:
            total_loss = lm_loss * lm_coeff + mc_loss * mc_coeff

        Arg:
            lm_loss: Loss associated with language modeling task
            mc_loss: Loss associated with next sentence prediction task:
                given a pair of sentence (A, B), determines whether B is
                the original answer (response) to A
        Returns:
            A weighted average of both losses
        """
        return lm_loss * self.lm_coeff + mc_loss * self.mc_coeff

    def __compute_mc_loss(self, mc_logits, mc_labels):
        """
        >>> model = Model()
        >>> mc_logits1 = torch.tensor([1.0, -1.0])
        >>> mc_logits2 = torch.tensor([1.0, 1.0])
        >>> mc_labels = torch.tensor([1.0, 1.0])
        >>> loss1 = model._Model__compute_mc_loss(mc_logits1, mc_labels)
        >>> loss2 = model._Model__compute_mc_loss(mc_logits1, mc_labels)
        >>> (loss1 > loss2).item()
        False
        """
        loss_func = nn.BCEWithLogitsLoss()
        return loss_func(mc_logits.view(-1), mc_labels.view(-1))

    def loss(self, lm_loss, mc_logits, mc_labels):
        """
        Compute combined model loss on
            + language modeling task
            + next sentence prediction task

        Args:
            lm_loss: Loss associated with language modeling task
            mc_logits: (batch_size, ) Prediction scores of the multiple choice
                classification head (scores for each choice before SoftMax).
                mc_logits = forward(...)[2]
            mc_labels: (batch_size, ) Binary labels for next sentence
                prediction task
                Given a pair of sentence (A, B), the label is:
                    + 1.0 when B is the original answer for A
                    + 0.0 when B is a distractor answer
        Returns:
            A tuple of (mc_loss, total_loss)
                + mc_loss: 1d tensor containing the next sentence
                    prediction task
                + total_loss: 1d tensor containing the combined loss
        """
        mc_loss = self.__compute_mc_loss(mc_logits, mc_labels)
        return (mc_loss, self.__compute_loss(lm_loss, mc_loss))

    def save(self, filename='model.pt'):
        if not os.path.isdir(self.savedir):
            print(f'create model output directory at {self.savedir}')
            os.mkdir(self.savedir)

        filepath = os.path.join(self.savedir, filename)
        print(f'saving model to {filepath}')
        torch.save(self.state_dict(), filepath)

    def load(self, filename='model.pt'):
        filepath = os.path.join(self.savedir, filename)
        print(f'loading model from {filepath}')
        model = torch.load(filepath, map_location=self.device.type)
        self.load_state_dict(model)

    def accuracy(self, lm_logits, lm_labels):
        preds = torch.argmax(lm_logits, dim=2)
        num_valid_preds = torch.sum(lm_labels != -100).double()
        num_correct_preds = torch.sum((preds == lm_labels))
        accuracy = (num_correct_preds / num_valid_preds)
        return (accuracy, num_correct_preds, num_valid_preds)

    def train_(self,
               train_dataloader: DataLoader,
               optimizer,
               scheduler,
               experiment,
               num_epochs,
               accumulation_steps):
        self.train()

        total_lm_loss = torch.tensor(0.0)
        num_batches = 0

        accumulated_correct_preds = torch.tensor(0.0)
        accumulated_valid_preds = torch.tensor(0.0)

        optimizer.zero_grad()
        with experiment.train():
            for which_epoch in range(num_epochs):
                for train_data in tqdm(train_dataloader):
                    num_batches += 1

                    input_ids, attention_mask, token_type_ids, mc_token_ids, \
                        lm_labels, mc_labels = self.__unpack_data(train_data)

                    loss, lm_loss, mc_loss, lm_logits, _ = self(
                        input_ids,
                        mc_labels,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        lm_labels=lm_labels
                    )
                    loss = loss / accumulation_steps
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

                    _, num_correct_preds, num_valid_preds = self.accuracy(
                        lm_logits, lm_labels)

                    accumulated_correct_preds += num_correct_preds
                    accumulated_valid_preds += num_valid_preds

                    if num_batches % accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        accuracy = accumulated_correct_preds / accumulated_valid_preds
                        experiment.log_metric(
                            'accuracy', accuracy.item())
                        experiment.log_metric(
                            'correct_pedictions',
                            accumulated_correct_preds.item())
                        accumulated_correct_preds = torch.tensor(0.0)
                        accumulated_valid_preds = torch.tensor(0.0)

                    total_lm_loss += lm_loss

                    # log metrics
                    experiment.log_metric('lm_loss*100', lm_loss.item() * 100)
                    experiment.log_metric('mc_loss*100', mc_loss.item() * 100)
                    loss = loss * accumulation_steps
                    experiment.log_metric('loss*100', loss.item() * 100)
                self.save(f'model-train-epoch{which_epoch}.pt')

        avg_word_loss = total_lm_loss / num_batches
        perplexity = torch.exp(avg_word_loss)
        experiment.log_metric("perplexity", perplexity.item())

        tstr = ''.join(c if c.isdigit() else '-' for c in str(datetime.now()))
        self.save(f'model-train_{tstr}.pt')

    def test_(self, test_dataloader, experiment):
        self.eval()
        total_lm_loss = torch.tensor(0.0)
        total_correct_preds = torch.tensor(0.0)
        total_valid_preds = torch.tensor(0.0)
        with experiment.test():
            with torch.no_grad():
                for num_batches, test_data in tqdm(
                        enumerate(test_dataloader, start=1)):
                    input_ids, attention_mask, token_type_ids, mc_token_ids, \
                        lm_labels, mc_labels = self.__unpack_data(test_data)

                    loss, lm_loss, mc_loss, lm_logits, _ = self(
                        input_ids,
                        mc_labels,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        mc_token_ids=mc_token_ids,
                        lm_labels=lm_labels
                    )

                    accuracy, n_correct_preds, n_valid_preds = self.accuracy(
                        lm_logits, lm_labels)

                    total_lm_loss += lm_loss
                    total_correct_preds += n_correct_preds
                    total_valid_preds += n_valid_preds

                    # log metrics
                    experiment.log_metric('lm_loss', lm_loss.item())
                    experiment.log_metric('mc_loss', mc_loss.item())
                    experiment.log_metric('loss', loss.item())

        avg_word_loss = total_lm_loss / num_batches
        perplexity = torch.exp(avg_word_loss)
        accuracy = total_correct_preds / total_valid_preds

        experiment.log_metric("final_perplexity", perplexity.item())
        experiment.log_metric("final_accuracy", accuracy.item())

    def __top_filtering(self,
                        lm_logits,
                        top_k=0,
                        top_p=0.9,
                        threshold=-math.inf,
                        filter_value=-math.inf):
        """ Filter a distribution of logits using top-k, top-p (nucleus)
                and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                    1d array representing the logits for last word
                top_k: <=0: no filtering, >0: keep only top k tokens with
                    highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of
                    candidates, where S is the smallest subset whose total
                    probability mass is greater than or equal to the threshold
                    top_p.
                    In practice, we select the highest probability tokens whose
                        cumulative probability mass exceeds the threshold
                        top_p.
                threshold: a minimal threshold to keep logits
        """
        # batch size should be 1
        assert lm_logits.dim() == 1

        # top p probability
        if top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(lm_logits,
                                                       descending=True)
            cumulative_probabilities = torch.cumsum(
                softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > top_p
            # Shift the indices to the right to keep also the first token
            #   above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to filter_value
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            lm_logits[indices_to_remove] = filter_value

        # top k elements
        top_k = min(top_k, lm_logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token
            # in the top-k tokens
            indices_to_remove = lm_logits < torch.topk(
                lm_logits, top_k)[0][..., -1, None]
            lm_logits[indices_to_remove] = filter_value

        indices_to_remove = lm_logits < threshold
        lm_logits[indices_to_remove] = filter_value
        return lm_logits

    def __drawword(self, probability_distribution):
        return torch.multinomial(probability_distribution, 1).item()

    def __answer(self,
                 prompt_ids,
                 answer_ids=None,
                 answer_min_len=2,
                 answer_max_len=20,
                 temperature=0.7,
                 top_k=0,
                 top_p=0.9,
                 threshold=-math.inf):
        if answer_ids is None:
            answer_ids = [tokenizer.sep_token_id]
        answer_ids_appearance = defaultdict(int)

        with torch.no_grad():
            for i in range(len(answer_ids), answer_max_len):
                pair_maxlen = len(prompt_ids) + answer_max_len
                max_len = min(pair_maxlen, self.window_size)

                input_ids, attention_mask, token_type_ids, _, mc_token_ids = ChatDataset.encode(
                    prompt_ids.copy(), answer_ids.copy(), max_len=max_len,
                    add_sep_token=False)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                mc_token_ids = mc_token_ids.to(self.device)

                # lm_logits: sequence_length * vocabulary_size
                lm_logits = self.__lm_logits(input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids)
                # lastword_logits: 1 * vocabulary_size
                lastword_logits = lm_logits[-mc_token_ids, :] / temperature
                # lastword_logits: vocabulary_size
                lastword_logits = self.__top_filtering(
                    lastword_logits.squeeze(), top_k, top_p, threshold)
                lastword_probs = softmax(lastword_logits, dim=-1)
                wordid = self.__drawword(lastword_probs)

                num_appearances = answer_ids_appearance[wordid]
                if num_appearances != 0:
                    # make appeared word less frequent
                    for i in range(num_appearances):
                        new_wordid = self.__drawword(lastword_probs)
                        if new_wordid != wordid:
                            wordid = new_wordid
                            break

                if i < answer_min_len and wordid in special_ids:
                    # do not terminate too early
                    while wordid in special_ids:
                        if lastword_probs.max().item() == 1:
                            print("Warning: model generating special token \
                                  with probability 1.")
                            # avoid infinitely looping over special token
                            break
                        wordid = self.__drawword(lastword_probs)

                if wordid in special_ids:
                    break
                answer_ids.append(wordid)
                answer_ids_appearance[wordid] += 1
        return answer_ids

    def answer(self, prompt: str, **kwargs):
        prompt_ids = ChatDataset.str2ids(prompt)
        answer_ids = self.__answer(prompt_ids, **kwargs)
        answer_tokens = tokenizer.convert_ids_to_tokens(
            answer_ids, skip_special_tokens=True)
        answer_str = ' '.join(tokenizer.convert_tokens_to_string(
            [token]).strip() for token in answer_tokens)

        return {
            'ids': answer_ids,
            'tokens': answer_tokens,
            'str': answer_str
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        help='specify where to load dataset (e.g. dataset/dataset.pickle)')
    parser.add_argument(
        '-l', '--load',
        help='specify the model name to be loaded from ./models directory')
    parser.add_argument('-T', '--train', action='store_true',
                        help='run training loop')
    parser.add_argument('-V', '--validate', action='store_true',
                        help='run validation loop')
    parser.add_argument('-t', '--test', action='store_true',
                        help='run testing loop')
    args = parser.parse_args()

    # Make sure you modify the `.comet.config` file
    experiment = Experiment()
    experiment.log_parameters(hyperparams)

    batch_size = hyperparams['batch_size']
    #  Load dataset
    target = args.dataset
    train_dataloader, validate_dataloader, test_dataloader = load_dataset(
        target, batch_size, window_size=hyperparams['window_size'])

    model = Model()

    if args.load:
        model.load(filename=args.load)
    if args.train:
        num_epochs = hyperparams['num_epochs']
        accumulation_steps = hyperparams['accumulation_steps']
        optimizer = transformers.AdamW(
            model.parameters(),
            lr=hyperparams['learning_rate'])
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, 500, 1000)
        model.train_(train_dataloader, optimizer, scheduler, experiment,
                     num_epochs, accumulation_steps)
    if args.validate:
        model.test_(validate_dataloader, experiment)
    if args.test:
        model.test_(test_dataloader, experiment)
