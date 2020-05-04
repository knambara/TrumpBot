import argparse
import os
from datetime import datetime

from tqdm import tqdm

import comet_ml
import torch
import transformers
from comet_ml import Experiment
from dataset import load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

hyperparams = {
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 6.25e-5
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
                 device=device,
                 lm_coeff=1.0,
                 mc_coeff=1.0,
                 savedir='models'):
        super().__init__()

        self.model = transformers.GPT2DoubleHeadsModel.from_pretrained('gpt2')

        self.lm_coeff = lm_coeff
        self.mc_coeff = mc_coeff

        self.savedir = savedir
        self.device = device
        self.to(device)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

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
        self.load_state_dict(torch.load(filepath))

    def train_(self,
               train_dataloader: DataLoader,
               optimizer,
               experiment,
               num_epochs):
        self.train()
        total_lm_loss = torch.tensor(0.0)
        num_batches = 0
        with experiment.train():
            for which_epoch in range(num_epochs):
                for train_data in tqdm(train_dataloader):
                    optimizer.zero_grad()
                    num_batches += 1

                    input_ids = train_data['input_ids'].to(self.device)
                    attention_mask = train_data['attention_mask'].to(
                        self.device)
                    token_type_ids = train_data['token_type_ids'].to(
                        self.device)
                    lm_labels = train_data['lm_labels'].to(self.device)
                    mc_labels = train_data['mc_labels'].to(self.device)

                    lm_loss, lm_logits, mc_logits, _ = self(
                        input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        lm_labels=lm_labels)

                    preds = torch.argmax(lm_logits, dim=2)
                    num_valid_preds = torch.sum(lm_labels != -100).double()
                    num_correct_preds = torch.sum((preds == lm_labels))
                    accuracy = (num_correct_preds / num_valid_preds).item()

                    mc_loss, loss = self.loss(lm_loss, mc_logits, mc_labels)
                    loss.backward()
                    optimizer.step()

                    total_lm_loss += lm_loss

                    # log metrics
                    experiment.log_metric('lm_loss', lm_loss)
                    experiment.log_metric('mc_loss', mc_loss)
                    experiment.log_metric('loss', loss)
                    experiment.log_metric('accuracy', accuracy)
                self.save(f'model-train-epoch{which_epoch}.pt')

        avg_word_loss = total_lm_loss / num_batches
        perplexity = torch.exp(avg_word_loss)
        experiment.log_metric("perplexity", perplexity.item())

        tstr = ''.join(c if c.isdigit() else '-' for c in str(datetime.now()))
        self.save(f'model-train_{tstr}.pt')


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
    args = parser.parse_args()

    # Make sure you modify the `.comet.config` file
    experiment = Experiment()
    experiment.log_parameters(hyperparams)

    batch_size = hyperparams['batch_size']
    #  Load dataset
    target = args.dataset
    train_dataloader, validate_dataloader, test_dataloader = load_dataset(
        target, batch_size)

    model = Model()

    if args.load:
        model.load(filename=args.load)
    if args.train:
        num_epochs = hyperparams['num_epochs']
        optimizer = AdamW(
            model.parameters(),
            lr=hyperparams['learning_rate'])
        model.train_(train_dataloader, optimizer, experiment, num_epochs)
