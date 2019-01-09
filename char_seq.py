import torch
import random
from torch.nn import Embedding, Linear, Module, LSTM, NLLLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn.functional import relu, log_softmax, softmax
from numpy.random import choice
import numpy as np

SEQ_LEN = 100
BATCH_SIZE = 128
HIDDEN_SIZE = 48
DATASET = 'data_medium.txt'
EPOCHS = 100
LR = 1e-2
GEN_LEN = 30

torch.manual_seed(0)
probability_distribution = np.array([0.08167, 0.01492, 0.02782, 0.04253, 0.12702,
    0.02228, 0.02015, 0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406,
    0.06749, 0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758,
    0.00978, 0.02360, 0.00150, 0.01974, 0.00074])
probability_distribution /= sum(probability_distribution)
letters = list('abcdefghijklmnopqrstuvwxyz')


class Model(Module):

    def __init__(self, out_size):
        super().__init__()
        self.character_embedding = Embedding(out_size, HIDDEN_SIZE)
        self.lstm = LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
        self.linear = Linear(HIDDEN_SIZE, out_size)

    def forward(self, input):
        embed = self.character_embedding(input)
        step, _ = self.lstm(embed)
        step = log_softmax(self.linear(step), dim=2)
        return step

    def generate(self, last_token, hidden=None):
        embed = self.character_embedding(last_token)
        step, hidden = self.lstm(embed, hidden)
        step = softmax(self.linear(step), dim=2)
        return step, hidden


class Characters():

    def __init__(self):
        self._char_indices = {}
        self._index_chars = {}
        self._next_index = 0

    def add_char(self, char):
        if char not in self._char_indices:
            self._char_indices[char] = self._next_index
            self._index_chars[self._next_index] = char
            self._next_index += 1

        return self._char_indices[char]

    def __getitem__(self, index):
        return self._index_chars[index]

    def __len__(self):
        return len(self._char_indices)


class CharacterDataset(Dataset):

    def __init__(self):
        with open(DATASET, 'r') as content_file:
            self.content = [list(word) for line in content_file for word in line.strip().lower()]

        dataset = []
        self.characters = Characters()
        for word in self.content:
            dataset.append(self.characters.add_char(word[0]))

        self.dataset = torch.Tensor(dataset).long()

    @property
    def vocab_size(self):
        return len(self.characters)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        data = self.dataset[index:index + SEQ_LEN]
        return data

    def __len__(self):
        return len(self.content) - BATCH_SIZE


class Trainer():

    def __init__(self, num_epochs):

        self.criterion = NLLLoss()
        self.dataset = CharacterDataset()
        dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, **dataloader_kwargs)

        self.model = Model(self.dataset.vocab_size).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=LR)

        self.epoch = 0
        self.num_epochs = num_epochs
        self.total_loss = 0.0

    def train(self):
        for self.epoch in range(self.epoch + 1, self.num_epochs + 1):
            self.total_loss = 0.0
            for batch_idx, batch in enumerate(self.dataloader):
                batch = batch.to(device)
                self.optimizer.zero_grad()

                output = self.model(batch)
                loss = self.criterion(output[:, :-1].reshape(-1, self.dataset.vocab_size), batch[:, 1:].reshape(-1))

                loss.backward()
                self.optimizer.step()
                self.total_loss += loss.item()
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * BATCH_SIZE, len(self.dataloader.dataset),
                    100. * batch_idx / len(self.dataloader), loss.item()))

            if self.epoch % 10 == 0:
                print(f'> {self.generate()}')

            print('====> Train set loss: {:.4f}'.format(self.total_loss / len(self.dataloader)))

    def sample_letter(self):
        draw = choice(letters, 1, p=probability_distribution)[0]
        return self.dataset.characters.add_char(draw)

    def generate(self):
        last_token = torch.tensor([[self.sample_letter()]], dtype=torch.long)
        sequence = [self.dataset.characters[last_token.item()]]
        hidden = None
        for _ in range(GEN_LEN):
            output, hidden = self.model.generate(last_token, hidden)
            _, last_token = output.topk(1)
            last_token = last_token.squeeze(0)
            sequence.append(self.dataset.characters[last_token.item()])
        return ''.join(sequence)



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = Trainer(EPOCHS)
    t.train()
    t.generate()
