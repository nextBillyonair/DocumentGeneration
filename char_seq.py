import torch
import random
from torch.nn import Embedding, Linear, Module, LSTM, NLLLoss
from torch.optim import Adam
from torch.nn.functional import relu, log_softmax

class Generator(Module):

    def __init__(self, out_size):
        super().__init__()
        self.character_embedding = Embedding(out_size, 48)
        self.lstm = LSTM(48, 48)
        self.linear = Linear(48, out_size)

    def forward(self, input, hidden):
        embed = self.character_embedding(input)
        step, hidden = self.lstm(embed, hidden)
        step = log_softmax(self.linear(step), dim=2)
        return step, hidden

class Characters():

    def __init__(self):
        self._char_indices = {}
        self._index_chars = {}
        self._next_index = 0

        self._add_char('<SOW>')
        self._add_char('<EOW>')
        self.add_word(list("abcdefghijklmnopqrstuvwxyz"))

    def add_word(self, word):
        indices = tuple(self._add_char(char) for char in list(word))
        assert len(indices) > 0, 'empty sentence'
        return indices

    def _add_char(self, char):
        if char not in self._char_indices:
            self._char_indices[char] = self._next_index
            self._index_chars[self._next_index] = char
            self._next_index += 1

        return self._char_indices[char]

    def __len__(self):
        return len(self._char_indices)

class Model():

    def __init__(self, out_size):
        self.model = Generator(out_size)

    def parameters(self):
        return self.model.parameters()

    def train(self, word, optimizer, criterion):
        optimizer.zero_grad()
        loss = 0.0

        last_token = torch.full((1,1), 0, dtype=torch.long)
        hidden = None
        for i in range(word.size(0)):
            output, hidden = self.model(last_token, hidden)
            _, most_likely_token_indices = output.topk(1)
            last_token = most_likely_token_indices.squeeze(1).detach()
            print(last_token, word[i].unsqueeze(0))

            if random.random() < 1.0:
                # print(last_token, word[i].unsqueeze(0).unsqueeze(0))
                last_token = word[i].unsqueeze(0).unsqueeze(0)
            loss += criterion(output.view(1, -1), word[i].unsqueeze(0))

        # Backprop
        loss.backward()
        optimizer.step()

        return loss.item()

class Dataset():

    def __init__(self):
        with open('data_small.txt', 'r') as content_file:
            self.content = [list(word) for word in content_file.read().lower().strip('\n').replace('\n', ' ').split(' ') if len(list(word)) > 0 ]

    def training_batches(self):
        for word in self.content:
            yield word

class Trainer():

    def __init__(self, num_epochs):
        self.criterion = NLLLoss()
        self.characters = Characters()
        self.dataset = Dataset()
        for data in self.dataset.training_batches():
            self.characters.add_word(data)
        self.model = Model(len(self.characters))
        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        self.epoch = 0
        self.num_epochs = num_epochs
        self.total_loss = 0.0

    def train(self):
        for self.epoch in range(self.epoch + 1, self.num_epochs + 1):
            for batch in self.dataset.training_batches():
                batch = torch.tensor(self.characters.add_word(batch))
                loss = \
                    self.model.train(batch, self.optimizer, self.criterion)
                self.total_loss += loss
            print(self.total_loss)
            self.total_loss = 0.0



t = Trainer(1000)
t.train()












#
