'''
Ho ten: Ton Thien Minh Anh
MSSV: 18110049
Bai thuc hanh Lab07 - Pytorch: NLP From Scratch: Classifying Names with a Character-Level RNN

Tutorial Link: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#evaluating-the-results
'''

# =============== PREPARING THE DATA ===============

# Download & Load Data
from __future__ import unicode_literals,print_function,division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('/home/ton-anh/Python Cho KHDL/Week7/data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters=len(all_letters)

def unicodeToAscii(s):
    '''
    Turn a Unicode string to plain ASCII
    Input: Unicode string
    Output: Plain ASCII string
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c)!= 'Mn'
        and c in all_letters
    )

# Test the unicodeToAscii function
print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of
#  names per language

category_lines = {}
all_categories = []



def readLines(filename):
    '''
    # Read a file and split into lines
    Input: filename
    Output: lines
    '''
    lines = open(filename,encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('/home/ton-anh/Python Cho KHDL/Week7/data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

print(category_lines['Italian'][:5])

# TURNING NAMES INTO TENSORS
import torch

def letterToIndex(letter):
    '''
    Find letter index from all_letters, e.g. "a" = 0
    Input: letter 
    Output: index of letter in all_letters
    '''
    return all_letters.find(letter)

def letterToTensor(letter):
    '''
    For demonstration, turn a letter into a <1 x n_letters> Tensor
    '''
    tensor = torch.zeros(1,n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    '''
    Turn a line into a <line_length x 1 x n_letters>,
    or an array of one-hot letter vectors
    '''
    tensor = torch.zeros(len(line),1,n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())

# =============== CREATING THE NETWORK ===============

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


input = letterToTensor('A')
hidden = torch.zeros(1,n_hidden)

output, next_hidden = rnn(input,hidden)

input = lineToTensor('Albert')
hidden = torch.zeros(1,n_hidden)

output, next_hidden = rnn(input[0],hidden)
print(output)

# =============== TRAINING ===============

# PREPARING FOR TRAINING

def categoryFromOutput(output):
    top_n,top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i],category_i

print(categoryFromOutput(output))

import random 

def randomChoice(l):
    return l[random.randint(0,len(l)-1)]

def randomTrainingExample():
    '''
    Random get a training example (a name and its language)
    '''
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)],dtype = torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor,line_tensor

for i in range(10):
    category,line,category_tensor,line_tensor = randomTrainingExample()
    print('category = ', category, '/ line =',line)

# TRAINING THE NETWORK

criterion = nn.NLLLoss()

learning_rate = 0.005 
# If you set this too high, it might explode. 
# If too low, it might not learn

def train(category_tensor,line_tensor):
    '''
    Train this network is show it a bunch of examples, 
    have it make guesses, and tell it if it’s wrong.

    Each loop of training will:

    + Create input and target tensors
    + Create a zeroed initial hidden state
    + Read each letter in and
        Keep hidden state for next letter
    + Compare final output to target
    + Back-propagate
    + Return the output and loss
    '''
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output,hidden = rnn(line_tensor[i],hidden)

    loss = criterion(output,category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied
    # by learning rate

    for p in rnn.parameters():
        p.data.add_(p.grad.data,alpha=-learning_rate)
    
    return output,loss.item()

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s-= m*60
    return '%dm %ds' %(m,s)

start = time.time()

print('\n ======= TRAINING THE NETWORK ======= \n')

for iter in range(1,n_iters+1):
    category,line,category_tensor,line_tensor = randomTrainingExample()
    output,loss = train(category_tensor,line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every ==0:
        guess,guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter,iter/n_iters *100,timeSince(start),loss,line,guess,correct))

    # Add current loss avg to list of losses
    if iter % plot_every ==0:
        all_losses.append(current_loss/plot_every)
        current_loss = 0


# PLOTTING THE RESULTS

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


# =============== EVALUATING THE RESULTS ===============

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

# RUNNING ON USER INPUT

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')