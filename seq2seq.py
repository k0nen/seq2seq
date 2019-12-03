# Implementation of https://arxiv.org/pdf/1409.3215.pdf

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# CUDA availability
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Encoder(nn.Module):
	def __init__(self, source_size, hidden_size, embed_size, n_layers):
		super(Encoder, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		self.embedding = nn.Embedding(source_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, n_layers)

	def forward(self, source):
		# source: [seq length, batch size]
		embedded = self.embedding(source)
		# embedding: [seq length, batch size, embed size]
		output, hidden = self.lstm(embedded)

		# output: [seq length, batch size, hidden size]
		return output, hidden


class Decoder(nn.Module):
	def __init__(self, hidden_size, embed_size, output_size, n_layers):
		super(Decoder, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.embedding = nn.Embedding(output_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, n_layers)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, source, hidden):
		# source: [batch size]
		source = source.unsqueeze(0)
		embedded = F.relu(self.embedding(source))
		# embedded: [1, batch size, embed size]
		output, hidden = self.lstm(embedded, hidden)
		# output: [1, batch size, hidden size]
		output = self.softmax(self.out(output.squeeze(0)))

		# output: [batch size, target corpus size]
		return output, hidden


class Seq2seq(nn.Module):
	def __init__(self, encoder, decoder):
		super(Seq2seq, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		assert(encoder.hidden_size == decoder.hidden_size)
		assert(encoder.n_layers == decoder.n_layers)

	def forward(self, source, target, teacher_forcing_ratio):
		# source: [source seq length, batch size]
		# target: [target seq length, batch size]
		target_len = target.shape[0]
		target_word_size = self.decoder.output_size
		batch_size = target.shape[1]

		# Tensor to store decoder outputs
		outputs = torch.zeros(target_len, batch_size, target_word_size).to(device)
		_, hidden = self.encoder(source)
		decoder_input = target[0, :]  # First input to decoder is SOS tokens

		for i in range(1, target_len):
			output, hidden = self.decoder(decoder_input, hidden)
			outputs[i] = output
			teacher_force = (random.random() < teacher_forcing_ratio)

			top = output.argmax(1)
			decoder_input = target[i] if teacher_force else top

		return outputs
