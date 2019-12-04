import torch
import torch.nn as nn
import pickle
import random

import corpus
import seq2seq
import train

# Options
mode = 'test'
load_corpus = False
corpus_path = 'data/corpus_small.dat'
load_model = True
model_path = 'data/model_small_save.pt'

# Hyperparameters
hidden_size = 256
embed_size = 256
n_layers = 1
n_epochs = 1000
batch_size = 16
clip = 1
learning_rate = 0.01  # See half-life in train.train()
teaching_rate = 0.5

print(f'{hidden_size} {embed_size} {n_layers} {n_epochs} {batch_size} {model_path}')

if __name__ == '__main__':
	# CUDA availability
	use_cuda = torch.cuda.is_available()
	device = torch.device('cuda' if use_cuda else 'cpu')

	# Load corpus
	if load_corpus:
		with open(corpus_path, 'rb') as f:
			input_lang, output_lang, pairs = pickle.load(f)
	else:
		input_lang, output_lang, pairs = corpus.read_file(
			'ENG', 'FRA', 'data/eng-fra.txt', True)
		pairs = corpus.filter_pairs(pairs)
		for pair in pairs:
			input_lang.add_sentence(pair[0])
			output_lang.add_sentence(pair[1])
		with open(corpus_path, 'wb') as f:
			pickle.dump((input_lang, output_lang, pairs), f)
	print(f'{len(pairs)} pairs, {input_lang.n_words} source, {output_lang.n_words} target')

	# Load model
	encoder = seq2seq.Encoder(input_lang.n_words, hidden_size, embed_size, n_layers)
	decoder = seq2seq.Decoder(hidden_size, embed_size, output_lang.n_words, n_layers)
	model = seq2seq.Seq2seq(encoder, decoder).to(device)
	if load_model:
		model.load_state_dict(torch.load(model_path))
		# train.train(model, (input_lang, output_lang, pairs), batch_size, n_epochs, learning_rate,
		#						teaching_rate, clip, model_path)
	else:
		def init_weights(m):
			for name, param in m.named_parameters():
				nn.init.uniform_(param.data, -0.08, 0.08)
		model.apply(init_weights)

	# Test or train
	if mode == 'train':
		train.train(model, (input_lang, output_lang, pairs), batch_size, n_epochs, learning_rate,
								teaching_rate, clip, model_path)
	else:
		model.load_state_dict(torch.load(model_path))
		model.eval()

		random.shuffle(pairs)
		set_size = int(len(pairs) * 0.9) // batch_size * batch_size
		train_set, valid_set = pairs[:set_size], pairs[set_size:]
		valid_batch = [corpus.pairs2batch(input_lang, output_lang, valid_set[i:i + batch_size])
									 for i in range(0, len(valid_set), batch_size)]
		print('load ok')
		train.sample(model, random.sample(valid_batch, 1)[0], output_lang)
