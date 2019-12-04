import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

import corpus
import util


def train_epoch(model, batch_list, optimizer, criterion, teaching_rate, clip):
	model.train()
	epoch_loss = 0

	for i, batch in enumerate(batch_list):
		source, target = batch
		optimizer.zero_grad()
		output = model(source, target, teaching_rate)

		# output: [target length, batch size, target corpus size]
		# target: [target length, batch size]

		# Reshape, ignore first row (See Seq2seq.forward())
		output_dim = output.shape[-1]
		output = output[1:].view(-1, output_dim)
		target = target[1:].view(-1)

		loss = criterion(output, target)
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()
		epoch_loss += loss.item()

	return epoch_loss / len(batch_list)


def evaluate(model, batch_list, criterion):
	model.eval()
	epoch_loss = 0
	with torch.no_grad():
		for batch in batch_list:
			source, target = batch
			output = model(source, target, 0)

			# output: [target length, batch size, target corpus size]
			# target: [target length, batch size]

			# Reshape, ignore first row (See Seq2seq.forward())
			output_dim = output.shape[-1]
			output = output[1:].view(-1, output_dim)
			target = target[1:].view(-1)

			loss = criterion(output, target)
			epoch_loss += loss.item()

	return epoch_loss / len(batch_list) if len(batch_list) > 0 else -1


def sample(model, batch, output_lang):
	model.eval()
	with torch.no_grad():
		source, target = batch
		output = model(source, target, 0)

		# output: [target length, batch size, target corpus size]
		# target: [target length, batch size]

		# Reshape, ignore first row (See Seq2seq.forward())
		output_dim = output.shape[-1]
		output = output[1:].argmax(2).t()
		target = target[1:].t()

		for i in range(len(output)):
			print(f'Output: {" ".join(output_lang.index2word[a] for a in output[i].tolist())}', end=' ')
			print(f'Target: {" ".join(output_lang.index2word[a] for a in target[i].tolist())}')


def train(model, dataset, batch_size, n_epoch, learning_rate, teaching_rate,
					clip, save_path):
	input_lang, output_lang, pairs = dataset

	best_valid_loss = float('inf')
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss(ignore_index=corpus.BUF_token)
	begin = time.time()

	set_size = int(len(pairs) * 0.9) // batch_size * batch_size
	random.shuffle(pairs)
	train_set, valid_set = pairs[:set_size], pairs[set_size:]
	valid_batch = [corpus.pairs2batch(input_lang, output_lang, valid_set[i:i+batch_size])
									for i in range(0, len(valid_set), batch_size)]

	for i in range(1, n_epoch + 1):
		random.shuffle(train_set)
		train_batch = [corpus.pairs2batch(input_lang, output_lang, train_set[i:i+batch_size])
										for i in range(0, len(train_set), batch_size)]

		train_loss = train_epoch(model, train_batch, optimizer, criterion, teaching_rate, clip)
		valid_loss = evaluate(model, valid_batch, criterion)
		sample(model, random.sample(valid_batch, 1)[0], output_lang)
		end = time.time()

		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			torch.save(model.state_dict(), save_path)

		if i % 5 == 0 and i <= 30:
			# learning_rate /= 2
			optimizer = optim.SGD(model.parameters(), lr=learning_rate)

		print(f'Epoch {i}({util.time2str(end - begin)}):', end=' ')
		print(f'Train {train_loss:.3f} | Valid {valid_loss:.3f}\n')
