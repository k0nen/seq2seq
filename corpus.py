import re
import unicodedata
import torch

SOS_token = 0
EOS_token = 1
BUF_token = 2

# For faster training, restrict to sentences with 10 words or less
MAX_LENGTH = 10

# CUDA availability
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Corpus:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS", 2: "BUF"}  # Start/End of sentence
		self.n_words = 3

	def add_sentence(self, sentence):
		for word in sentence.split(' '):
			self.add_word(word)

	def add_word(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1


def sentence2index(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]


def sentence2variable(lang, sentence, reverse, pad=0):
	indexes = [SOS_token] + sentence2index(lang, sentence)
	indexes.append(EOS_token)
	if pad > 0:
		indexes += [BUF_token] * (pad - len(indexes))
	if reverse:
		indexes = indexes[::-1]
	result = torch.LongTensor(indexes).view(-1, 1)
	return result


def pair2variable(input_lang, output_lang, pair, pad_input=0, pad_output=0):
	input_variable = sentence2variable(input_lang, pair[0], True, pad_input)
	target_variable = sentence2variable(output_lang, pair[1], False, pad_output)
	return input_variable, target_variable


def unicode2ascii(s):
	# http://stackoverflow.com/a/518232/2809427
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn')


# Trim non-alphabets, to lowercase
def normalize_str(s):
	s = unicode2ascii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s


def read_file(lang1, lang2, name, reverse=False):
	# Read the file and split into lines
	lines = open(name, encoding='utf-8').read().strip().split('\n')
	pairs = [[normalize_str(s) for s in l.split('\t')] for l in lines]

	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		input_lang = Corpus(lang2)
		output_lang = Corpus(lang1)
	else:
		input_lang = Corpus(lang1)
		output_lang = Corpus(lang2)

	return input_lang, output_lang, pairs


def filter_pairs(pairs):
	"""eng_prefixes = (
		"i am ", "i m ",
		"he is", "he s ",
		"she is", "she s",
		"you are", "you re ",
		"we are", "we re ",
		"they are", "they re "
	)"""

	eng_prefixes = ("i", "he", "she", "you", "we", "they")

	def filter_pair(p):
		return len(p[0].split(' ')) < MAX_LENGTH and \
			len(p[1].split(' ')) < MAX_LENGTH and \
			p[1].startswith(eng_prefixes)

	return [pair for pair in pairs if filter_pair(pair)]


def pairs2batch(input_lang, output_lang, pairs):
	max_source = max(s.count(' ') for s, _ in pairs) + 3
	max_target = max(t.count(' ') for _, t in pairs) + 3
	sources, targets = [], []
	for a, b in pairs:
		source, target = pair2variable(input_lang, output_lang, (a, b), max_source, max_target)
		sources.append(source)
		targets.append(target)

	sources = torch.cat(sources, dim=1).to(device)
	targets = torch.cat(targets, dim=1).to(device)
	return sources, targets
