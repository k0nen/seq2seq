import time
import math


def time2str(s):
	m = math.floor(s / 60)
	s -= int(m * 60)
	return f'{m}m {int(s)}s'
