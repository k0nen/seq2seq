filename = 'data/model_small.txt'
with open(filename, 'r') as f:
	lines = f.readlines()

data = [lines[i].strip() for i in range(5, len(lines), 4)]
data = [(
	float(line[line.index('Train')+6:line.index('Train')+11]),
	float(line[line.index('Valid')+6:line.index('Valid')+11])
	) for line in data]

for a, b in data:
	print(a, b)
