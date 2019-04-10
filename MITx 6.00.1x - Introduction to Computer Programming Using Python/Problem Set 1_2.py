s = 'azcbobobegbobgbobobhakl'
count = 0
b_index = 0

for b in s:
	if b == 'b':
		b_index = s.index('b',b_index)
		if s[b_index:b_index + 3] == 'bob':
			count += 1
			b_index += 1
		else:
			b_index += 1

print('Number of times bob occurs is:',count)
