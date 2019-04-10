s = 'abdgzcdefhmn'
order_1 = ''
order_2 = ''
last_alpha = ''

for alpha in s:

#	print()
#	print('alpha:',alpha)
#	print('last_alpha:',last_alpha)
#	print('order_1:',order_1)
#	print('order_2:',order_2)

	if str(alpha) >= str(last_alpha):
#		print('alpha > last_alpha TRUE')
		order_1 = order_1 + alpha
		last_alpha = alpha
#	elif len(order_1) > len(order_2):
#		print('order_1 > order_2 TRUE')
#		order_2 = order_1
#		order_1 = alpha
#		last_alpha = alpha
	else:
#		print('nada')
		order_1 = alpha
		last_alpha = alpha

	if len(order_1) > len(order_2):
#		print('second order_1 > order_2 TRUE')
		order_2 = order_1
		
print('Longest substring in alphabetical order is:',order_2)
