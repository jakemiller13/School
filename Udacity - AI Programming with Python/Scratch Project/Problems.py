points = 204  # use this input to make your submission

# write your if statement here

if points >= 1 and points <= 50:
    prize = 'wooden rabbit'
elif points >= 51 and points <= 150:
    prize = 'no-prize'
elif points >= 151 and points <= 180:
    prize = 'wafer-thin mint'
elif points >= 181 and points <= 200:
    prize = 'penguin'

try:
    print('Congratulations! You won a ' + prize)
except NameError:
    print('Oh dear, no prize this time.')