# Hangman game
#

# -----------------------------------
# Helper code
# You don't need to understand this helper code,
# but you will have to know how to use the functions
# (so be sure to read the docstrings!)

import random

WORDLIST_FILENAME = "words.txt"

def loadWords():
    """
    Returns a list of valid words. Words are strings of lowercase letters.
    
    Depending on the size of the word list, this function may
    take a while to finish.
    """
    print("Loading word list from file...")
    # inFile: file
    inFile = open(WORDLIST_FILENAME, 'r')
    # line: string
    line = inFile.readline()
    # wordlist: list of strings
    wordlist = line.split()
    print("  ", len(wordlist), "words loaded.")
    return wordlist

def chooseWord(wordlist):
    """
    wordlist (list): list of words (strings)

    Returns a word from wordlist at random
    """
    return random.choice(wordlist)

# end of helper code
# -----------------------------------

# Load the list of words into the variable wordlist
# so that it can be accessed from anywhere in the program
wordlist = loadWords()

def isWordGuessed(secretWord, lettersGuessed):
    
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: boolean, True if all the letters of secretWord are in lettersGuessed;
      False otherwise
    '''

    userLetter = input("Please guess a letter: ")

    if userLetter in secretWord and userLetter not in lettersGuessed:
        secretWordDict.pop(userLetter)
        lettersGuessed = lettersGuessed.extend(userLetter)
        getGuessedWord(secretWord, userLetter)
        print("Good guess:" + getGuessedWord(secretWord))
    elif userLetter in lettersGuessed:
        print("Oops! You've already guessed that letter: " + getGuessedWord(secretWord, userLetter))
    else:
        lettersGuessed = lettersGuessed.extend(userLetter)
        print("Oops! That letter is not in my word: " + getGuessedWord(secretWord, userLetter))
        guesses -= 1
    
    if guesses == 0:
        return "Sorry, you ran out of guesses. The word was " + secretWord + "."
    
    if secretWordDict == {}:
        return True
    else:
        return False

def getGuessedWord(secretWord, userLetter):
    
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters and underscores that represents
      what letters in secretWord have been guessed so far.
    '''
    
    for j in range(len(secretWord)):
        if secretWord[j] == userLetter:
            blankSecretWord[j] = userLetter
                
    return ' '.join(blankSecretWord)

def getAvailableLetters(lettersGuessed):
    
    '''
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters that represents what letters have not
      yet been guessed.
    '''
    
    if userLetter in remainingLetters:
        print("Good guess: " + getGuessedWord(secretWord))
        remainingLetters.remove(letter)
    elif userLetter in lettersGuessed:
        print("Oops! You've already guessed that letter: " + getGuessedWord(secretWord))
    else:
        print("Oops! That letter is not in my word: " + getGuessedWord(secretWord))
    
    return "Available letters: " + ''.join(remainingLetters)

def hangman(secretWord):
    
    '''
    secretWord: string, the secret word to guess.

    Starts up an interactive game of Hangman.

    * At the start of the game, let the user know how many 
      letters the secretWord contains.

    * Ask the user to supply one guess (i.e. letter) per round.

    * The user should receive feedback immediately after each guess 
      about whether their guess appears in the computers word.

    * After each round, you should also display to the user the 
      partially guessed word so far, as well as letters that the 
      user has not yet guessed.

    Follows the other limitations detailed in the problem write-up.
    '''

    #Welcome message
    print("Welcome to the game, Hangman!")
    print("I am thinking of a word that is " + str(len(secretWord)) + " letters long.")
    
    #Define function-wide variables
    guesses = 8 #initialize number of guesses
    secretWordDict = {} #dictionary for secretWord
    lettersGuessed = [] #list to keep track of letters guessed
    blankSecretWord = list('_' * len(secretWord)) #underscores for each letter in secretWord
    
    #Turns secretWord into dictionary to easily store letters
    for i in secretWord:
        secretWordDict[i] = i

    #Initializes remainingLetters so it doesn't have to be imported each time
    import string
    remainingLetters = list(string.ascii_lowercase)
    
    while isWordGuessed(secretWord, lettersGuessed) == False:        
        print("You have " + guesses + "left.")







# When you've completed your hangman function, uncomment these two lines
# and run this file to test! (hint: you might want to pick your own
# secretWord while you're testing)

secretWord = chooseWord(wordlist).lower()
hangman(secretWord)
