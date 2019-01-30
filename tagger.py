# -*- coding: utf-8 -*-
"""
=================================POS TAGGER====================================

PROBLEM:
    The model is intended to find part of speech tagger for each and every file 
    in a test file.

    To generate a tag of a word, the program calculates the count
    that a tag follows specific word/phrase. It picks the next maximum
    probability tag depending based on a probability distribution

USAGE:
    The program can be called via a command-line interface.
    It takes 2 arguments i.e. file1 & file2:
        file1 is to take the train file for training the model
        file2 is to take the test file to identify POS tags on it
        
    Both training and test files are provided by the Professor

COMMAND FORMAT:
    python tagger.py file1 file 2 > Tagfile

Tagfile is the file that will be generated that contains the test files words
and also have tags associated with each word

EXAMPLE:
    If the input is as below:
    ... The grand jury commented on a number of
    ... other topics, AMONG them the Atlanta and  
    then the tagger will result in:
    ... The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
    ... other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC

ALGORITHM (Step-by-step):
    1. Takes train and test file from the user in the format as mentioned
    2. Processes the train file - removes the square brackets, remove "/"
        replaces the OR symbols, tokenize it and converts the file in below format:
            [(word1, tag1), (word2, tag2), (word3, tag3),………… (wordN, tagN)]
        and save this list in 'e'
    3. Processes the test file - removes the square brackets, tokenize it and
        converts in the below format:
            [(word1), (word2), (word3),………… (wordN)]
        and save this list in 'test'
    4. We have used classes to train and test the model. The reason for using 
        classes is that we need to incorporate some rules also while training
        the model. So, we have made the model in such a way that when we call 
        one class, it automatically calls the other class ad both the classes
        are used to train the model. To do so, we have passed the name of one 
        class as the object name in other class.
    5. Class 1 i.e. "MostLikely" is the class that calculates the conditional
        probabilities for each tag associated with word from train file and 
        and helps in predicting the most likely tag for the words in test file.
        it starts with train() method that takes the tagged sentences
         - count number of times a word is given each tag
         - select the tag used most often for the word

        predict() first set all the tags for the words in test file as 'NN'
        Before going to prediction() in class 1, it first goes to the class 2
        because the name of class 1 is passed as an abject in class 2. Class 2
        is the classes to implement 5 personalized rules as shown below:
            Rule1: Since, capitalized words are proper nouns, it replaces the 
                    tags to NNP if the word is in title case
            Rule2: Since, words ending in -s are plural nouns, it replaces the 
                    tags to NNS if the word is in plural form
            Rule3: Since, words with an initial digit are numbers, it replaces 
                    the tags to CD if the word is a digit
            Rule4: Since, words with hyphens are adjectives, it replaces the 
                    tags to JJ if the words have hyphen in them
            Rule5: Since, words ending with -ing are gerunds, it replaces the 
                    tags to NNS if the word ends with 'ing'
        After processing the test file with the rules, most of the tags will be
        replaced from 'NN' to the tags associated with the rules.
        Then the program goes to prediction(), it:
        - Procsses the test file after the rules class
        - uses the count table formed after training the train file and
            applies those counts on the test file.
        - see the maximum count of a tag for a particular word
        - assigns that tag to the word
    6. main()
        - Object is created for the class 2, since class 1 name is passed as
            the object name in class 2, therefore, class 1 is also called and 
            predict function is also called as that method is called in 
            class 2.
        - Calls the train method using the object
        - Greets the user
        - calls the prediction method and prints the words and tag as:
            word1/tag1 word2/tag2 word3/tag3 .......... wordN/tagN
  
ACCURACIES:
Before applying the rules:  84.41%
After applying the only rule1: 87.85% 
After applying the rule2 with above rule: 88.92%
After applying the rule3  with above rules: 89.62%
After applying the rule4  with above rules: 89.93%
After applying all the 5 rules: 90.21

Note: The words also include puntuations

AUTHOR NAME: Avneet Pal Kour & Paras Sethi
DATE: Feb 24, 2018
"""

from __future__ import division
import nltk
from collections import defaultdict
import argparse

# TAKING THE FILE INPUTS
parser = argparse.ArgumentParser()
parser.add_argument(dest='file1', type=argparse.FileType('r')) 
parser.add_argument(dest='file2', type=argparse.FileType('r')) 
args = parser.parse_args()
tr = args.file1
tx = args.file2

# READING THE TRAINING FILE
train=tr.read()
# REMOVE SQUARE BRACKETS
train=train.replace('[', '')
train=train.replace(']', '')
# IN TRAIN FILE, WE HAVE INSTANCE LIKE SUMMER/WINTER AND A SINGLE TAG IS GIVEN
# FOR THAT SO WE HAVE REPLACED THE SYMBOL \/ WITH OUR NAME AND LATER WE ARE
# REPLACING IT WITH TAG
train=train.replace('\/', ' Paras ')
train=train.replace('/', ' ')
train=train.replace('|NN', '')
# TOKENIZING THE TRAIN FILE
b=nltk.RegexpTokenizer('\n', gaps=True).tokenize(train)
c=[]
for i in b:
    c.append(nltk.WhitespaceTokenizer().tokenize(i))

e=[]
def my_range(start, end, step):
    while start < end:
        yield start
        start += step
r=len(c)
# CONVERTING THE TRAIN FILE IN THE BELOW FORMAT
# [(word1, tag1), (word2, tag2), (word3, tag3),………… (wordN, tagN)]
for i in my_range(0,r,1):
    y=len(c[i])
    d=[]
    for k in my_range(0,y,1):
        if c[i][k]=='Paras':
            c[i][k]=c[i][k+2]
    for j in my_range(0,y,2):
        f=(c[i][j],c[i][j+1])
        d.append(f)
    e.append(d)

test=tx.read() # READING TEST FILE

# REMOVE SQUARE BRACKETS
test=test.replace('[', '')
test=test.replace(']', '')
test=test.replace('\/', ' ')

# TOKENIZING TEST FILE converts in the below format:
#            [(word1), (word2), (word3),………… (wordN)]
b=nltk.RegexpTokenizer('\n', gaps=True).tokenize(test)
test_sentences=[]
for i in b:
    test_sentences.append(nltk.WhitespaceTokenizer().tokenize(i))

class MostLikely(object):
    def __init__(self):
        self._word_tags = {}
    
    # train() method that takes the tagged sentences
    #     - count number of times a word is given each tag
    #     - select the tag used most often for the word    
    def train(self, tagged_sentences):
        # count number of times a tag follows a word
        count = defaultdict(lambda: defaultdict(lambda: 0))
        for sent in tagged_sentences:
            for word, tag in sent:
                count[word][tag] += 1           
        for word in count:
            tag_count = count[word]
            tag = max(tag_count, key=tag_count.get)
            self._word_tags[word] = tag
        
    def predict(self, test_sentences):
        # predict the tags for each word in the sentence,
        # using the most common tag, or NN if never seen
        return [self._word_tags.get(word, 'NN') for word in test_sentences]
    
    # prediction(), it:
    #    - Procsses the test file after the rules class
    #    - uses the count table formed after training the train file and
    #        applies those counts on the test file.
    #    - see the maximum count of a tag for a particular word
    #    - assigns that tag to the word
    def prediction(self, test_sentences):
        tags = self.predict(test_sentences)
        tagged = [word + '/' + tag for (word, tag) in zip(test_sentences,tags)]
        return ' '.join(tagged)


class Rules(MostLikely):
    def predict(self, test_sentences):
        # super function is used to gain access to inherit method predict () 
        # method from class 1
        tags = super(Rules, self).predict(test_sentences)
        for i, word in enumerate(test_sentences):
            if word not in self._word_tags:
                # Since, capitalized words are proper nouns, it replaces the 
                #    tags to NNP if the word is in title case
                if word.istitle():
                    tags[i] = 'NNP'
                # Since, words ending in -s are plural nouns, it replaces the 
                #    tags to NNS if the word is in plural form
                elif word.endswith('s'):
                    tags[i] = 'NNS'
                # Since, words with an initial digit are numbers, it replaces 
                #    the tags to CD if the word is a digit
                elif word[0].isdigit():
                    tags[i] = 'CD'
                # Since, words with hyphens are adjectives, it replaces the 
                #    tags to JJ if the words have hyphen in them
                elif '-' in word:
                    tags[i] = 'JJ'
                # Since, words ending with -ing are gerunds, it replaces the 
                #    tags to NNS if the word ends with 'ing'
                elif word.endswith('ing'):
                    tags[i] = 'VBG'              
        return tags

def main():
    # Creating object of Rules() class
    MostLikely=Rules()
    # Training the model
    MostLikely.train(e)
    for i in test_sentences:
        print(MostLikely.prediction(i)) #predicting the tags of the test file
main()