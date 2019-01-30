# POS-Tagger

Problem to Solve:

Write a python program called tagger.py which will take as input a training file containing part of speech tagged text, and a file containing text to be part of speech tagged. the program should implement the "most likely tag" baseline.

Note that this assignment is based on problem 5.6 from page 171 of JM. For each word in the training data, assign it the POS tag that maximizes p(tag|word). Assume that any word found in the test data but not in training data (i.e. an unknown word) is an NN and find the accuracy of the most likely tagger on a given test file. Record that accuracy in the overview comment, and then add at least 5 rules to the tagger and see how those rules affect the accuracy. Make certain to also include the rules you add and the resulting accuracy in the overview comment as well.

The input for this assignment is found in the files section of the web site (PA3.zip). The training data is pos-train.txt, and the text to be tagged is pos-test.txt. There is also a gold standard (manually tagged) version of the test file found in pos-test-key.txt that you will use to evaluate the tagged output.

Here's an example of how the tagger.py program should be run from the command line. Note that the program output should go to STDOUT, so the file named used below could be anything. This program will learn the most likely tag tagger from the train data, and then tag the test file based on that model. 

$ python tagger.py pos-train.txt pos-test.txt > pos-test-with-tags.txt

Note that the tagger should not modify pos-test.txt in any way, and that the output of the program should make certain to handle each tagged item in the test data. You will note that in both the training and test data phrases are enclosed in brackets [] - those indicate phrasal boundaries, and you may ignore these since we don't use them in POS tagging.

You should also write a utility program called scorer.py which will take as input the POS tagged output and compare it with the gold standard "key" data which I have placed in the Files section of our group (pos-test-key.txt). the scorer program should report the overall accuracy of the tagging,
and provide a confusion matrix. Again, this program should write output to STDOUT.

The scorer program should be run as follows:
$ python scorer.py pos-test-with-tags.txt pos-test-key.txt > pos-taggingreport.txt

Note that if the accuracy is unusually low (less than the most likely tag baseline) that is a sign there is a significant problem in the tagger, and you should work to resolve that before submission.
