from collections import Counter
from collections import defaultdict
import numpy as np
import re
import random

## Tokenization
def Tok(filename):
    # extract the lines from the training set
    text = open("hobbit-train.txt").read()

    # split the sentence to form words on period
    sents = text.split('.')
    out = []

    for i in sents:
        l=(re.sub('\n',' ',i))
        out.append(l)
    # Removing the header section of the corpus
    out = out[3:]
    # Removing the author and chapter details of the first sentence of the corpus
    out[0] = out[0].replace(' TOLKIEN   Chapter I   AN UNEXPECTED PARTY   ', '')
    return out

#############################################################################
## Segmentation
def Seg(corpus):
    # Adding start <s> and end of sentence marker </s>
    out = []
    for i in corpus:
        l=' <s> ' + i + ' </s> '
        out.append(l)
    # Creating a Hamlet corpus with segmentation
    return  ''.join(out)

corpus = Tok("hobbit-train.txt")
sentenses = Seg(corpus)
word1 = sentenses.split()


# Counter is used to obtain the frequencies of the words
# A dictionary is formed with keys as items and values are the frequency in the collection
rawfreqs = Counter(word1)
# print(len(rawfreqs))

# print(rawfreqs["Gandalf"])

# defaultdic to create values for new entries to the index using the method
# word2vec is a dictionary with keys as strings and values as integer index for each string
# based on the order of word exploration
word2index = defaultdict(lambda: len(word2index))
# UNK = word2index["<UNK>"]
# print(UNK)


[word2index[word] for word, freq in rawfreqs.items() if freq > 1]
# we ignore the words that happened once in training

#print("The length of words has reduced from ",len(rawfreqs))
#print("to",len(word2index))

# we change the word2index to map new words to UNK and not add them to the lexicon
# word2index = defaultdict(lambda: UNK, word2index)


# for Trigram model we need all the trigram words from the corpus
trigram_list = []
for i in range(1, len(word1)):
    trigram_list.append((word1[i-2], word1[i-1],word1[i]))


bigram_list = []
for i in range(2, len(word1)):
    bigram_list.append((word1[i-1],word1[i]))


w2i = len(word2index)

## Word that is found at the index
def i2w(i):
    a = list(word2index.keys())
    b = list(word2index.values())
    return a[b.index(i)]

########################################################################
###  UNIGRAM
unigrams = [word2index[word] for word in word1]
unigramFreqs = Counter(unigrams)

def unigramProb(unigram, k = 1):
    return ((unigramFreqs[unigram] + k ) / (sum(unigramFreqs.values()) + k*w2i))

def unigramSentenceProb(words):
    return sum(([unigramProb(word2index[word]) for word in words]))

##########################################################################
###  BIGRAM
bigrams = [ (word2index[word1[i-1]], word2index[word1[i]]) for i in range(1, len(word1)) ]
bigramFreqs = Counter(bigrams)

def bigramProb(bigram,k = 1):
    return ((bigramFreqs[bigram] + k) / (unigramFreqs[bigram[0]] + k*w2i))

def logbigramProb(bigram,k = 1):
    return np.log((bigramFreqs[bigram] + k) / (unigramFreqs[bigram[0]] + k*w2i))

def bigramSentenceProb(words):
    return (sum(([bigramProb((word2index[words[i-1]], word2index[words[i]])) for i in range(1, len(words))])))

##########################################################################
### Trigrams
trigrams = [ (word2index[word1[i-2]],word2index[word1[i-1]], word2index[word1[i]]) for i in range(2, len(word1)) ]
trigramFreqs = Counter(trigrams)

def trigramProb(trigram , k =1):
    return ((trigramFreqs[trigram] + k) / (bigramFreqs[trigram[0],trigram[1]] + k*w2i))
def logtrigramProb(trigram, k = 1):
    return np.log((trigramFreqs[trigram] + k) / (bigramFreqs[trigram[0],trigram[1]] + k*w2i))

def trigramSentenceProb(words):
    uni = unigramProb(word2index[words[0]])
    bi = bigramProb((word2index[words[0]],word2index[words[1]]))
    tri = (sum(([trigramProb((word2index[words[i-2]], word2index[words[i-1]], word2index[words[i]])) for i in range(2, len(words))])))

    return uni + bi + tri

def trigramlogSentenceProb(words):
    uni = unigramProb(word2index[words[0]])
    bi = logbigramProb((word2index[words[0]],word2index[words[1]]),1)
    tri = (sum(([logtrigramProb((word2index[words[i-2]], word2index[words[i-1]], word2index[words[i]]),1) for i in range(2, len(words))])))

    return uni + bi + tri

###############################################################################
def perplexity(logprob,length):
    perplexity = np.exp(-logprob / length)
    return perplexity

################################################################################

def BeginBigram():
    temp = [ (i, bigramProb(i ,1)) for i in bigrams if i[0] == word2index['<s>']]

    a = [i[0] for i in temp]
    b = [i[1] for i in temp]
    pick = random.choices(a,b)
    return pick

def BeginTrigram(bigram):
    temp = [(i, trigramProb(i,1)) for i in trigrams if i[0] == bigram[0][0] and i[1] == bigram[0][1]]
    a = [i[0] for i in temp]
    b = [i[1] for i in temp]
    pick = random.choices(a,b)
    return pick

##############################################################################
# shannon algo
def SGG():
     temp = []
     begin = BeginBigram()
     temp.append(i2w(begin[0][1]))

    # continue until we reach end of sentc marker
     while (begin[0][1] != word2index['</s>']):
         trigram_pick = BeginTrigram(begin)
         begin = [(trigram_pick[0][1],trigram_pick[0][2])]
         temp.append(i2w(trigram_pick[0][2]))

     ## Adding the start marker to the beginning of sentence marker
     marker_sentence = ['<s>']
     marker_sentence.extend(temp)
     return marker_sentence

## SGG Testing
## Generate 50 sentences and calculate their perplexity
def SGGTest():
    sequence = []
    print("List of random sentence and their perplexities")
    f = open("prasad-veena-text-part2.txt", "w")
    for i in range(0,50):
        sentence = SGG()
        logprobtri, perplextri = testtri(sentence)
        print(' '.join(sentence[1:-1])," ----> ",perplextri)
        f.write("{} ---->  {}\n".format(' '.join(sentence[1:-1]),(perplextri)))
        sequence.extend(sentence)

    finallogprobtri, finalperplextri = testtri(sequence)
    print("Perplexity of the entire sequence ",finalperplextri)
    f.write("Perplexity of the entire sequence ---> {}\n".format(finalperplextri))
    #### Result
    #### Perplexity of the entire sequence ---> 1221.5857327343342
    f.close()

##################################################################################
## Trigram

def testtri(test):
    logprobtri = trigramlogSentenceProb(test)
    n = len(test)
    perplextri = perplexity(logprobtri,n)
    return  logprobtri, perplextri

def testing():
    hw_test = open("hw-test.txt").read()
    tests = hw_test.split('\n')
    print("Length of test data = ", len(tests))
    f = open("prasad-veena-test.txt","w")
    print("Results")
    for test in tests:
        line = test.split()
        logprobuni, perplexuni = testuni(line)
        # print(logprobuni + " " +  perplexuni)
        logprobbi, perplexbi = testbi(line)
        # print(logprobbi + " " + perplexbi)
        logprobtri,  perplextri = testtri(line)
        ## Trigram language model results
        print(logprobtri,"  ", perplextri)
        f.write("{}\n".format(perplextri))
    f.close()

def main():
    # testing()
    SGGTest()

if __name__== "__main__":
    main()



















