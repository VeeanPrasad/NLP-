from collections import Counter
from collections import defaultdict
import numpy as np

# extract the lines from the training set
text = open("hobbit-train.txt").read()
print(len(text))

# split the sentence to form words on white spaces
words = text.split()
print(len(words))

# Counter is used to obtain the frequencies of the words
# A dictionary is formed with keys as items and values are the frequency in the collection
rawfreqs = Counter(words)
print(len(rawfreqs))

print(rawfreqs["Gandalf"])

# defaultdic to create values for new entries to the index using the method
# word2vec is a dictionary with keys as strings and values as integer index for each string
# based on the order of word exploration
word2index = defaultdict(lambda: len(word2index))
UNK = word2index["<UNK>"]
# print(UNK)


[word2index[word] for word, freq in rawfreqs.items() if freq > 1]
# we ignore the words that happened once in training

#print("The length of words has reduced from ",len(rawfreqs))
#print("to",len(word2index))

# we change the word2index to map new words to UNK and not add them to the lexicon
word2index = defaultdict(lambda: UNK, word2index)


# for Trigram model we need all the trigram words from the corpus
trigram_list = []
for i in range(1, len(words)):
    trigram_list.append((words[i-2], words[i-1],words[i]))
# print(trigram_list)

bigram_list = []
for i in range(1, len(words)):
    bigram_list.append((words[i-1],words[i]))
# print(bigram_list)

w2i = len(word2index)
###  UNIGRAM
unigrams = [word2index[word] for word in words]
unigramFreqs = Counter(unigrams)
#print(len(unigrams))
def unigramProb(unigram):
    return np.log((unigramFreqs[unigram] + 1 ) / (sum(unigramFreqs.values()) + w2i))

def unigramSentenceProb(words):
    return sum(([unigramProb(word2index[word]) for word in words]))

###  BIGRAM
bigrams = [ (word2index[words[i-1]], word2index[words[i]]) for i in range(1, len(words)) ]
bigramFreqs = Counter(bigrams)
#print(len(bigrams))
#print(bigramFreqs.items())

def bigramProb(bigram):
    return np.log((bigramFreqs[bigram] + 1) / (unigramFreqs[bigram[0]] + (w2i)))

def bigramSentenceProb(words):
    return (sum(([bigramProb((word2index[words[i-1]], word2index[words[i]])) for i in range(1, len(words))])))

### Trigrams
trigrams = [ (word2index[words[i-2]],word2index[words[i-1]], word2index[words[i]]) for i in range(2, len(words)) ]
trigramFreqs = Counter(trigrams)
#print(len(trigrams))
#print(trigramFreqs.items())

def trigramProb(trigram):
    return np.log((trigramFreqs[trigram] + 1) / (bigramFreqs[trigram[0],trigram[1]] + (w2i)))

def trigramSentenceProb(words):
    uni = unigramProb(word2index[words[0]])
    bi = bigramProb((word2index[words[0]],word2index[words[1]]))
    tri = (sum(([trigramProb((word2index[words[i-2]], word2index[words[i-1]], word2index[words[i]])) for i in range(2, len(words))])))

    return uni + bi + tri

def perplexity(logprob,length):
    perplexity = np.exp(-logprob / length)
    return perplexity


## For testing
test1 = "In a hole in the ground"
## Unigram
print("Unigram testing")
logprobuni = unigramSentenceProb(test1.split())
print("Prob: ",logprobuni)
print("Perplexity: ",perplexity(logprobuni,len(test1)))

def testuni(test):
    logprobuni = unigramSentenceProb(test)
    n = len(test)
    perplexuni = perplexity(logprobuni,n)
    return  logprobuni, perplexuni

## Bigram
print("Bigram testing")
logprobbi= bigramSentenceProb(test1.split())
print("Prob: ",logprobbi)
print("Perplexity: ",perplexity(logprobbi,len(test1)))

def testbi(test):
    logprobbi = bigramSentenceProb(test)
    n = len(test)
    perplexbi = perplexity(logprobbi,n)
    return  logprobbi, perplexbi

## Trigram
print("Trigram testing")
logprobtri = trigramSentenceProb(test1.split())
print("Prob: ", logprobtri)
print("Perplexity: ",perplexity(logprobtri,len(test1)))

def testtri(test):
    logprobtri = trigramSentenceProb(test)
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
    testing()

if __name__== "__main__":
    main()



















