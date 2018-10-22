import storm
import re
import csv


from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import words as nltk_words

#start process_tweet
def processTweet(tweet):
    # process the tweets
    tweet = ''.join([i if ord(i) < 128 else ' ' for i in tweet]) 
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet



	#initialize stopWords


	#start replaceTwoOrMore
def replaceTwoOrMore(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
	#end

	#start getStopWordList
def getStopWordList(stopWordListFileName):
     #read the stopwords file and build a list
     stopWords = []
     fp = open(stopWordListFileName, 'r')
     line = fp.readline()
     while line:
         word = line.strip()
         stopWords.append(word)
         line = fp.readline()
     fp.close()
     return stopWords
	

	#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
        #split tweet into words
    stopWords = []
    stopWords = getStopWordList('E:/(3) Semester 2 July 2018\COMP90019_Distributed Computing Project/From fahmin/CreatingSample/data/stopwords.txt')
    words = tweet.split()
    for w in words:
            #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
            #strip punctuation
        w = w.strip('\'"?,.')
            #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
            #ignore if it is a stop word
        if(w in stopWords or val is None ):
            continue
        else:
            w = snowball_stemmer.stem(w)
            featureVector.append(w.lower())
    return featureVector
	#end

class tweetRead(storm.BasicBolt):

   
    


    def initialize(self, conf, context):
        self._conf = conf
        self._context = context
        self._output = open('E:/(3) Semester 2 July 2018/COMP90019_Distributed Computing Project/From fahmin/CreatingSample/outputfile.txt', 'w')

    def process(self, tup):
        line = tup.values[0]
        output = open('E:/(3) Semester 2 July 2018/COMP90019_Distributed Computing Project/From fahmin/CreatingSample//outputfile.txt', 'a+')
        #output.write(line +'\n')
        
        processedTweet = processTweet(line)
        featureVector = getFeatureVector(processedTweet)
        output.write(str(featureVector) + '\n')
        
        top_50 = getStopWordList('E:/(3) Semester 2 July 2018/COMP90019_Distributed Computing Project/From fahmin/CreatingSample/data/top_50.txt')
        t_string = ""
        flag = 0
        for w in featureVector:
            if(w in top_50):
                row_dict = {}
                flag = 1
                for i in top_50:
                    row_dict[i] = featureVector.count(i)
                    t_string = t_string + str(featureVector.count(i)) + ","
                t_string = t_string + '2' + ","
                output.write(str(t_string) + '\n')
                break;
        output.close()
        storm.emit([flag,t_string])

tweetRead().run()

