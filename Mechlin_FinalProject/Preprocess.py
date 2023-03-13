import numpy as np
import nltk
import csv
import re


class vectorLoader:
    def __init__(self,file):
        self.vectorSet = file

    def loadVocab(self,write,corpus):
        c = open(corpus,'r',encoding='utf8')
        text = c.read()
        c.close()
        tokens = nltk.word_tokenize(text)
        w = open(write,'w',encoding='utf8')
        with open(self.vectorSet, 'r',encoding='utf8') as f:
            for line in f:
                word = line[0:line.index(' ')]
                for token in tokens:
                    if word == token.lower():
                        w.write(line)
                        break
            w.close()
            f.close()

def embed(file):
    embeddings_dict = {}
    with open(file, 'r',encoding= 'utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def isPunct(token):
    punctList = ['.', '!', '?']
    result = False
    for p in punctList:
        result = result or token == p
    return result

def clean_text(text):
    text = text.lower()
    #Delete all links in text
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    #Delete all emojis
    emoji = re.compile("["
                        u"\U0001F600-\U0001FFFF"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    text = emoji.sub(r'', text)
    #Delete the rest of the tokens
    text = re.sub(r"[,\"@#$%^&*(){}/;`~:<>+=-]", "", text)
    tokens = nltk.word_tokenize(text)
    return tokens

def vectorizeSentences(sentences,vector_dict):
    sentenceCnt = len(sentences)
    sentenceVectors = np.array([[0]*50]*sentenceCnt)
    j = 0
    for i in range(sentenceCnt):
        sum = np.array([0]*50)
        j = 0
        tokens = clean_text(sentences[i])
        print(tokens)
        while True:
            if isPunct(tokens[j]):
                break
            try:
                sum = sum + vector_dict[tokens[j]]
            except:
                pass 
            j = j + 1
        sentenceVectors[i] = sum
        print(sentenceVectors[i][0])
        i = i + 1
    return sentenceVectors

def tabulate(vectors,filename):
    f = open(filename,'w', encoding = 'utf8')
    c = csv.writer(f)
    columnNum = len(vectors)
    row = ["ID","relations"]
    c.writerow(row)
    for i in range(1,columnNum) :
        row = [str(i),'']
        for j in range(i):
            print("j: " + str(j))
            mag1 = (vectors[i]**2).sum()**0.5
            mag2 = (vectors[j]**2).sum()**0.5
            dot = np.dot(vectors[i],vectors[j])
            similarity = dot/(mag1*mag2)
            row[1] = row[1] + f',{str(j)}:{similarity}'
        row[1] = row[1][1:]
        c.writerow(row)

def parseSentences(read,write):
    r = open(read,'r',encoding='utf8')
    w = open(write,'w', encoding='utf8')
    c = csv.writer(w)
    sentenceList = []
    c.writerow(["content"])
    start = r.tell()
    end = 0
    cur = '#'
    i = 0
    while not cur == '':
        cur = r.read(1)
        if isPunct(cur):
            end = r.tell()
            r.seek(start)
            print(end-start)
            sentenceList.append(r.read(end-start).strip())
            c.writerow([sentenceList[i]])
            start = end
            i = i+1
    return sentenceList
def results(displayNum):
    r = open('export.csv','r',encoding='utf8')
    c= csv.DictReader(r)
    lines = []
    on = []
    max = 0
    i = -1
    for line in c:
        if i == 0:
            i += 1
        else:
            lines.append(line)
            on.append(False)
            i += 1
    for i in range(displayNum):
        max = 0
        maxIndex = 0
        for j in range(len(lines)):
            if float(lines[j]['score']) > max and on[j] == False:
                max = float(lines[j]['score'])
                maxIndex = j
        on[maxIndex] = True

    for i in range(len(lines)):
        if on[i] == True:
            print(lines[i]['\ufeffname'])

        
            


#loader = vectorLoader("glove.6B.50d.txt")
#loader.loadVocab("volcano.6B.50d.txt","volcano.txt")
#sentences = parseSentences('volcano.txt','sentences.csv')
#vector_dict = embed("volcano.6B.50d.txt")
#vectors = vectorizeSentences(sentences,vector_dict)
#tabulate(vectors,"relations.csv")
results(5)