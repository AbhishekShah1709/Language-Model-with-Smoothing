import random
import numpy as np
import re

def read_data(filename):
    f = open(filename)
    content = f.read()
    f.close()

    return content

def tokenize(data, vocab_dct):

    punct = '''!()-[]{};:"\,<>./?@#$%^&*_~'''

    ## Add space before and after punctuations
    tmp_data = ""
    for char in data:
        if char in punct:
            tmp_data += " " + char + " "
        else:
            tmp_data += char

    ## Handling single quotes/apostrophe as per the requirements
    fin_data = ""
    for idx in range(len(tmp_data)):
        if idx==0:
            fin_data += ' ' + tmp_data[idx]
        elif idx==len(tmp_data)-1:
            fin_data += tmp_data[idx] + ' '
        else:
            if tmp_data[idx]=='\'' and tmp_data[idx-1]!=' ' and tmp_data[idx+1]!=' ':
                fin_data += ' ' + tmp_data[idx]
            elif tmp_data[idx]=='\'' and tmp_data[idx-1]==' ' and tmp_data[idx+1]!=' ': 
                fin_data += tmp_data[idx] + ' '
            elif tmp_data[idx]=='\'' and tmp_data[idx-1]!=' ' and tmp_data[idx+1]==' ': 
                fin_data += ' ' + tmp_data[idx]
            else:
                fin_data += tmp_data[idx]

    fin_data = fin_data.strip(" ")
    words = fin_data.split(' ')
    
    for word in words:
        if vocab_dct.get(word):
            value = vocab_dct[word]
            vocab_dct[word] = value+1
        else:
            vocab_dct[word] = 1

#    print(vocab_dct)
    return fin_data, vocab_dct

def convert_to_lowercase(data):
    data = data.lower()
    return data

def repl_str(obj):
    return obj.group(0)[0]

def rem_punct(data):
    data = re.sub(r'(([^\w\s])\2+)', repl_str, data)
    return data

def rem_urls(data):
    urls_exp = "((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"

    data = re.sub(urls_exp, '<URL>', data)
    return data

def rem_mentions(data):
    mentions_exp = "@(\w+)"

    data = re.sub(mentions_exp, '<MENTION>', data)
    return data

def rem_hashtags(data):
    hashtags_exp = "#(\w+)"

    data = re.sub(hashtags_exp, '<HASHTAG>', data)
    return data

def rep_tokens(data):

    data = re.sub('< HASHTAG >', '<HASHTAG>', data)
    data = re.sub('< URL >', '<URL>', data)
    data = re.sub('< MENTION >', '<MENTION>', data)

    return data

def preprocessing(data, vocab_dct):

    lines = []
    for line in data.split("\n"):
    
        line = convert_to_lowercase(line)
        line = rem_urls(line)
        line = rem_hashtags(line)
        line = rem_mentions(line)
        line = rem_punct(line)
        line, vocab_dct = tokenize(line, vocab_dct)
        line = rep_tokens(line)
    
        lines.append(line)
    
    return lines, vocab_dct

def train_test_split(data, cutoff):

    data_lst = data.split("\n")
    tot_cnt = len(data_lst)

    random.shuffle(data_lst)
    
    train_data_cnt = int(tot_cnt*cutoff)
    train_data = data_lst[:train_data_cnt]
    test_data = data_lst[train_data_cnt:]

    fin_train_data = ""
    fin_test_data = ""

    for i in range(len(train_data)):
        fin_train_data += train_data[i] + "\n"

    for i in range(len(test_data)):
        fin_test_data += test_data[i] + "\n"
    
    return fin_train_data, fin_test_data


def make_ngrams_dct(initial_str, dct, words, n, ending_word_dct=None):
    
    if dct.get(initial_str):
        val = dct[initial_str]
        dct[initial_str] = val+1
    else:
        dct[initial_str] = 1

    rem_str = initial_str

    for i in range(n,len(words)):
        rem_str = rem_str.split("_",1)
        
        if len(rem_str)>=2:
            print(rem_str)
            rem_str = rem_str[1]
            rem_str = rem_str + "_" + words[i]

            if n==4:
                if ending_word_dct.get(words[i]):
                    value = ending_word_dct[words[i]]
                    ending_word_dct[words[i]] = value+1
                else:
                    ending_word_dct[words[i]] = 1
        else:
            rem_str = words[i]
        
        if dct.get(rem_str):
            val = dct[rem_str]
            dct[rem_str] = val+1
        else:
            dct[rem_str] = 1
    
    return dct

def make_ngrams(data_lines):
    
    _4gram_dct = {}
    _3gram_dct = {}
    _2gram_dct = {}
    _1gram_dct = {}
    ending_word_dct = {}
    
    for line in data_lines:
        line = line.strip()
        words = line.split(" ")
    
        if len(words)>=4:
            initial_str = words[0] + "_" + words[1] + "_" + words[2] + "_" + words[3]
            if ending_word_dct.get(word[3]):
                value = ending_word_dct[word[3]]
                ending_word_dct[word[3]] = value+1
            else:
                ending_word_dct[word[3]] = 1           
            
            _4gram_dct = make_ngrams_dct(initial_str, _4gram_dct, words, 4, ending_word_dct)
        
        if len(words)>=3:
            initial_str = words[0] + "_" + words[1] + "_" + words[2]
            _3gram_dct = make_ngrams_dct(initial_str, _3gram_dct, words, 3)
        
        if len(words)>=2:
            initial_str = words[0] + "_" + words[1]
            _2gram_dct = make_ngrams_dct(initial_str, _2gram_dct, words, 2)
        
        if len(words)>=1:
            initial_str = words[0]
            _1gram_dct = make_ngrams_dct(initial_str, _1gram_dct, words, 1)

    return _1gram_dct, _2gram_dct, _3gram_dct, _4gram_dct, ending_word_dct 


def kneyser_ney_smoothing(sent, _1dct, _2dct, _3dct, _4dct, ending_word_dct, maxi):

    ###### gram = sent

#    last_word = sent[-1]
#    p_cont = ending_word_dct[last_word]/len(_4dct.keys())
#
#    def kneser(gram, maxi):
    x = len(gram)
    d = 0.75
    if find_val(gram[:-1]) == 0:
        lamda = random.uniform(0,1)
        term = random.uniform(0.00001,0.0001)
    else:
        if x == maxi:
            term = max(0,find_val(gram)-d)/find_val(gram[:-1])
        else:
            term = max(0,(sum(token[1:] == gram for token in ngrams_c[len(gram) + 1].keys()) - d)) / find_val(gram[:-1])
        lamda = d * sum(token[:-1] == gram[:-1] for token in ngrams_c[len(gram)].keys()) / find_val(gram[:-1])
    if x == 1:
       return term
    else:
        return term + lamda * kneser(gram[:-1], maxi)


def perplexity(sent):

    prob = kneyser_ney_smoothing(medical_1dct, medical_2dct, medical_3dct, medical_4dct, ending_word_dct)
    word_count = len(sent.split(" "))
    perplex_score = np.power((1/prob), word_count)

    return perplex_score

##########################################################################################3

#general_data = "This has to be 'what are you saying????' said isn''t it!!!! @wow #wowing"
general_data = read_data('./dataset/general-tweets.txt')

general_vocab_dct = {}
preprocessed_general_data, general_vocab_dct = preprocessing(general_data, general_vocab_dct)
#print(preprocessed_general_data)
#exit(0)

##########################################################################################3

#europarl_data = read_data('./dataset/europarl-corpus.txt')
#medical_data = read_data('./dataset/medical-corpus.txt')
europarl_data = "This has to be 'what are you saying????' said isn''t it!!!! @wow #wowing"
medical_data = "This has to be 'what are you saying????' said isn''t it!!!! @wow #wowing this has to be"

train_europarl_data, test_europarl_data = train_test_split(europarl_data, 0.8)
train_medical_data, test_medical_data = train_test_split(medical_data, 0.8)

europarl_vocab_dct = {}
preprocessed_europarl_data, europarl_vocab_dct = preprocessing(europarl_train_data, europarl_vocab_dct)
medical_vocab_dct = {}
preprocessed_medical_data, medical_vocab_dct = preprocessing(medical_train_data, medical_vocab_dct)

europarl_1dct, europarl_2dct, europarl_3dct, europarl_4dct, ending_word_dct = make_ngrams(preprocessed_europarl_data)
medical_1dct, medical_2dct, medical_3dct, medical_4dct, ending_word_dct = make_ngrams(preprocessed_medical_data)


for line in test_europarl_data:

    acc = perplexity(line)
    print(line, end=": ")
    print(acc)

for line in test_medical_data:

    acc = perplexity(line)
    print(line, end=": ")
    print(acc)

#for k,v in europarl_dct.items():
#    print(k, end=": ")
#    print(v)
