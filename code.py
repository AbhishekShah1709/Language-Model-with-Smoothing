import random
import numpy as np
import re
import math

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


def make_ngrams_dct(initial_tup, dct, words, n, ending_word_dct=None):
    
#    if dct.get(initial_tup):
#        val = dct[initial_tup]
#        dct[initial_tup] = val+1
#    else:
#        dct[initial_tup] = 1
#
#    rem_tup = initial_tup

    for i in range(0,len(words)-n+1):
        tmp_tup = tuple(words[i:i+n])

        if dct.get(tmp_tup):
            val = dct[tmp_tup]
            dct[tmp_tup] = val+1
        else:
            dct[tmp_tup] = 1

#        rem_str = rem_str.split("_",1)
#        
#        if len(rem_str)>=2:
#            print(rem_str)
#            rem_str = rem_str[1]
#            rem_str = rem_str + "_" + words[i]
#
##            if n==4:
##                if ending_word_dct.get(words[i]):
##                    value = ending_word_dct[words[i]]
##                    ending_word_dct[words[i]] = value+1
##                else:
##                    ending_word_dct[words[i]] = 1
#        else:
#            rem_str = words[i]
#        
#        if dct.get(rem_str):
#            val = dct[rem_str]
#            dct[rem_str] = val+1
#        else:
#            dct[rem_str] = 1
#    
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
            #initial_str = words[0] + "_" + words[1] + "_" + words[2] + "_" + words[3]
            initial_tup = tuple(words[0:4])

#            if ending_word_dct.get(word[3]):
#                value = ending_word_dct[word[3]]
#                ending_word_dct[word[3]] = value+1
#            else:
#                ending_word_dct[word[3]] = 1           
#            
            _4gram_dct = make_ngrams_dct(initial_tup, _4gram_dct, words, 4, ending_word_dct)
        
        if len(words)>=3:
            initial_tup = tuple(words[0:3])
            _3gram_dct = make_ngrams_dct(initial_tup, _3gram_dct, words, 3)
        
        if len(words)>=2:
            initial_tup = tuple(words[0:2])
            _2gram_dct = make_ngrams_dct(initial_tup, _2gram_dct, words, 2)
        
        if len(words)>=1:
            initial_tup = tuple(words[0:1])
            _1gram_dct = make_ngrams_dct(initial_tup, _1gram_dct, words, 1)

    return _1gram_dct, _2gram_dct, _3gram_dct, _4gram_dct, ending_word_dct 


def find_val(gram, dct_lst):
    if len(gram) == 0:
        return sum(dct_lst[1].values())
    if gram in dct_lst[len(gram)].keys():
        return dct_lst[len(gram)][gram] 
    else:
        return 0


def witten2(sent, dct_lst):
    unique = 0
    for c in dct_lst[len(sent)+1].keys():
        if(c[:-1] == sent):
            
            unique += 1
    k = float(unique + int(find_val(sent, dct_lst)))
    if(k != 0):
        return (unique / k)
    else:
        return random.uniform(0.00001,0.0001)

def witten_smoothing(sent, dct_lst):
    x = len(sent)
    if x == 1:
        return find_val(sent, dct_lst) / sum(dct_lst[1].values())
    if(find_val(sent[:-1], dct_lst) != 0):
        return ((1 - witten2(sent[:-1], dct_lst)) * (find_val(sent, dct_lst) / find_val(sent[:-1], dct_lst))) + (witten2(sent[:-1], dct_lst) * witten_smoothing(sent[:-1], dct_lst))
    else:
        return random.uniform(0.00001, 0.0001) +  (witten2(sent[:-1], dct_lst) * witten_smoothing(sent[:-1], dct_lst))


def kneyser_ney_smoothing(sent, dct_lst, ending_word_dct, maxi):

    ###### gram = sent

#    last_word = sent[-1]
#    p_cont = ending_word_dct[last_word]/len(_4dct.keys())
#
#    def kneser(gram, maxi):
    x = len(sent)
    d = 0.75
    if find_val(sent[:-1], dct_lst) == 0:
        lamda = random.uniform(0,1)
        term = random.uniform(0.00001,0.0001)
    else:
        if x == maxi:
            term = max(0,find_val(sent, dct_lst)-d)/find_val(sent[:-1], dct_lst)
        else:
            term = max(0,(sum(token[1:] == sent for token in dct_lst[len(sent) + 1].keys()) - d)) / find_val(sent[:-1], dct_lst)
        lamda = d * sum(token[:-1] == sent[:-1] for token in dct_lst[len(sent)].keys()) / find_val(sent[:-1], dct_lst)
    if x == 1:
       return term
    else:
        return term + lamda * kneyser_ney_smoothing(sent[:-1], dct_lst, ending_word_dct, maxi)

def perplexity(sent, dct_lst):

    sent_words = sent.split(" ")
    prob = 1
    tot_kneyser = 0
    tot_witten = 0

    for i in range(len(sent_words)-3):
    
        temp_sent = tuple(sent_words[i:i+4])
        val_kneyser = kneyser_ney_smoothing(temp_sent, dct_lst,  ending_word_dct, len(temp_sent))
        tot_kneyser += math.log(val_kneyser)
        
        val_witten = witten_smoothing(temp_sent, dct_lst)
        tot_witten += math.log(val_witten)
#        prob *= kneyser_ney_smoothing(temp_sent, medical_1dct, medical_2dct, medical_3dct, medical_4dct, ending_word_dct, len(temp_sent))

    word_count = len(sent_words)
    perplex_score_kneyser = -(tot_kneyser/word_count)
    perplex_score_witten = -(tot_witten/word_count)
#    perplex_score = np.power((1/final_prob), word_count)

    return perplex_score_kneyser, perplex_score_witten

##########################################################################################3

##general_data = "This has to be 'what are you saying????' said isn''t it!!!! @wow #wowing"
#general_data = read_data('./dataset/general-tweets.txt')
#
#general_vocab_dct = {}
#preprocessed_general_data, general_vocab_dct = preprocessing(general_data, general_vocab_dct)
##print(preprocessed_general_data)
##exit(0)

##########################################################################################3

europarl_data = read_data('./dataset/europarl-corpus.txt')
medical_data = read_data('./dataset/medical-corpus.txt')
#europarl_data = "This has to be 'what are you saying????' said isn''t it!!!! @wow #wowing"
#medical_data = "This has to be 'what are you saying????' said isn''t it!!!! @wow #wowing this has to be"

europarl_train_data, europarl_test_data = train_test_split(europarl_data, 0.8)
medical_train_data, medical_test_data = train_test_split(medical_data, 0.8)

europarl_vocab_dct = {}
preprocessed_europarl_data, europarl_vocab_dct = preprocessing(europarl_train_data, europarl_vocab_dct)
medical_vocab_dct = {}
preprocessed_medical_data, medical_vocab_dct = preprocessing(medical_train_data, medical_vocab_dct)

europarl_1dct, europarl_2dct, europarl_3dct, europarl_4dct, ending_word_dct = make_ngrams(preprocessed_europarl_data)
medical_1dct, medical_2dct, medical_3dct, medical_4dct, ending_word_dct = make_ngrams(preprocessed_medical_data)


europarl_combined_lst = []
europarl_combined_lst.append([0])
europarl_combined_lst.append(europarl_1dct)
europarl_combined_lst.append(europarl_2dct)
europarl_combined_lst.append(europarl_3dct)
europarl_combined_lst.append(europarl_4dct)

for line in europarl_test_data.split("\n"):
    
    acc_kneyser, acc_witten = perplexity(line, europarl_combined_lst)
    print("***************************************")
    print("KNEYSER-NEY SMOOTHING")
    print(line, end=": ")
    print(acc_kneyser)
    print()
    print("WITTEN SMOOTHING")
    print(line, end=": ")
    print(acc_witten)
    print("***************************************")
    print()
    print()

for line in test_medical_data:

    acc_kneyser, acc_witten = perplexity(line, medical_combined_lst)
    print("***************************************")
    print("KNEYSER-NEY SMOOTHING")
    print(line, end=": ")
    print(acc_kneyser)
    print()
    print("WITTEN SMOOTHING")
    print(line, end=": ")
    print(acc_witten)
    print("***************************************")
    print()
    print()
