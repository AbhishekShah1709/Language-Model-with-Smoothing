import random
import numpy as np
import re
import math
import time

epsilon = 0.00000001

def read_data(filename):
    f = open(filename)
    content = f.read()
    f.close()

    return content

def tokenize(data, vocab_dct=None):

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
   
    if vocab_dct==None:
        return fin_data

    for word in words:
        if vocab_dct.get(word):
            value = vocab_dct[word]
            vocab_dct[word] = value+1
        else:
            vocab_dct[word] = 1

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

def preprocessing_data(data, vocab_dct):

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


def preprocessing_line(line):
    
    line = convert_to_lowercase(line)
    line = rem_urls(line)
    line = rem_hashtags(line)
    line = rem_mentions(line)
    line = rem_punct(line)
    line = tokenize(line)
    line = rep_tokens(line)
    
    return line

def train_test_split(data):

    data_lst = data.split("\n")
    tot_cnt = len(data_lst)

#    random.shuffle(data_lst)
    
    train_data_cnt = tot_cnt-1000
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
    
    for i in range(0,len(words)-n+1):
        tmp_tup = tuple(words[i:i+n])

        if dct.get(tmp_tup):
            val = dct[tmp_tup]
            dct[tmp_tup] = val+1
        else:
            dct[tmp_tup] = 1

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
            initial_tup = tuple(words[0:4])

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


def helper_fnc(gram, dct_lst):

    gram_len = len(gram)

    if gram_len == 0:
        tot_sum = sum(dct_lst[1].values())
        return tot_sum
    if dct_lst[gram_len].get(gram):
        return dct_lst[gram_len][gram] 
    else:
        return 0


def witten_helper(sent, dct_lst):
    
    u_cnt = sum([c[:-1]==sent for c in dct_lst[len(sent)+1].keys()])
    k = float(u_cnt + int(helper_fnc(sent, dct_lst)))
    
    if k == 0:
        return random.uniform(0.00001,0.0001)
    else:
        return (u_cnt/k)

def witten_smoothing(sent, dct_lst):
    
    x = len(sent)
    
    if x == 1:
        tot_sum = sum(dct_lst[1].values())
        return helper_fnc(sent, dct_lst)/tot_sum
    
    if(helper_fnc(sent[:-1], dct_lst) != 0):
        helper_val = witten_helper(sent[:-1], dct_lst)

        f_term = (1-helper_val)*(helper_fnc(sent, dct_lst)/helper_fnc(sent[:-1], dct_lst))
        s_term = helper_val*witten_smoothing(sent[:-1], dct_lst)
        
        return f_term + s_term 
    
    else:
        f_term = random.uniform(0.00001, 0.0001)
        s_term = witten_helper(sent[:-1], dct_lst)*witten_smoothing(sent[:-1], dct_lst)
        return f_term + s_term 


def kneyser_ney_smoothing(sent, dct_lst, max_len):

    d = 0.75
    x = len(sent)

    ret_val = helper_fnc(sent[:-1], dct_lst)

    if ret_val == 0:
        term = random.uniform(0.00001,0.0001)
        lamda = random.uniform(0,1)
    else:
        if x != max_len:
            term = max(0,(sum(token[1:] == sent for token in dct_lst[x+1].keys())-d))
            term = term/ret_val
        else:
            term = max(0,helper_fnc(sent, dct_lst)-d)
            term = term/ret_val
        lamda = d*sum(token[:-1] == sent[:-1] for token in dct_lst[x].keys())
        lamda = lamda/ret_val

    if x == 1:
       return term
    else:
        return term + lamda*kneyser_ney_smoothing(sent[:-1], dct_lst, max_len)

def perplexity(sent, dct_lst):

    sent_words = sent.split(" ")
    prob = 1
    tot_kneyser = 0
    tot_witten = 0
    t1 = time.time()
    for i in range(len(sent_words)-3):
    
        temp_sent = tuple(sent_words[i:i+4])
#        t1 = time.time()
        val_kneyser = kneyser_ney_smoothing(temp_sent, dct_lst, len(temp_sent))
#        t2 = time.time()
#        print(t2-t1)
        tot_kneyser += math.log(val_kneyser+epsilon)
       
#        print("-----")
#        t1 = time.time()
        val_witten = witten_smoothing(temp_sent, dct_lst)
#        t2 = time.time()
#        print(t2-t1)
        tot_witten += math.log(val_witten+epsilon)

    t2 = time.time()
#    print(t2-t1)
    word_count = len(sent_words)
    perplex_score_kneyser = -(tot_kneyser/word_count)
    perplex_score_witten = -(tot_witten/word_count)

    return perplex_score_kneyser, perplex_score_witten

##########################################################################################3

general_data = read_data('./dataset/general-tweets.txt')

general_vocab_dct = {}
preprocessed_general_data, general_vocab_dct = preprocessing_data(general_data, general_vocab_dct)

#for line in preprocessed_general_data:
#
#    print(line)

##########################################################################################3

europarl_data = read_data('./dataset/europarl-corpus.txt')
medical_data = read_data('./dataset/medical-corpus.txt')

europarl_train_data, europarl_test_data = train_test_split(europarl_data)
medical_train_data, medical_test_data = train_test_split(medical_data)

europarl_vocab_dct = {}
preprocessed_europarl_data, europarl_vocab_dct = preprocessing_data(europarl_train_data, europarl_vocab_dct)
medical_vocab_dct = {}
preprocessed_medical_data, medical_vocab_dct = preprocessing_data(medical_train_data, medical_vocab_dct)

europarl_1dct, europarl_2dct, europarl_3dct, europarl_4dct, ending_word_dct = make_ngrams(preprocessed_europarl_data)
medical_1dct, medical_2dct, medical_3dct, medical_4dct, ending_word_dct = make_ngrams(preprocessed_medical_data)


europarl_combined_lst = []
europarl_combined_lst.append([0])
europarl_combined_lst.append(europarl_1dct)
europarl_combined_lst.append(europarl_2dct)
europarl_combined_lst.append(europarl_3dct)
europarl_combined_lst.append(europarl_4dct)

fe = open("./dataset_split/euro_part_3", "r")
content_part_euro = fe.read()
fe.close()

f1 = open("./outputs/L1_3_train.txt", "w")
f2 = open("./outputs/L2_3_train.txt", "w")

for line in content_part_euro.split("\n"):

    print(line)
    pre_line = preprocessing_line(line)
    acc_kneyser, acc_witten = perplexity(pre_line, europarl_combined_lst)
   
    f1.write(line+"\t"+str(acc_kneyser)+"\n")
    f2.write(line+"\t"+str(acc_witten)+"\n")

f1.close()
f2.close()

medical_combined_lst = []
medical_combined_lst.append([0])
medical_combined_lst.append(medical_1dct)
medical_combined_lst.append(medical_2dct)
medical_combined_lst.append(medical_3dct)
medical_combined_lst.append(medical_4dct)

fm = open("./dataset_split/med_part_3", "r")
content_part_med = fm.read()
fm.close()


f3 = open("./outputs/L3_3_train.txt", "w")
f4 = open("./outputs/L4_3_train.txt", "w")

for line in content_part_med.split("\n"):

    pre_line = preprocessing_line(line)
    acc_kneyser, acc_witten = perplexity(pre_line, medical_combined_lst)
    
    f3.write(line+"\t"+str(acc_kneyser)+"\n")
    f4.write(line+"\t"+str(acc_witten)+"\n")

f3.close()
f4.close()
