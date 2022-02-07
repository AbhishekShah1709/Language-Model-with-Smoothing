f1 = open("L1_test_set_perplexity", "w")

for i in range(1):

    f = open("./L1_" + str(i+1) + "_test.txt", "r")
    content = f.read()
    f.close()

    corpus = content.split("\n")
    cnt=0
    for j in range(len(corpus)):
        f1.write(corpus[j]+"\n")
        cnt+=1
        if cnt==1000:
            break

f1.close()



