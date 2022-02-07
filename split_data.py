f = open("./dataset/medical-corpus.txt")
content = f.read()
f.close()

cnt=0
file_cnt=1

f = open("./dataset_split/med_part_" + str(file_cnt), "w")

for line in content.split("\n"):

    f.write(line+"\n")

    cnt+=1

    if cnt==1000:
        cnt=0
        file_cnt+=1
        f.close()
        f = open("./dataset_split/med_part_" + str(file_cnt), "w")

f.close()
