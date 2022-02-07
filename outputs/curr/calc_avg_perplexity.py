f = open("./test_set/L4_test_set_perplexity")
content = f.read()
f.close()

tot=0
cnt=0

for line in content.split("\n"):

    if line!="":
        val = line.split("\t")[1]
        tot+=float(val)
        cnt+=1

print(tot)
print(tot/cnt)
