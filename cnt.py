f = open("./dataset/medical-corpus.txt")
content = f.read()
f.close()

cnt = 0
dct = {}
for line in content.split("\n"):
    lst = line.split(" ")
    if dct.get(len(lst)):
        val = dct.get(len(lst))
        dct[len(lst)] = val + 1
    else:
        dct[len(lst)] = 1

print(dct)
