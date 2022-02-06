lst = ['ab', 'ac', 'bg', 'fe']

#val = [sum(token[:-1] == 'a') for token in lst]
val = sum([(token[:-1]=='a') for token in lst])
print(val)
