f = open(".gitignore", "r")
cleaned_lst = []
while True:
    line = f.readline()
    if not line: break
    if line[0:2] in ["./", ".\\"]:
    	cleaned_lst.append(line[2:])
    print ("\"{0}\" to \"{1}\"".format(line, line[2:]))

f.close()

f = open(".gitignore", "a")
for line in cleaned_lst:
	f.write(line)
f.close()