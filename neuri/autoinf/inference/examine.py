s = set()
with open("log", "r") as f:
    for line in f:
        name, t = line.strip().split(" ")
        if t == "start":
            s.add(name)
        else:
            s.remove(name)
print(s)
