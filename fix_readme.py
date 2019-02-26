with open("readme.md", "r") as r:
    with open("README.md", "w") as w:
        for line in r:
            if "START" in line:
                break
        for line in r:
            if "<NN>\n" in line:
                line = line[:-5]
            w.write(line)

