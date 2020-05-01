import glob

read_files = glob.glob("remarks/*.txt")

with open("remarks_corpus.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
