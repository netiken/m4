import numpy as np
import argparse

def convert(args):
    print("calling convert, opening " + args.input)
    f = open(args.input, 'r')
    print("opened")
    print(f)
    text = ""

    for line in f.readlines()[1:]:
        colon = line.find(":")
        hops = line[colon + 1:]
        hops = hops.split(",")
        hops = hops[1: len(hops) - 1]
        count = len(hops) + 1
        text += str(count) + " "

        comma = hops[0].find("-")
        text += hops[0][:comma] + " "
        for path in hops[1: len(hops)]:
        #for i in range(len(hops[1: len(hops) -1])):
            comma = path.find("-")
            text += path[:comma] + " "
        comma = hops[len(hops) - 1].find("-")
        text += hops[len(hops) - 1][comma + 1:] + "\n"

    f.close()


    print("writing to " + args.output)
    print(text)
    f = open(args.output, 'w')
    f.write(text)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    convert(args)
