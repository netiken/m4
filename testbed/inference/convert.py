import numpy as np
import argparse

def convert(args):
    print("calling convert, opening " + args.input)
    f = open(args.input + "/path_1.txt", 'r')
    print("opened")
    print(f)
    text = ""

    flow_to_path = []

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

        flow_to_path.append(hops)

    f.close()


    print("writing to " + args.output)
    #print(text)
    f = open(args.output + "/flow_to_path.txt", 'w')
    f.write(text)
    f.close()

    flink = np.load(args.input + "/flink.npy")
    hop_to_id = dict()
    for i in range(len(flink)):
        hop_to_id[flink[i]] = i

    flow_to_links = []
    text = ""
    for i in range(len(flow_to_path)):
        path = flow_to_path[i]
        text += str(len(path)) + " "
        for hop in path:
            text += str(hop_to_id[hop]) + " "
        text += "\n"

    f = open(args.output + "/flow_to_links.txt", 'w')
    f.write(text)
    f.close()

    #text = ""
    #for i in range(len(flink) - 1):
    #    link = flink[i]
    #    text += link[:link.find("-")] + " " + link[link.find("-") + 1:] + "\n"

    #link = flink[len(flink) - 1]
    #text += link[:link.find("-")] + " " + link[link.find("-") + 1:]

    #f = open(args.output + "/flink.txt", 'w')
    #f.write(text)
    #f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    convert(args)
