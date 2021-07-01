import sys
import getopt
import configparser
import subprocess as sp


def main(argv):
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    task = config["Analysis"]["task"]

    if task in ["topic-analysis", "multi-class"]:
        sp.check_call(["python", task + "/train.py", sys.argv[1]])
    elif task == "bayes-topic":
        sp.check_call(["python", "topic-analysis/train_bayes.py", sys.argv[1]])
    else:
        print("Invalid task defined.")
        exit()

if __name__ == "__main__":
    main(sys.argv[1:])
    
