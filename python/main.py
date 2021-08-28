from __future__ import absolute_import
import sys
import configparser
import subprocess as sp
import os

sys.path.append(os.path.realpath("./utils/"))
sys.path.append(os.path.realpath("./topic_analysis/"))

def main(argv):
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    task = config["Analysis"]["task"]
    if task in ["topic_analysis", "multi_class"]:
        sp.check_call(["python", task + "/train.py", sys.argv[1]])
    else:
        raise ValueError("Invalid task defined.")

if __name__ == "__main__":
    main(sys.argv[1:])
