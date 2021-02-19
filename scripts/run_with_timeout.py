import signal
import subprocess as sub
import datetime
from time import sleep
from signal import SIGINT
from glob import glob
import os
import sys

def get_elapsed(start):
    return (datetime.datetime.now() - start).total_seconds()


def run_cmd(cmd, stdin, stdout, stderr, timeout=1800, poll_interval=1, sigint_grace_period=60):
    p = sub.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr)
    start_time = datetime.datetime.now()
    result = "exact"
    while p.poll() is None and get_elapsed(start_time) < timeout:
        sleep(poll_interval)
    if p.poll() is None:
        p.send_signal(sig=SIGINT)
        sleep(sigint_grace_period)
    if p.poll() is None:
        p.kill()
        result = "none"
    else:
        result = "some"
    return result, get_elapsed(start_time)



if len(sys.argv) !=3:
    raise ValueError("provide directory to store results and directory to glob for files")
root = sys.argv[1]
grfiles = sys.argv[2]

if not os.path.exists(root):
    os.mkdir(root)
for file in glob(grfiles + "/he0*.gr"):
    base = os.path.basename(file)
    instance = os.path.splitext(base)[0]

    stdin = open(file, "r")
    stdout = open(root + "/" + instance + ".td", "w")
    log = open(root + "/" + instance + "_log.txt", "w")
    stderr = open(root + "/" + instance + "_err_log.txt", "w")

    result, time = run_cmd(["./target/release/arboretum-cli", "--mode", "exact"], stdin, stdout, stderr)
    log.write(str(instance) + " " + str(result) + " " + str(time))
    stdin.close()
    stdout.close()
    log.close()
    stderr.close()
