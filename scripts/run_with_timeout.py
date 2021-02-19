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



if len(sys.argv) !=2:
    raise ValueError("provide directory to store results")
root = sys.argv
os.mkdir(root)

log = open(root + "/log.txt", "w")
stderr = open(root + "/err_log.txt", "w")
for file in glob("~/Treewidth-PACE-2017-instances/gr/heuristic/he0*.gr"):
    base = os.path.basename(file)
    instance = os.path.splitext(base)[0]

    stdin = open(file, "r")
    stdout = open(root + "/example.td", "w")
    result, time = run_cmd(["./target/release/arboretum-cli", "--mode", "exact"], stdin, stdout, stderr)
    log.write(str(instance) + " " + str(result) + " " + str(time))
