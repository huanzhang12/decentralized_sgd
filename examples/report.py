#!/usr/bin/env python2

import os
import sys
import random

nodes_file = sys.argv[1]
f = open(nodes_file)
hosts = []
for line in f:
    line = line.strip()
    if not line:
        continue
    t = line.split(":")
    hosts.append(t[0])
    s = int(t[1]) / 1000
f.close()

f = open(nodes_file, "w")
while True:
    r = random.randrange(2,15)
    if r != s:
        break
i = 0
for host in hosts:
    f.write("{}:{}\n".format(host, r * 1000 + i))
    i = i + 1
f.close()



