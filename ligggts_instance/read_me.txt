Installation:

open terminal

import liggghts
print(liggghts.__file__)

change liggghts.py in this folder with the one in this folder


How to run silent:

import liggghts


cmd =["-screen","/dev/null"]
sim = liggghts.liggghts(cmdargs=cmd)
sim.file("init.sim")
