#!/usr/bin/env python

import sys
import numpy as np

from sim import *
from scenario import *
from AStarAlgo import *

import copy

# use the nearest water source then the nearest building
# repeat
def shortest(env):
    fires = []
    lakes = []
    drone = None
    for b in env:
        x, y, z = b
        x = np.float(x)
        y = np.float(y)
        z = np.float(z)
        if env[b] == "orange":
            fires.append(np.array([x, y, z]))
        if env[b] == "blue":
            lakes.append(np.array([x, y, z]))
        if env[b] == "drone":
            drone = np.array([x, y, z])
    print(fires)
    print(lakes)
    print("Drone:", drone)
    path = []
    totpath = 0
    while len(fires) != 0:
        #print("fires:", fires)
        minlake = np.inf
        minlakepos = -1
        for i,l in enumerate(lakes):
            lpos = copy.deepcopy(l)
            lpos[1] +=1
            if np.linalg.norm(drone - lpos) < minlake:
                minlake = np.linalg.norm(drone-lpos)
                minlakepos = lpos
        totpath += minlake
        drone = minlakepos
        x,y,z = minlakepos
        path.append((x,y,z))
        #print totpath, minlake, minlakepos
        minfire = np.inf
        minfirepos = -1
        fireidx = -1
        for i,f in enumerate(fires):
            fpos = copy.deepcopy(f)
            fpos[1] +=2
            if np.linalg.norm(drone - fpos) < minfire:
                minfire = np.linalg.norm(drone - fpos)
                minfirepos = fpos
                fireidx = i
        totpath += minfire
        drone = minfirepos
        x,y,z = minfirepos
        path.append((x,y,z))
        #print totpath, minfire, minfirepos
        del fires[fireidx]
    return path

def shortest_new(env):
    fires = []
    lakes = []
    drone = None
    for b in env:
        x, y, z = b
        x = np.float(x)
        y = np.float(y)
        z = np.float(z)
        if env[b] == "orange":
            fires.append(np.array([x, y, z]))
        if env[b] == "blue":
            lakes.append(np.array([x, y, z]))
        if env[b] == "drone":
            drone = np.array([x, y, z])
    print(fires)
    print(lakes)
    print("Drone:", drone)
    path = []
    totpath = 0
    while len(fires) != 0:
        #print("fires:", fires)
        minlake = np.inf
        minlakepos = -1
        minfirepos = -1
        fireidx = -1
        minfire = np.inf
        for i,l in enumerate(lakes):
            lpos = copy.deepcopy(l)
            lpos[1] +=1

            # minfire = np.inf
            # minfirepos = -1
            # fireidx = -1
            for j,f in enumerate(fires):
                fpos = copy.deepcopy(f)
                fpos[1] +=2
                if 2* np.linalg.norm(lpos - fpos) + np.linalg.norm(drone - lpos) < minfire:
                    minfire = 2*np.linalg.norm(lpos - fpos) +  np.linalg.norm(drone - lpos)
                    # minlake = np.linalg.norm(drone - lpos)
                    minlakepos = lpos
                    minfirepos = fpos
                    fireidx = j
            
            # if np.linalg.norm(drone - lpos) < minlake:
            #     minlake = np.linalg.norm(drone-lpos)
            #     minlakepos = lpos
        totpath += minfire
        # drone = minlakepos
        x,y,z = minlakepos
        path.append((x,y,z))
        #print totpath, minlake, minlakepos
        # totpath += minfire
        drone = minfirepos
        x,y,z = minfirepos
        path.append((x,y,z))
        #print totpath, minfire, minfirepos
        del fires[fireidx]
    return path

def shortest_two(env):
    fires = []
    lakes = []
    drone = None
    for b in env:
        x, y, z = b
        x = np.float(x)
        y = np.float(y)
        z = np.float(z)
        if env[b] == "orange":
            fires.append(np.array([x, y, z]))
        if env[b] == "blue":
            lakes.append(np.array([x, y, z]))
        if env[b] == "drone":
            drone = np.array([x, y, z])
    print(fires)
    print(lakes)
    print("Drone:", drone)
    path = []
    totpath = 0
    while len(fires) != 0:
        #print("fires:", fires)

        #print totpath, minlake, minlakepos
        minfire = np.inf
        minfirepos = -1
        fireidx = -1
        for i,f in enumerate(fires):
            fpos = copy.deepcopy(f)
            fpos[1] +=2
            if np.linalg.norm(drone - fpos) < minfire:
                minfire = np.linalg.norm(drone - fpos)
                minfirepos = fpos
                fireidx = i
        #totpath += minfire
        #drone = minfirepos
        #x,y,z = minfirepos
        #path.append((x,y,z))
        minlake = np.inf
        minlakepos = -1
        for i,l in enumerate(lakes):
            lpos = copy.deepcopy(l)
            lpos[1] +=1
            if np.linalg.norm(drone - minfirepos) < minlake:
                minlake = np.linalg.norm(drone-lpos)
                minlakepos = lpos
        totpath += minlake
        drone = minlakepos
        x,y,z = minlakepos
        path.append((x,y,z))
        drone = minfirepos
        x,y,z = minfirepos
        path.append((x,y,z))
        #print totpath, minfire, minfirepos
        del fires[fireidx]
    return path

def check(x,y,z):
    c = True
    if x > 50 or x < -50:
        print("x bad", x)
        c = False
    if y > 50 or y < 0:
        print("y bad", y)
        c = False
    if z > 50 or z < -50:
        print("z bad", z)
        c = False
    return c

# pick the nearest building and use the nearest water source to that building
# repeat
def findpath(s, e):
    dx = e[0] - s[0]
    dy = e[1] - s[1]
    dz = e[2] - s[2]

    ax = np.abs(dx)*2
    ay = np.abs(dy)*2
    az = np.abs(dz)*2

    sx = np.sign(dx)
    sy = np.sign(dy)
    sz = np.sign(dz)

    x = s[0]
    y = s[1]
    z = s[2]

    path = []
    if ax >= np.max([ay, az]):
        yd = ay - ax/2
        zd = az - ax/2
        while True:
            if x == e[0]: break
            if (yd >= 0):
                y = y+sy
                yd = yd - ax
            if zd >= 0:
                z = z + sz
                zd = zd-ax
            x = x + sx
            yd = yd + ay
            zd = zd + az
            path.append([x, y, z])
            if not check(x, y, z):
                print("1:", x, y, z, sy)
    elif ay >= np.max([ax,az]):
        xd = ax - ay/2
        zd = az - ay/2
        while True:
            if y == e[1]: break
            if xd >= 0:
                x = x+sx
                xd = xd - ay
            if zd >= 0:
                z = z+sz
                zd = zd - ay
            y = y+sy
            xd = xd + ax
            zd = zd + az
            path.append([x, y, z])
            if not check(x, y, z):
                print("2:", x, y, z)
    elif az >= np.max([ax, ay]):
        xd = ax - az/2
        yd = ay - az/2
        while True:
            if z == e[2]: break
            if xd >= 0:
                x = x + sx
                xd = xd - az
            if yd >= 0:
                y = y + sy
                yd = yd - az
            z = z + sz
            xd = xd + ax
            yd = yd + ay
            path.append([x, y, z])
            if not check(x, y, z):
                print("3:", x, y, z)
    return path

if __name__ == "__main__":
    #sim = simulator(graphics=True)
    #sim.initialize("blankmap")
    #sim.generate_scenario()
    scene = scenario_cityfire()
    scene.generate_scenario()
    env = scene.simul.state()
    p = shortest(env)
    print(p)

    tot = 0
    dx, dy, dz = scene.simul.drone
    drone = np.array([dx, dy, dz])
    p2 = findpath(drone.astype(np.int), p[0].astype(np.int))
    scene.simul.trace(p2)
    tot+=len(p2)
    print(p2)
    for i in range(len(p)-1):
        p2 = findpath(p[i].astype(np.int), p[i+1].astype(np.int))
        print(p2, len(p2))
        tot += len(p2)
        scene.simul.trace(p2)
    print(tot)
    raw_input("hit enter")
    scene.simul.delete_trace()

    tot = 0
    e = p[0].astype(np.int)
    p2 = aStar((dx, dy, dz), (e[0], e[1], e[2]), scene.simul.state())
    newp=[]
    for x in p2:
        q = np.array([x[0], x[1], x[2]])
        newp.append(q)
    print(p2, len(p2))
    tot += len(p2)
    scene.simul.trace(newp)
    for i in range(len(p) - 1):
        s = p[i].astype(np.int)
        e = p[i+1].astype(np.int)
        print("start:", s, "end:", e)
        p2 = aStar((s[0], s[1], s[2]), (e[0], e[1], e[2]), scene.simul.state())
        newp = []
        for x in p2:
            q = np.array([x[0], x[1], x[2]])
            newp.append(q)
        print(p2, len(p2))
        tot += len(p2)
        scene.simul.trace(newp)
    print(tot)


