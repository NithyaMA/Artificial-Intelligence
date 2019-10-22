#!/usr/bin/env python
#
# docs for vpython for python 3
# http://www.glowscript.org/docs/VPythonDocs/VisualIntro.html
# docs for visual (vpython for python 2)
# http://vpython.org/contents/docs_vp5/visual/VisualIntro.html
# 

import sys, os, re, time
import numpy as np

if os.environ.get('SIM_NOBROWSER') == None:
    ver = sys.version_info.major
    if ver == 2: 
        from visual import *
    else:
        from vpython import *
    colormap = \
    {
    "red":    vector(1,   0,   0),
    "orange": vector(1,   0.5, 0),
    "yellow": vector(1,   1,   0),
    "green":  vector(0,   1,   0),
    "teal":   vector(0,   1,   0.5),
    "blue":   vector(0,   0,   1),
    "indigo": vector(0.5, 0,   1),
    "violet": vector(1,   0,   1),
    "black":  vector(0,   0,   0),
    "gray":   vector(0.5, 0.5, 0.5),
    "drone":  vector(1,   1,   1),
    "blue":   color.blue,
    "grid":   vector(0.3, 0.8, 0.3)
    }

class block:
    nextId = 0
    def __init__(self, in_color, in_vpBox):
        self.color = in_color
        self.vpBox = in_vpBox
        self.id = block.nextId
        block.nextId += 1

class simulator:
    blocks = {}
    inv_blocks = {}
    legal = [-1, 0, 1]
    history = []
    lakes = []

    def __init__(self, graphics=False, debug=False):
        if os.environ.get('SIM_NOBROWSER') == None:
            self.graphics = graphics
        else:
            self.graphics = False
        self.scenario = np.zeros(shape=(101, 50, 101))
        self.env = {}
        self.drone = None
        self.droneId = None
        self.droneattached = None
        self.path = []
        self.debug = debug
        if self.graphics: self.initializegraphics()

    def initializegraphics(self):
        if ver == 2:
            scene = display(title="Firefighter", forward=(0,-1,0))          
        # the following is broken, using canvas() disables rotate
        #else:
        #    scene = canvas(title="Firefighter", forward=vector(0, -1, 0))
        #    scene.select()
        fx,fz,ex,ez = -51,-51,50,50
        spacing = 1
        offset = 0.5
        ypos = -0.5
        size = 0.05
        curve(pos=[vector(fx+offset,ypos,fz+offset), vector(ex+offset,ypos,fz+offset)], color=colormap["grid"], radius=size)
        curve(pos=[vector(ex+offset,ypos,fz+offset), vector(ex+offset,ypos,ez+offset)], color=colormap["grid"], radius=size)
        curve(pos=[vector(ex+offset,ypos,ez+offset), vector(fx+offset,ypos,ez+offset)], color=colormap["grid"], radius=size)
        curve(pos=[vector(fx+offset,ypos,ez+offset), vector(fx+offset,ypos,fz+offset)], color=colormap["grid"], radius=size)
        """
        for x in range(fx+spacing, ex, spacing):
            curve(pos=[vector(x+offset, ypos, fz+offset), vector(x+offset, ypos, ez+offset)], color=colormap["grid"], radius=size)
        for z in range(fz+spacing, ez, spacing):
            curve(pos=[vector(fx+offset, ypos, z+offset), vector(ex+offset, ypos, z+offset)], color=colormap["grid"], radius=size)
        """

    def initialize(self, filename):
        if self.debug: print("DEBUG: initialize", filename)
        for line in open(filename):
            m = re.search("(-?\d+),(-?\d+),(-?\d+) (\w+)", line.strip())
            if m == None:
                if self.debug: print("Failed to parse line:", line.strip())
                continue
            self.add(m.group(1), m.group(2), m.group(3), m.group(4))
        self.inv_blocks = {v: k for k, v in self.blocks.items()}
        if self.graphics: self.updateenv()

    def state(self):
        return {k: v.color for k, v in self.env.items()}

    def getBlockIds(self):
        return {v.id: {"color": v.color, "pos": k} for k, v in self.env.items()}

    def togrid(self, x,y,z): return x+50, y, z+50

    def tosim(self, x,y,z): return x-50, y, z-50

    def log(self, x, y, z, item):
        self.history.append([x, y, z, item])

    def add(self, x, y, z, item):
        # Convert the position strings to integers
        (ix, iy, iz) = (np.int(x), np.int(y), np.int(z))

        # add items to the scenario
        mapx,mapy,mapz = self.togrid(ix, iy, iz)

        # make sure this is legal
        if (mapx > 100) or (mapx < 0) or (mapy > 49) or (mapy < 0) or (mapz > 100) or (mapz < 0):
            print("Failed to add", item, "to", x, y, z, "position out-of-bounds")
        elif self.scenario[mapx,mapy,mapz] != 0:
            print("Failed to add", item, "to", x, y, z, "position occupied")
        else:
            # Check if this is a color/item we've seen before
            if item not in self.blocks.keys():
                self.blocks[item] = len(self.blocks) + 1
            self.scenario[mapx, mapy, mapz] = self.blocks[item]
            # Update the environment
            vpBox = None
            if self.graphics:
                boxPos = vector(ix, iy, iz)
                boxScale = vector(1, 1, 1)
                boxColor = colormap[item]
                vpBox = box(pos=boxPos, size=boxScale, color=boxColor)

            self.env[(ix, iy, iz)] = block(item, vpBox)

            if item == 'drone':
                self.drone = (ix, iy, iz)
                self.droneId = self.env[(ix,iy,iz)].id
            self.log(x,y,z, item)
            if self.debug: print("DEBUG: Added", item, "at", x,y,z)

    def attach(self):
        if self.drone == None:
            print("Drone is not on map")
            return False
        if self.droneattached != None:
            print("Drone is already attached to a block")
            return False

        # convert to internal representation
        x, y, z = self.drone
        mapx, mapy, mapz = self.togrid(x, y, z)

        # make sure the attach is legal
        if mapy - 1 < 0 or mapy -1 > 49:
            print("Drone trying to attach outside simulator", x, y-1, z)
            return False

        # make sure a block is available
        if self.scenario[mapx, mapy-1, mapz] < 1 or self.scenario[mapx, mapy-1, mapz] > len(self.blocks):
            print("Drone trying to attach when no block is available", x, y-1, z)
            return False

        # do the attach
        self.droneattached = self.scenario[mapx, mapy-1, mapz]
        return True

    def move(self, dx, dy, dz):
        if self.drone == None:
            print("Drone is not on map")
            return False

        if dx not in self.legal or dy not in self.legal or dz not in self.legal:
            print("Drone", dx, dy, dz, "is not a legal move")
            return False

        x,y,z = self.drone
        # make sure this is a legal move
        if (x+dx < -50) or (x+dx > 50) or (y+dy < 0) or (y+dy > 50) or (z+dz < -50) or (z+dz > 50):
            print("Drone trying to move outside map to", x+dx, y+dy, z+dz)
            return False

        # convert to internal representation
        mapx, mapy, mapz = self.togrid(x, y, z)
        destx, desty, destz = self.togrid(x+dx, y+dy, z+dz)

        # check if positions occupied
        if self.scenario[destx, desty, destz] != 0:
            # Skip this check if we're carrying a block and trying to move straight down
            if not (self.droneattached != None and dx == 0 and dy == -1 and dz == 0):
                print("Destination",destx, desty, destz,"already occupied")
                return False
        if self.droneattached != None and desty -1 < 0:
            print("Destination too low as drone has a block attached")
            return False
        if self.droneattached != None and self.scenario[destx, desty-1, destz] != 0 \
        and self.scenario[destx, desty-1, destz] != self.blocks["drone"]:
            print("Drone is attached and position below destination is occupied")
            return False

        # move the drone
        self.scenario[mapx, mapy, mapz] = 0
        self.scenario[destx,desty,destz] = self.blocks["drone"]
        self.drone = (x+dx, y+dy, z+dz)

        # if a block is attached move it as well
        if self.droneattached:
            if (mapx, mapy-1, mapz) not in self.lakes:
                self.scenario[mapx, mapy-1, mapz] = 0
            self.scenario[destx, desty-1, destz] = self.droneattached

        # Update the state info
        if self.droneattached:
            tempDrone = self.env[(x,y,z)]
            tempBlock = self.env[(x,y-1,z)]
            del self.env[(x,y,z)]
            del self.env[(x,y-1,z)]
            self.env[(x+dx,y+dy,z+dz)] = tempDrone
            self.env[(x+dx,y-1+dy,z+dz)] = tempBlock
        else:
            self.env[(x+dx,y+dy,z+dz)] = self.env[(x,y,z)]
            del self.env[(x,y,z)]

        self.log(x+dx, y+dy, z+dz, "drone")
        if self.graphics: self.updateenv()
        return True

    def release(self):
        if self.drone == None:
            print("Drone is not on map")
            return False
        if self.droneattached == None:
            print("Drone is not attached")
            return False

        x,y,z = self.drone
        mapx, mapy, mapz = self.togrid(x, y, z)

        # get the position where the block will land
        pos = None
        open_spaces = np.where(self.scenario[mapx, :mapy, mapz] == 0)
        if len(open_spaces[0]) == 0:
            # the block is sitting on the ground or on a block stack
            pos = mapy-1
        else:
            # the block is in the air, drop it to the lowest available position
            pos = open_spaces[0][0]
        # release from the drone
        self.scenario[mapx, mapy-1, mapz] = 0
        # put the block on the map
        # first check if its a fire, if so clear it
        if self.scenario[mapx, pos, mapz] == self.blocks["orange"]:
            self.scenario[mapx, pos, mapz] = 0
        # otherwise just drop the block
            self.scenario[mapx, pos, mapz] = self.droneattached
        newx, newy, newz = self.tosim(mapx, pos, mapz)
        
        # Update the state info
        if y-1 != newy:
            self.env[(newx,newy,newz)] = self.env[(x, y-1, z)]
            del self.env[(x, y-1, z)]
        
        self.log(newx, newy, newz, self.inv_blocks[self.droneattached])
        # turn attached off
        self.droneattached = None
        if self.graphics: self.updateenv()
        return True

    def speak(self, string):
        print("speak() not implemented")
        return False

    def printmap(self):
        print(self.scenario)

    def updateenv(self):
        if len(self.env) == 0:
            print("Environment has not been initialized.")
            return False
        for pos in self.env:
            item = self.env[pos].vpBox
            item.pos = vector(pos[0], pos[1], pos[2])
        return True

    # to show a flight path
    def trace(self, route):
        for i in route:
            x, y, z = i
            pathbox = box(pos=vector(x, y, z), size=vector(1,1,1), color=color.cyan)
            self.path.append(pathbox)

    def delete_trace(self):
        while len(self.path) != 0:
            self.path[0].visible = False
            del self.path[0]

if __name__ == "__main__":
    sim = simulator(graphics=True)
    sim.initialize("testmap")

    p = []
    for x in range(0, 50):
        p.append([x, x, x])
    if os.environ.get('SIM_NOBROWSER') == None:
        sim.trace(p)
        time.sleep(5)
        sim.delete_trace()
        time.sleep(1)
    """
    print("Initial state:", sim.state())
    sim.move(1,1,1)
    sim.move(0,0,1)
    sim.move(0,0,1)
    sim.attach()
    time.sleep(1)
    print("After moving and attaching:", sim.state())
    sim.release()
    time.sleep(1)
    print("After releasing:", sim.state())
    sim.attach()
    time.sleep(1)
    sim.move(0,1,0)
    time.sleep(1)
    print("After reattaching and flying upwards:", sim.state())
    sim.release()
    print("After releasing again.", sim.state())
    """

