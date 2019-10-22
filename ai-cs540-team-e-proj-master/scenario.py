#!/usr/bin/env python

import numpy as np
import sim

class scenario_cityfire:

    def __init__(self, in_file=None, graphics=True):
        self.simul = sim.simulator(graphics = graphics)
        if in_file is not None:
            self.simul.initialize(in_file)

    def generate_building(self, xpos, zpos, width, height, fire=0):
        fireadded = False
        for i in range(xpos, xpos+width):
            if i < -50 or i > 50: break
            for j in range(zpos, zpos+width):
                if j < -50 or j > 50: break
                for k in range(0, height):
                    self.simul.add(i,k,j, "gray")
                if fire and not fireadded:
                    self.simul.add(i, k+1, j, "orange")
                    fireadded = True

    def generate_lake(self, xpos, zpos, width1, width2):
        for i in range (xpos, xpos + width1):
            if i < -50 or i > 50: break
            for j in range(zpos, zpos + width2):
                if j < -50 or j > 50: break
                self.simul.add(i, 0, j, "blue")
                self.simul.lakes.append((i,0,j))

    def generate_river(self, side):
        # use x axis
        if side == 0 or side == 2:
            zpos = -50
            if side == 2: zpos = 50
            for xpos in range(-50, 51):
                self.simul.add(xpos, 0, zpos, "blue")
        if side == 1 or side == 3:
            xpos = -50
            if side == 3: xpos = 50
            for zpos in range(-50, 51):
                self.simul.add(xpos, 0, zpos, "blue")

    def generate_scenario(self, buildings = 100, lakes = 20, seed=None):
        if seed is not None:
            np.random.seed(seed)

        for x in range(0, lakes):
            xpos = np.random.randint(-50, 50)
            zpos = np.random.randint(-50, 50)
            # if the position is already taken, pick a different one
            mapx,mapy,mapz = self.simul.togrid(xpos, 0, zpos)
            while np.any(self.simul.scenario[mapx:mapx+1, mapy, mapz:mapz+1]):
                xpos = np.random.randint(-50, 50)
                zpos = np.random.randint(-50, 50)
                mapx,mapy,mapz = self.simul.togrid(xpos, 0, zpos)
            self.generate_lake(xpos, zpos, 1, 1)

        for b in range(0, buildings):
            xpos = np.random.randint(-50, 50)
            zpos = np.random.randint(-50, 50)
            width = np.random.randint(1, 4)
            height = np.random.randint(10, 48)
            dofire = 1 if (b % 2) == 0 else 0

            mapx, mapy, mapz = self.simul.togrid(xpos, 0, zpos)
            while np.any(self.simul.scenario[mapx:mapx+width, mapy:mapy+height+dofire, mapz:mapz+width]):
                xpos = np.random.randint(-50, 50)
                zpos = np.random.randint(-50, 50)
                mapx, mapy, mapz = self.simul.togrid(xpos, 0, zpos)
            self.generate_building(xpos, zpos, width, height, fire=dofire)

        # place the drone randomly if it hasn't already been placed
        if self.simul.drone is None:
            xpos = np.random.randint(-50, 50)
            ypos = np.random.randint(0,50)
            zpos = np.random.randint(-50, 50)
            mapx,mapy,mapz = self.simul.togrid(xpos, ypos, zpos)
            while np.any(self.simul.scenario[mapx:mapx+1, mapy:mapy+1, mapz:mapz+1]):
                xpos = np.random.randint(-50, 50)
                ypos = np.random.randint(0,50)
                zpos = np.random.randint(-50, 50)
                mapx,mapy,mapz = self.simul.togrid(xpos, ypos, zpos)
            self.simul.add(xpos, ypos, zpos, "drone")

    def generate_river_scenario(self, rivers = 1, buildings = 100, seed=None):
        if seed is not None:
            np.random.seed(seed)

        for i in range(rivers):
            self.generate_river(i)

        for b in range(0, buildings):
            xpos = np.random.randint(-50, 50)
            zpos = np.random.randint(-50, 50)
            width = np.random.randint(1, 4)
            height = np.random.randint(10, 48)
            dofire = 1 if (b % 2) == 0 else 0

            mapx, mapy, mapz = self.simul.togrid(xpos, 0, zpos)
            while np.any(self.simul.scenario[mapx:mapx+width, mapy:mapy+height+dofire, mapz:mapz+width]):
                xpos = np.random.randint(-50, 50)
                zpos = np.random.randint(-50, 50)
                mapx, mapy, mapz = self.simul.togrid(xpos, 0, zpos)
            self.generate_building(xpos, zpos, width, height, fire=dofire)

        # place the drone randomly if it hasn't already been placed
        if self.simul.drone is None:
            xpos = np.random.randint(-50, 50)
            ypos = np.random.randint(0,50)
            zpos = np.random.randint(-50, 50)
            mapx,mapy,mapz = self.simul.togrid(xpos, ypos, zpos)
            while np.any(self.simul.scenario[mapx:mapx+1, mapy:mapy+1, mapz:mapz+1]):
                xpos = np.random.randint(-50, 50)
                ypos = np.random.randint(0,50)
                zpos = np.random.randint(-50, 50)
                mapx,mapy,mapz = self.simul.togrid(xpos, ypos, zpos)
            self.simul.add(xpos, ypos, zpos, "drone")

    def save_map(self, filename="cityfire_map"):
        of = open(filename, 'w')
        for k,v in self.simul.env.items():
            of.write(str(k[0]) + "," + str(k[1]) + "," + str(k[2]) + " " + v.color + "\n")
        of.close()


if __name__ == "__main__":
    scene = scenario_cityfire()
    scene.generate_river_scenario(rivers=4)
    print(scene.simul.blocks)
