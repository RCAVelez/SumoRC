import pybullet as p
import os

class SumoArena:
    def __init__(self, client):
        sumoArenaFile = os.path.join(os.path.dirname(__file__), 'sumoarenamodel.urdf')
        #sumoArenaFile = os.path.join(os.path.dirname(__file__), 'arena.SLDASM/urdf/arena.SLDASM.urdf')
        p.loadURDF(fileName=sumoArenaFile,basePosition=[0, 0, 0],physicsClientId=client)
