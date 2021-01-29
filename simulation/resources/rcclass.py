import pybullet as p
import os
import math
import numpy as np
import pybullet_data

class RCClass:
    def __init__(self, client, basePosition=[0, 0, 0.1], baseOrientation=[0,0,0,0]):
        self.client = client
        rcFile = os.path.join(os.path.dirname(__file__), 'assem4.SLDASM/urdf/assem4.SLDASM.urdf')
        self.car = p.loadURDF(fileName=rcFile, basePosition=basePosition, baseOrientation=baseOrientation, physicsClientId=client, flags=p.URDF_USE_SELF_COLLISION)

        #only for testing
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # startPos = [1,0,0.5]
        # startOrientation = p.getQuaternionFromEuler([0,0,0])
        # self.boxId = p.loadURDF("r2d2.urdf",startPos, startOrientation, flags=p.URDF_USE_SELF_COLLISION)
        #
        # startPos = [0,1,0.5]
        # startOrientation = p.getQuaternionFromEuler([0,0,0])
        # self.boxId2 = p.loadURDF("r2d2.urdf",startPos, startOrientation, flags=p.URDF_USE_SELF_COLLISION)
        #
        # startPos = [-1,0,0.5]
        # startOrientation = p.getQuaternionFromEuler([0,0,0])
        # self.boxId3 = p.loadURDF("r2d2.urdf",startPos, startOrientation, flags=p.URDF_USE_SELF_COLLISION)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # get joint info
        # print(p.getNumJoints(self.car))
        # for joint in range(0,p.getNumJoints(self.car)):
        #     print(p.getJointInfo(self.car,joint))
        # print('\n\n\n\n')
        #0 -> front left -> wheel0
        #1 -> front right -> wheel1
        #2 -> back left -> wheel2
        #3 -> back right -> wheel3
        self.drive_joints = [0, 1, 2, 3]

    #action to velocity for continuous movement
    #0 -> -10
    #1 -> 0
    #2 -> 10

    def applyAction(self, action):
        maxForce = 500
        p.setJointMotorControlArray(bodyUniqueId=self.car,
                                    jointIndices=[0,1,2,3],
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=action,
                                    forces=[maxForce,maxForce,maxForce,maxForce])

    def getObservation(self):
        #Get the position and orientation of the car in the simulation
        position, orientation = p.getBasePositionAndOrientation(self.car, self.client)

        #Use rayCast as sensors
        rayFrom = []
        rayTo = []
        numRays = 16 #number of rays
        rayLen = 154 #length of dohyo
        #rayIds = []
        #rayHitColor = [1, 0, 0]
        #rayMissColor = [0, 1, 0]

        for i in range(numRays):
          rayFrom.append(position)
          rayTo.append([
              rayLen * math.sin(2.0 * math.pi * float(i) / numRays),
              rayLen * math.cos(2.0 * math.pi * float(i) / numRays),
              0.5
          ])
          #rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor))

        results = p.rayTestBatch(rayFrom, rayTo)

        #Only for visualizing rays and results
        observations = []
        count = 0.0
        for result in results:
            sensor = [count]
            count += 1.0
            sensor.append(position[0])
            sensor.append(position[1])
            sensor.append(position[2])
            if result[0] > 1:
                #something detected in that sensor
                #print(result)
                sensor.append(float(result[0]))
                sensor.append(float(result[1]))
                sensor.append(result[2])
                sensor.append(result[3][0])
                sensor.append(result[3][1])
                sensor.append(result[3][2])
                sensor.append(result[4][0])
                sensor.append(result[4][1])
                sensor.append(result[4][2])

                #sensors.append(result)
            else:
                #nothing was detected in that sensor
                for i in range(0,9):
                    sensor.append(0.0)
                #sensors.append((0, 0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
            observations.append(sensor)

        #observation = tuple([position, tuple(sensors)])

        # for i in range(numRays):
        #   hitObjectUid = results[i][0]
        #
        #   if (hitObjectUid < 0):
        #     hitPosition = [0, 0, 0]
        #     p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
        #   else:
        #     hitPosition = results[i][3]
        #     p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])
        #
        # p.removeAllUserDebugItems()z
        return observations
