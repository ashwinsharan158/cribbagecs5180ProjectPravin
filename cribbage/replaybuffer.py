import random
import numpy as np
from collections import deque


class replaybuffer:
    """
        Implement replay buffer to store trajectory.
    """

    def __init__(self):
        self.replayBufferSize = 2000
        self._dataTrajectory = deque()
        self.currSize = 0

    def addEpisode(self, episode):
        if self.currSize > self.replayBufferSize:
            self._dataTrajectory.popleft()
        else:
            self.currSize = self.currSize + 1
        self._dataTrajectory.append(episode)

    def sampleBatch(self, batchSize):
        obsStatesList, actionsList, rewardList, nextStatesList, donesList = [], [], [], [], []
        for _ in range(min(self.currSize, batchSize)):
            idx = np.random.randint(0, self.currSize)
            x = self._dataTrajectory[idx]
            obs, action, next_obs, reward, d = x
            obsStatesList.append(obs)
            actionsList.append(action)
            rewardList.append(reward)
            nextStatesList.append(next_obs)
            donesList.append(d)
        return obsStatesList, actionsList, rewardList, nextStatesList, donesList

    def sampleSeqBatch(self, batchSize):
        obsStatesList, actionsList, rewardList, nextStatesList, donesList = [], [], [], [], []
        for _ in range(batchSize):
            idx = np.random.randint(0, self.currSize)
            x = self._dataTrajectory[idx]
            ##for x in reversed(episode):
            obs, action, next_obs, reward, d = x
            obsStatesList.append(obs)
            actionsList.append(action)
            rewardList.append(reward)
            nextStatesList.append(next_obs)
            donesList.append(d)
            # break
        return obsStatesList, actionsList, rewardList, nextStatesList, donesList
        # return np.array(obsStatesList), np.array(actionsList), np.array(rewardList), np.array(nextStatesList), np.array(
        #   donesList)
