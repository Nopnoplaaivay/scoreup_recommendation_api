import random
import os
import json

from model.environment import Environment

class OnlineMemory:
    def __init__(self, env):
        self.batch = []
        self.memory = {}
        self.env = env

    def process_transitions(self, req):

        cur_chapter = req["chapter"]
        user_id = req["user_id"]
        transitions = req["transitions"]

        if cur_chapter not in self.memory:
            self.memory[cur_chapter] = {}
        if user_id not in self.memory:
            self.memory[cur_chapter][user_id] = {"scores": [], "best_score": 0}

        '''Process transitions'''
        for transition in transitions:
            """Prepare transition"""
            raw_state = transition["state"][1:]
            raw_next_state = transition["next_state"][1:]
            raw_action = transition["action"]

            state = self.env.convert_state(raw_state)
            next_state = self.env.convert_state(raw_next_state)
            action = self.env.get_action(raw_action)
            done = transition["done"]
            reward = self.env.reward_func(state, next_state)

            transition = {
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }

            self.batch.append(transition)

    def reset(self):
        self.batch = [] 
'''
Memory: 
{
    "chuong-1": {
        "io1u34b123498u192" : {
            "scores": [0, 0,53, 0,78, 1, 0,88, 0,9],
            "best_score":  
        }
    }

    "chuong-2":
}
'''