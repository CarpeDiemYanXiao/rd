from model.net import Qnet
from env.low_level_env import Testing_env, load_second_features, position_to_action_float
import math
from tool.demonstration import (
    making_multi_level_dp_demonstration,
    make_q_table,
    get_dp_action_from_qtable,
    make_q_table_reward,
)
from RL.util.graph import get_test_contrast_curve_high_level
from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse
import os
import torch
import sys

# 由于在for循环中 np计算会存有一点点剩余 导致剩余的position不全是0 进而导致出现买不全的现象
sys.path.append(".")


def load_minute_features(dataset_name):
    """Load minute-level features for the given dataset."""
    return np.load(f"data/feature/{dataset_name}/minitue_feature.npy").tolist()


def build_model_path_list_dict(dataset_name):
    """Dynamically build the 5x5 model path dictionary for a given dataset.
    Keys 0-4 correspond to initial_action positions.
    Each key maps to a list of 5 model paths (model_0..model_4).
    """
    d = {}
    for pos in range(5):
        d[pos] = [
            f"result_risk/{dataset_name}/potential_model/initial_action_{pos}/model_{m}.pth"
            for m in range(5)
        ]
    return d


# Module-level defaults (used as fallbacks only)
transcation_cost = 0.0001
back_time_length = 1
max_holding_number = 0.5
action_dim = 5


class high_level_testing_env(Testing_env):
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_name="NVDA",
        high_level_tech_indicator_list=None,
        low_level_tech_indicator_list=None,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        action_dim=action_dim,
        early_stop=0,
        initial_action=0,
        model_path_list_dict=None,
    ):
        if high_level_tech_indicator_list is None:
            high_level_tech_indicator_list = load_minute_features(dataset_name)
        if low_level_tech_indicator_list is None:
            low_level_tech_indicator_list = load_second_features(dataset_name)
        if model_path_list_dict is None:
            model_path_list_dict = build_model_path_list_dict(dataset_name)
        self.device = "cpu"
        self.high_llevel_tech_inidcator_list = high_level_tech_indicator_list
        self.low_level_agent_list_dict = {}
        for key in model_path_list_dict:
            self.low_level_agent_list_dict[key] = []
            for model_path in model_path_list_dict[key]:
                model = Qnet(
                    int(len(low_level_tech_indicator_list)), int(action_dim), 128
                ).to("cpu")
                model.load_state_dict(
                    torch.load(
                        model_path,
                        map_location=torch.device("cpu"),
                    )
                )
                self.low_level_agent_list_dict[key].append(model)

        super(high_level_testing_env, self).__init__(
            df=df,
            tech_indicator_list=low_level_tech_indicator_list,
            transcation_cost=transcation_cost,
            back_time_length=back_time_length,
            max_holding_number=max_holding_number,
            action_dim=action_dim,
            early_stop=early_stop,
            initial_action=initial_action,
        )
        self.action_space = spaces.Discrete(
            len(self.low_level_agent_list_dict))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(back_time_length * len(self.tech_indicator_list),),
        )
        self.macro_action_history = []
        self.timestamp_history = []
        self.macro_state_history = []
        self.macro_reward_history = []
        # log for model
        self.chosen_model_history = []

    def reset(self):
        self.macro_action_history = []
        self.timestamp_history = []
        self.state, self.info = super(high_level_testing_env, self).reset()
        high_level_state = self.data[self.high_llevel_tech_inidcator_list].values
        self.info["high_level_state"] = high_level_state
        self.chosen_model_history = []

        return self.state, self.info

    def step(self, action):
        position_idx = int(round(position_to_action_float(
            self.position, self.action_dim, self.max_holding_number)))
        position_idx = max(0, min(position_idx, self.action_dim - 1))
        self.chosen_model = self.low_level_agent_list_dict[position_idx][action]
        self.chosen_model_history.append(position_idx * 5 + action)
        reward_mintue = 0
        current_minute = self.data.iloc[-1].timestamp.minute
        current_hour = self.data.iloc[-1].timestamp.hour
        while self.terminal == False:
            ts = self.data.iloc[-1].timestamp
            # Break at minute boundary: when second == 59 OR when minute/hour changes
            # (handles day boundaries in US stock data)
            if ts.second == 59:
                break
            if ts.minute != current_minute or ts.hour != current_hour:
                break
            self.timestamp_history.append(ts)
            self.macro_state_history.append(self.state)
            macro_action = self.pose_macro_action(self.state, self.info)
            self.macro_action_history.append(macro_action)
            self.state, reward, done, self.info = super(
                high_level_testing_env, self
            ).step(macro_action)
            self.macro_reward_history.append(reward)
            reward_mintue += reward
        if self.terminal == True:
            return self.state, reward_mintue, done, self.info
        else:
            self.timestamp_history.append(self.data.iloc[-1].timestamp)
            self.macro_state_history.append(self.state)
            macro_action = self.pose_macro_action(self.state, self.info)
            self.macro_action_history.append(macro_action)
            self.state, reward, done, self.info = super(
                high_level_testing_env, self
            ).step(macro_action)
            self.macro_reward_history.append(reward)
            reward_mintue += reward
            self.info["high_level_state"] = self.data[
                self.high_llevel_tech_inidcator_list
            ].values
        return self.state, reward_mintue, done, self.info

    def pose_macro_action(self, state, info):
        x = torch.unsqueeze(torch.FloatTensor(
            state).reshape(-1), 0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float(), 0
        ).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]), 0
        ).to(self.device)
        actions_value = self.chosen_model.forward(
            x, previous_action, avaliable_action)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action


if __name__ == "__main__":
    dataset = "NVDA"
    df = pd.read_feather(f"data/{dataset}/valid.feather")
    reward_list = []
    test_env = high_level_testing_env(df, dataset_name=dataset)
    state, info = test_env.reset()
    done = False
    while not done:
        state, reward_mintue, done, info = test_env.step(1)
        reward_list.append(reward_mintue)
    (
        portfit_magine,
        final_balance,
        required_money,
        commission_fee,
    ) = test_env.get_final_return_rate(slient=False)
    get_test_contrast_curve_high_level(
        df, "test_high_level.pdf", reward_list, test_env.required_money
    )
