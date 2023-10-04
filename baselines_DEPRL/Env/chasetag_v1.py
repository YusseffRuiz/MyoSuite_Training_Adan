""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com), Chun Kwang Tan (cktan.neumove@gmail.com)
================================================= """


"""
Modified version of chasetag_v0 environment, made by Adan. 
"""

import collections
import gym
import numpy as np
import pink
from math import sqrt
import random
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0
from myosuite.utils.quat_math import quat2euler, euler2quat


class ChallengeOpponent:
    """
    Training Opponent for the Locomotion Track of the MyoChallenge 2023.
    Contains several different policies. For the final evaluation, an additional
    non-disclosed policy will be used.
    """

    def __init__(self, sim, rng, probabilities: list, min_spawn_distance: float):
        self.dt = 0.01
        self.sim = sim
        self.opponent_probabilities = probabilities
        self.min_spawn_distance = min_spawn_distance
        self.reset_opponent(rng)

    def reset_noise_process(self):
        self.noise_process = pink.ColoredNoiseProcess(
            beta=2, size=(2, 2000), scale=10, rng=self.rng
        )

    def get_opponent_pose(self):
        """
        Get opponent Pose
        :return: The  pose.
        :rtype: list -> [x, y, angle]
        """
        angle = quat2euler(self.sim.data.mocap_quat[0, :])[-1]
        return np.concatenate([self.sim.data.mocap_pos[0, :2], [angle]])

    def set_opponent_pose(self, pose: list):
        """
        Set opponent pose directly.
        :param pose: Pose of the opponent.
        :type pose: list -> [x, y, angle]
        """
        self.sim.data.mocap_pos[0, :2] = pose[:2]
        self.sim.data.mocap_quat[0, :] = euler2quat([0, 0, pose[-1]])

    def move_opponent(self, vel: list):
        """
        This is the main function that moves the opponent and should always be used if you want to physically move
        it by giving it a velocity. If you want to teleport it to a new position, use `set_opponent_pose`.
        :param vel: Linear and rotational velocities in [-1, 1]. Moves opponent
                  forwards or backwards and turns it. vel[0] is assumed to be linear vel and
                  vel[1] is assumed to be rotational vel
        :type vel: list -> [lin_vel, rot_vel].
        """
        self.opponent_vel = vel
        assert len(vel) == 2
        vel[0] = np.abs(vel[0])
        vel = np.clip(vel, -1, 1)
        pose = self.get_opponent_pose()
        x_vel = vel[0] * np.cos(pose[-1] + 0.5 * np.pi)
        y_vel = vel[0] * np.sin(pose[-1] + 0.5 * np.pi)
        pose[0] -= self.dt * x_vel
        pose[1] -= self.dt * y_vel
        pose[2] += self.dt * vel[1]
        pose[:2] = np.clip(pose[:2], -5.5, 5.5)
        self.set_opponent_pose(pose)

    def random_movement(self):
        """
        This moves the opponent randomly in a correlated
        pattern.
        """
        return self.noise_process.sample()

    def static_training_opponent(self):
        self.opponent_policy = "static_stationary"

    def sample_opponent_policy(self):
        """
        Takes in three probabilities and returns the policies with the given frequency.
        """
        rand_num = self.rng.uniform()

        if rand_num < self.opponent_probabilities[0]:
            self.opponent_policy = "static_stationary"
        elif rand_num < self.opponent_probabilities[0] + self.opponent_probabilities[1]:
            self.opponent_policy = "stationary"
        elif (
            rand_num
            < self.opponent_probabilities[0]
            + self.opponent_probabilities[1]
            + self.opponent_probabilities[2]
        ):
            self.opponent_policy = "random"

    def update_opponent_state(self):
        """
        This function executes an opponent step with
        one of the control policies.
        """
        if (
            self.opponent_policy == "stationary"
            or self.opponent_policy == "static_stationary"
        ):
            opponent_vel = np.zeros(
                2,
            )

        elif self.opponent_policy == "random":
            opponent_vel = self.random_movement()

        else:
            raise NotImplementedError(
                f"This opponent policy doesn't exist. Chose: static_stationary, stationary or random. Policy was: {self.opponent_policy}"
            )
        self.move_opponent(opponent_vel)

    def reset_opponent(self, rng=None):
        """
        This function should initially place the opponent on a random position with a
        random orientation with a minimum radius to the model.
        """
        test_time = 2
        if rng is not None:
            self.rng = rng
            self.reset_noise_process()

        self.opponent_vel = np.zeros((2,))
        #self.sample_opponent_policy() ##uncomment line for advanced training
        self.static_training_opponent()  ## created opponent only for basic static training
        dist = 0
        while dist < self.min_spawn_distance:
            pose = [
                self.rng.uniform(-5, 5),
                self.rng.uniform(-5, 5),
                self.rng.uniform(-2 * np.pi, 2 * np.pi),
            ]
            dist = np.linalg.norm(pose[:2] - self.sim.data.body("root").xpos[:2])
        if self.opponent_policy == "static_stationary":
            if test_time == 1:
                pose[:] = [
                    0,
                    3,
                    1,
                ]  ### Change static position of the objective [x, y, orientation] , only x,y are used, orientation doesn't really matter
            if test_time == 2:
                randTemp = random.randint(0, 10)
                if randTemp > 4 and randTemp <= 8:
                    pose[:] = [random.randint(2, 5), random.randint(-5, 0), 5]
                elif randTemp <= 4:
                    pose[:] = [random.randint(-4, 4), -random.randint(2, 5), 0]
                else:
                    pose[:] = [0, 3, 1]

        self.set_opponent_pose(pose)
        self.opponent_vel[:] = 0.0


class ChaseTagEnvV0(WalkEnvV0):
    DEFAULT_OBS_KEYS = [
        "internal_qpos",
        "qvel", # original name internal_qvel
        "grf",
        "com_vel",
        "feet_heights",
        "height",
        "feet_rel_positions",
        "torso_angle",
        "opponent_pose",
        "opponent_vel",
        "qpos_without_xy", ### Same as qpos_without_xy, changing name from model_root_pos
        "model_root_vel",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
    ]

    # You can change reward weights here
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "vel_reward": 50.0,
        # "cyclic_hip": -10,
        "falling" : -100,
        "solved": 100,
        "done": 0,
        "distance": -0.1,
        "lose": -100,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # This flag needs to be here to prevent the simulation from starting in a done state
        # Before setting the key_frames, the model and opponent will be in the cartesian position,
        # causing the step() function to evaluate the initialization as "done".
        self.startFlag = False

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(
            model_path=model_path, obsd_model_path=obsd_model_path, seed=seed
        )
        self._setup(**kwargs)

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        opponent_probabilities=[0.1, 0.45, 0.45],
        reset_type="none",
        win_distance=0.5,
        min_spawn_distance=2,
        **kwargs,
    ):
        self._setup_convenience_vars()

        self.reset_type = reset_type
        self.maxTime = 20

        self.win_distance = win_distance
        self.grf_sensor_names = ["r_foot", "r_toes", "l_foot", "l_toes"]
        self.opponent = ChallengeOpponent(
            sim=self.sim,
            rng=self.np_random,
            probabilities=opponent_probabilities,
            min_spawn_distance=min_spawn_distance,
        )
        self.success_indicator_sid = self.sim.model.site_name2id("opponent_indicator")
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            reset_type=reset_type,
            **kwargs,
        )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0
        self.startFlag = True

    def get_obs_dict(self, sim):
        obs_dict = {}

        # Time
        obs_dict["time"] = np.array([sim.data.time])

        # proprioception
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["internal_qpos"] = sim.data.qpos[7:35].copy()
        obs_dict["qvel"] = sim.data.qvel[6:34].copy() * self.dt
        obs_dict["grf"] = self._get_grf().copy()
        obs_dict["feet_heights"] = self._get_feet_heights().copy() ## added by Adan
        obs_dict["height"] = np.array([self._get_height()]).copy() ## added by Adan
        obs_dict["feet_rel_positions"] = self._get_feet_relative_position().copy()
        obs_dict["com_vel"] = np.array([self._get_com_velocity().copy()])
        obs_dict["torso_angle"] = self.sim.data.body("pelvis").xquat.copy()
        # obs_dict['com'] = sim.data. ## data in anything that changes with time, otherwise is model

        obs_dict["muscle_length"] = self.muscle_lengths()
        obs_dict["muscle_velocity"] = self.muscle_velocities()
        obs_dict["muscle_force"] = self.muscle_forces()

        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        # exteroception
        obs_dict["opponent_pose"] = self.opponent.get_opponent_pose()[:].copy()
        obs_dict["opponent_vel"] = self.opponent.opponent_vel[:].copy()
        obs_dict["qpos_without_xy"] = sim.data.qpos[:2].copy()
        obs_dict["model_root_vel"] = sim.data.qvel[:2].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        """
        Rewards are computed from here, using the <self.weighted_reward_keys>.
        These weights can either be set in this file in the
        DEFAULT_RWD_KEYS_AND_WEIGHTS dict, or when registering the environment
        with gym.register in myochallenge/__init__.py
        """
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        fall_action = self._get_fall()
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        win_cdt = self._win_condition()
        lose_cdt = self._lose_condition()
        score = self._get_score(float(self.obs_dict["time"])) if win_cdt else 0

        # reward, in modification, add other variables
        distance = np.linalg.norm(
            obs_dict["qpos_without_xy"][..., :2] - obs_dict["opponent_pose"][..., :2]
        )

        rwd_dict = collections.OrderedDict(
            (
                # Perform reward tuning here --
                # Update Optional Keys section below
                # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
                # Example: simple distance function
                # Optional Keys
                ("vel_reward", vel_reward),
                # ("cyclic_hip", cyclic_hip),
                ("act_reg", act_mag),
                ("lose", lose_cdt),
                ("distance", distance * distance),
                ("falling", fall_action),
                # Must keys
                ("sparse", score),
                ("solved", win_cdt),
                ("done", self._get_done()),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        # Success Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :] = (
            np.array([0, 2, 0, 0.1]) if rwd_dict["solved"] else np.array([2, 0, 0, 0])
        )

        return rwd_dict

    def get_metrics(self, paths):
        """
        Evaluate paths and report metrics
        """
        # average sucess over entire env horizon
        score = np.mean([np.sum(p["env_infos"]["rwd_dict"]["sparse"]) for p in paths])
        points = np.mean([np.sum(p["env_infos"]["rwd_dict"]["solved"]) for p in paths])
        times = np.mean(
            [np.round(p["env_infos"]["obs_dict"]["time"][-1], 2) for p in paths]
        )
        # average activations over entire trajectory (can be shorter than horizon, if done) realized

        metrics = {
            "score": score,
            "points": points,
            "times": times,
        }
        return metrics

    def step(self, *args, **kwargs):
        self.opponent.update_opponent_state()
        obs, reward, done, info = super().step(*args, **kwargs)
        return obs, reward, done, info

    def reset(self):
        if self.reset_type == "random":
            qpos, qvel = self._get_randomized_initial_state()
        elif self.reset_type == "init":
            qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel)
        self.opponent.reset_opponent(self.np_random)
        return obs

    def viewer_setup(self, *args, **kwargs):
        """
        Setup the default camera
        """
        distance = 20.0
        azimuth = 90
        elevation = -15
        lookat = None
        self.sim.renderer.set_free_camera_settings(
            distance=distance, azimuth=azimuth, elevation=elevation, lookat=lookat
        )
        render_tendon = True
        render_actuator = True
        self.sim.renderer.set_viewer_settings(
            render_actuator=render_actuator, render_tendon=render_tendon
        )

    def _get_randomized_initial_state(self):
        # randomly start with flexed left or right knee
        if self.np_random.uniform() < 0.5:
            qpos = self.sim.model.key_qpos[2].copy()
            qvel = self.sim.model.key_qvel[2].copy()
        else:
            qpos = self.sim.model.key_qpos[3].copy()
            qvel = self.sim.model.key_qvel[3].copy()

        # randomize qpos coordinates
        # but dont change height or rot state
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def _setup_convenience_vars(self):
        """
        Convenience functions for easy access. Important: There will be no access during the challenge evaluation to this,
        but the values can be hard-coded, as they do not change over time.
        """
        self.actuator_names = np.array(self._get_actuator_names())
        self.joint_names = np.array(self._get_joint_names())
        self.muscle_fmax = np.array(self._get_muscle_fmax())
        self.muscle_lengthrange = np.array(self._get_muscle_lengthRange())
        self.tendon_len = np.array(self._get_tendon_lengthspring())
        self.musc_operating_len = np.array(self._get_muscle_operating_length())

    def _get_done(self):
        if self._lose_condition():
            return 1
        if self._win_condition():
            return 1
        return 0

    def _lose_condition(self):
        # fall condition for phase 1
        if self.sim.data.body("pelvis").xpos[2] < 0.5:
            return 1
        root_pos = self.sim.data.body("pelvis").xpos[:2]
        return (
            1
            if float(self.obs_dict["time"]) >= self.maxTime
            or (np.abs(root_pos[0]) > 6.5 or np.abs(root_pos[1]) > 6.5)
            #or self._get_fall()
            else 0
        )

    def _win_condition(self):
        root_pos = self.sim.data.body("pelvis").xpos[:2]
        opp_pos = self.obs_dict["opponent_pose"][..., :2]
        return (
            1
            if np.linalg.norm(root_pos - opp_pos) <= self.win_distance
            and self.startFlag
            else 0
        )

    # Helper functions

    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]


    def _get_body_mass(self):
        """
        Get total body mass of the biomechanical model.
        :return: the weight
        :rtype: float
        """
        return self.sim.model.body("root").subtreemass

    def _get_fall(self):
        """
        Get center-of-mass height.
        """
        # height = self._get_com()[2]
        height = self.sim.data.body("pelvis").xpos[2]

        """
        Get feet height
        """
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("talus_r")
        feet_height = (
            self.sim.data.body_xpos[foot_id_l][2]
            + self.sim.data.body_xpos[foot_id_r][2]
        ) / 2

        """
        Get difference between knees and feet, if they are similar, the model has fallen down
        """
        if (height - feet_height) < 0.61:
            return 1
        else:
            return 0

    def _get_com(self):
        """
        Compute the center of mass of the robot.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com = self.sim.data.xipos
        return np.sum(mass * com, 0) / np.sum(mass)

    def _get_feet_heights(self):
        """
        Get the height of both feet.
        """
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("talus_r")
        return np.array(
            [
                self.sim.data.body_xpos[foot_id_l][2],
                self.sim.data.body_xpos[foot_id_r][2],
            ]
        )

    def _get_height(self):
        """
        Get center-of-mass height.
        """
        return self._get_com()[2]

    def _get_feet_relative_position(self):
        """
                Get the feet positions relative to the pelvis.
                """
        foot_id_l = self.sim.model.body_name2id("talus_l")
        foot_id_r = self.sim.model.body_name2id("talus_r")
        pelvis = self.sim.model.body_name2id("pelvis")
        return np.array(
            [
                self.sim.data.body_xpos[foot_id_l] - self.sim.data.body_xpos[pelvis],
                self.sim.data.body_xpos[foot_id_r] - self.sim.data.body_xpos[pelvis],
            ]
        )

    def _get_score(self, time):
        time = np.round(time, 2)
        return 1 - (time / self.maxTime)

    def _get_muscle_lengthRange(self):
        return self.sim.model.actuator_lengthrange.copy()

    def _get_tendon_lengthspring(self):
        return self.sim.model.tendon_lengthspring.copy()

    def _get_muscle_operating_length(self):
        return self.sim.model.actuator_gainprm[:, 0:2].copy()

    def _get_muscle_fmax(self):
        return self.sim.model.actuator_gainprm[:, 2].copy()

    def _get_grf(self):
        return np.array(
            [
                self.sim.data.sensor(sens_name).data[0]
                for sens_name in self.grf_sensor_names
            ]
        ).copy()

    def _get_pelvis_angle(self):
        return self.sim.data.body("pelvis").xquat.copy()

    def _get_joint_names(self):
        """
        Return a list of joint names according to the index ID of the joint angles
        """
        return [
            self.sim.model.joint(jnt_id).name
            for jnt_id in range(1, self.sim.model.njnt)
        ]

    def _get_actuator_names(self):
        """
        Return a list of actuator names according to the index ID of the actuators
        """
        return [
            self.sim.model.actuator(act_id).name
            for act_id in range(1, self.sim.model.na)
        ]
