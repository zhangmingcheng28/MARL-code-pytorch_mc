#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：multi-agent RL
@File ：simple_spread_with_obstacle.py
@Author ：
@Date : 2023/11/15
"""
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        '''
        设定世界中的障碍物位置
        '''
        #world.obstacles = np.array([[0.8,0.8], [0.8,-0.8], [-0.8,0.8], [-0.8,-0.8]])

        obstacles = []
        h_obstacle_x = np.linspace(-0.7, 0.7, num=50)
        h_obstacle_y = np.full((len(h_obstacle_x)), 0.7)
        obstacles.append(np.column_stack((h_obstacle_x,h_obstacle_y)))

        h_obstacle_y = np.linspace(-1.0, 0.0, num=40)
        h_obstacle_x = np.full((len(h_obstacle_y)), 0.5)
        obstacles.append(np.column_stack((h_obstacle_x, h_obstacle_y)))

        h_obstacle_y = np.linspace(-1.0, 1.0, num=100)
        h_obstacle_x = np.full((len(h_obstacle_y)), -1.0)
        obstacles.append(np.column_stack((h_obstacle_x, h_obstacle_y)))

        h_obstacle_y = np.linspace(-1.0, 1.0, num=100)
        h_obstacle_x = np.full((len(h_obstacle_y)), 1.0)
        obstacles.append(np.column_stack((h_obstacle_x, h_obstacle_y)))

        h_obstacle_x = np.linspace(-1.0, 1.0, num=100)
        h_obstacle_y = np.full((len(h_obstacle_x)), 1.0)
        obstacles.append(np.column_stack((h_obstacle_x, h_obstacle_y)))

        h_obstacle_x = np.linspace(-1.0, 1.0, num=100)
        h_obstacle_y = np.full((len(h_obstacle_x)), -1.0)
        obstacles.append(np.column_stack((h_obstacle_x, h_obstacle_y)))

        obstacles = np.vstack(obstacles)
        world.obstacles = obstacles

        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        #world.collaborative = True  #取消合作，各算各的

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            '''
            agent.radar_range: 无人机感知范围，为了方便计算，这边范围是个以无人机为中心四个像限的矩形,见observation
            '''
            agent.radar_range = 0.3

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            #agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            while True:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                if self.is_collision_with_obstacle(agent, world.obstacles) ==False:
                    break

            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            '''
            agent.state.ladar_info 用于保存四个像限的雷达观测信息
            agent.state.done 用于判断是否坠机
            agent.max_speed 是最大飞行速度，前后两步之间差分的速度不能通过障碍物
            '''
            agent.state.ladar_info = np.zeros(4)
            agent.state.done = False
            agent.max_speed = agent.size*2 / world.dt

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision_with_obstacle(self,agent,obstacles):
        '''
        计算agent是否与障碍物碰撞
        inputs:
            agent: 与is_collision函数中的agent相同
            obstacles：为world.obstacles, np.array
        '''
        # dist_list = []
        # for obstacle in obstacles:
        #     delta_pos = agent.state.p_pos-obstacle
        #     dist = np.sqrt(np.sum(np.square(delta_pos)))
        #     dist_list.append(dist)

        dist_list = np.sqrt(np.sum((obstacles - agent.state.p_pos) ** 2, axis=1))
        #最小的距离大于agent尺寸，则没有碰撞，返回false
        if min(dist_list) > agent.size:
            return False
        else:
            return True

    def is_collision(self, agent1, agent2):
        '''
        排除自己撞自己
        '''
        if agent1.name == agent2.name:
            return False

        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos))) #平方和开根号
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 2
        '''
        1118将最近landmark点距离改为对应landmakr点距离
        '''
        agent_num = int(agent.name.split(' ')[-1])
        l = world.landmarks[agent_num]
        dists = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
        rew -= dists
        if dists<agent.size:
            rew +=1
        #print(agent_num,dists)

        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1

            '''
            agent.state.done 判断飞机是否撞到障碍物
            如果撞到 agent.state.done = True,且在后续步长中持续会 -rew
            
            
            agent.state.ladar_info 中记录下四个像限是否有障碍物，为了让agent提前远离障碍物
            每个像限中有障碍物的时候 ladar_info 为 1，就降低一个小的reward；
            没有障碍物，ladar_info 为 0， 没有给reward
            '''

            if agent.state.done == True:
                rew -= 5

            elif self.is_collision_with_obstacle(agent,world.obstacles):
                rew -= 5
                agent.state.done = True

            rew -=  np.sum(agent.state.ladar_info)*0.05

        return rew

    def observation(self, agent, world):
        #------------------------------------------------
        '''
        agent.radar_range: 无人机感知范围，为了方便计算，这边范围是个以无人机为中心四个像限的矩形
        '''
        # radar_info = []
        # #agent.state.p_pos = np.array([0.6,0.6])
        #
        # agent_front_point = np.array([agent.state.p_pos[0] + agent.radar_range, agent.state.p_pos[1]])
        # agent_back_point =  np.array([agent.state.p_pos[0] - agent.radar_range, agent.state.p_pos[1]])
        # agent_left_point =  np.array([agent.state.p_pos[0] , agent.state.p_pos[1]+ agent.radar_range])
        # agent_right_point = np.array([agent.state.p_pos[0] , agent.state.p_pos[1]- agent.radar_range])
        #
        # def is_point_in_rectangle(rectangle, points):
        #     rectangle_min_x = min(rectangle[0][0],rectangle[1][0])
        #     rectangle_min_y = min(rectangle[0][1], rectangle[1][1])
        #     rectangle_max_x = max(rectangle[0][0], rectangle[1][0])
        #     rectangle_max_y = max(rectangle[0][1], rectangle[1][1])
        #
        #     # points = points[points[:, 0] > rectangle_min_x]
        #     # points = points[points[:, 0] < rectangle_max_x]
        #     # points = points[points[:, 1] > rectangle_min_y]
        #     # points = points[points[:, 1] < rectangle_max_y]
        #     # print('old: ',len(points))
        #     # if len(points)>0:
        #     #     return 1
        #     # else:
        #     #     return 0
        #
        #     top_left = np.array([rectangle_min_x,rectangle_min_y])
        #     bottom_right = np.array([rectangle_max_x,rectangle_max_y])
        #
        #     is_inside = np.all((points >= top_left) & (points <= bottom_right), axis=1) #快30%左右
        #
        #     if len(is_inside)>0:
        #         return 1
        #     else:
        #         return 0
        #
        # radar_info.append(is_point_in_rectangle([agent_front_point, agent_left_point],world.obstacles))
        # radar_info.append(is_point_in_rectangle([agent_front_point, agent_right_point], world.obstacles))
        # radar_info.append(is_point_in_rectangle([agent_back_point, agent_left_point],world.obstacles))
        # radar_info.append(is_point_in_rectangle([agent_back_point, agent_right_point], world.obstacles))
        # radar_info = np.array(radar_info)
        # agent.state.ladar_info = radar_info

        # ------------------------------------------------

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        #return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [agent.state.ladar_info] + entity_pos + other_pos + comm)

    def is_done(self,agent,world):
        return agent.state.done