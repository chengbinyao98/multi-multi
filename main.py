from environment import Env
from tools import Tools
from DQN import *
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import math

# 图类

if __name__ == '__main__':
    # 实例化
    n = 3
    env = Env()
    tools = Tools()
    with tf.Session() as sess:
        rl = DQN(
            sess=sess,
            s_dim=3 * n,
            a_dim=int(math.pow(env.range,n)),
            batch_size=128,
            gamma=0.99,
            lr=0.01,
            epsilon=0.1,
            replace_target_iter=300
        )
        tf.global_variables_initializer().run()

        # 画图
        plt.ion()
        plt.figure(figsize=(100, 5))    # 设置画布大小
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        # # 实时路况图
        # pos_x = []
        # pos_y = []
        # color_table = ['k', 'b', 'r', 'y', 'g', 'c', 'm']
        # txt_table = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        #
        # reward图
        epi = []
        success = []

        dic_state = env.reset(n, tools)
        for episode in range(10000):
            print('episode',episode)

            # reward图
            suss = 0
            total = 0

            for time in range(20):                                              # 一个回合持续20步
                dic_action = {n:[]}                                             # 实际动作字典
                dic_add_action = {n:[]}                                         # 神经网络动作

                for num in range(len(dic_state[n])):
                    temp_state = tools.get_list(dic_state[n][num])          # 车组中所有车辆状态合成
                    # print(temp_state)
                    add_range = rl.choose_action(temp_state)    # 学习到车组的动作组合
                    dic_add_action[n].append(add_range)                    # 记录车组的动作组合，用于之后的学习

                    # 车组动作组合转换成车辆的单个动作增量
                    add = []
                    b = []
                    for k in range(len(dic_state[n][num])):
                        s = add_range // env.range                          # 商
                        y = add_range % env.range                           # 余数
                        b = b + [y]
                        add_range = s
                    b.reverse()
                    for i in b:
                        add.append(i)

                    # 转换成车辆的单个动作
                    act = []
                    for dim in range(len(dic_state[n][num])):
                        act.append(dic_state[n][num][dim][1] - env.range/2 + add[dim])
                    dic_action[n].append(act)

                # print('更新'n
                dic_state_, dic_reward, unchange_dic_state_, draw_act, draw_pos = env.step(dic_action,dic_state,tools,n)  # dicreward改成一个值
                # unchange_dic_state_:没进行车辆更新之前的车辆下一时刻的状态
                # draw_pos:用于画图，记录（位置更新）之前的车辆位置
                # draw_act:用于画图，记录（位置更新）之前的车辆动作

                # 画图

                plt.sca(ax1)
                ax1.cla()  # 清空画布
                plt.axis([0, 210, 0, 0.1])  # 坐标轴范围
                # x_major_locator = MultipleLocator(5)  # 把x轴的刻度间隔设置为1，并存在变量里
                # plt.gca()  # ax为两条坐标轴的实例
                # plt.xaxis.set_major_locator(MultipleLocator(5))  # 把x轴的主刻度设置为1的倍数
                # figure1.tick_params(axis='both', which='major', labelsize=5)  # 坐标轴字体大小
                y1 = []
                y2 = []
                for i in range(len(draw_pos)):
                    y1.append(0)
                    y2.append(0.02)

                txt = []
                for t in range(len(draw_pos)):
                    draw_act[t] = draw_act[t] * env.per_section
                    txt.append(t)
                plt.scatter(draw_pos, y1, marker="o")  # 画图数据
                plt.scatter(draw_act, y2, marker="o")  # 画图数据
                for m in range(len(draw_pos)):
                    plt.text(draw_pos[m] * 1.005, y1[m] * 1.005, txt[m],
                            fontsize=10, color="k", style="italic", weight="light",
                            verticalalignment='center', horizontalalignment='right', rotation=0)
                for p in range(len(draw_act)):
                    plt.text(draw_act[p] * 1.005, y2[p] * 1.005, txt[p],
                            fontsize=10, color="k", style="italic", weight="light",
                            verticalalignment='center', horizontalalignment='right', rotation=0)
                plt.pause(env.frame_slot)

                if time == 19:
                    done = 1
                else:
                    done = 0

                for numb in range(len(dic_state[n])):
                    l_temp_state = tools.get_list(dic_state[n][numb])
                    l_temp_state_ = tools.get_list(unchange_dic_state_[n][numb])
                    l_temp_action = dic_add_action[n][numb]
                    l_temp_reward = 0
                    for dim in range(len(dic_state[n][numb])):
                        l_temp_reward += dic_reward[n][numb][dim]
                        suss += dic_reward[n][numb][dim]
                        total += env.choose_beam_slot + env.data_beam_slot
                    rl.store_transition_and_learn(l_temp_state, l_temp_action, l_temp_reward, l_temp_state_, done)

                dic_state = dic_state_
                # if done:
                #     break

            plt.sca(ax2)
            ax2.cla()
            # plt.ylim([0, 5000])  # 坐标轴范围
            success.append(suss / total)
            epi.append(len(epi))
            plt.plot(epi, success)
            plt.pause(env.frame_slot)

            # print('成功率',suss/total)






