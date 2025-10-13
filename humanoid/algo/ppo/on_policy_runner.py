# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import os #quản lí đường dẫn file
import time #đo thời gian rollout & learning
import torch #core deep learning framework
# import wandb 
import statistics #tính mean của reward buffer
from collections import deque # dùng quêu cố đinh kích thước cho reward/length buffer
from datetime import datetime # tạo timestamp cho tên experiment
from .ppo import PPO # class thuật toán PPO
from .actor_critic import ActorCritic # Neural network model
from humanoid.algo.vec_env import VecEnv # interface môi trường vertorized
from torch.utils.tensorboard import SummaryWriter # Ghi log TensorBoard


class OnPolicyRunner:

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        # env: môi trường vectorized, train_cfg: config dict chứa toàn bộ hyperparameters, log_dir: đường dẫn lưu logs & checkpoint
        # device: cpu hoặc cuda

        # lấy tham số config chung chung
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg


        # self.wandb_run_name = (
        #     datetime.now().strftime("%b%d_%H-%M-%S")
        #     + "_"
        #     + train_cfg["runner"]["experiment_name"]
        #     + "_"
        #     + train_cfg["runner"]["run_name"]
        # )
        self.device = device
        self.env = env

        # xác định số chiều critic observation, xác định nên tạo bao nhiêu chiều observation
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        # Lấy giá trị của biến cấu hình policy_class_name là parse sang class mới
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic

        # đối tượng của chúng ta đã khai báo ở trên và chuyển model lên trên gpu
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # lấy giá trị của biến algorithm_class_name nhằm mục đích để lấy được tên class áp dụng thuật toán
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO

        # khởi động một instance của lớp thuật toán học được chỉ định trong cấu hình
        # self alg sẽ tự động quản lý toàn bộ quá trình học của policy chính sách và value function trong thuật toán PPO
        # mục đích: sau khi khởi tạo self.alg sẽ có các phương thức như act() (để lấy hành động),
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        # Thiết lập số bước steps thu thập dữ liệu trong mỗi môi trường env cho mỗi iteration học. Giá trị này được lấy từ self.cfg 
        # Ngữ cảnh: trong mỗi PPO, mỗi iteration học bao gồm một giai đoạn thu thập dữ liệu và mõi giai đoạn cập nhật mô hinhf
        # và num_steps_per_env sẽ quyết định độ dài của rollout trong mỗi môi trường 
        self.num_steps_per_env = self.cfg["num_steps_per_env"]

        # Thiết lập khoản thời gian interval để lưu trạng thái mô hình (checkpoint) trong quá trình học 
        # Trong rl việc lưu mô hình định kì giúp tránh mất dữ liệu nếu quá trình học bị gián đoạn 
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        # khởi tạo bộ nhớ lưu trữ storage để quản lý dữ liệu rollout. Trong PPO, thuật toán cần được thu thập dữ liệu từ môi trường trước khi thực hiện bước
        # cập nhật mô hình. Bộ nhớ này giúp lưu trữ dữ liệu đó một cách hiệu quả
        self.alg.init_storage(
            self.env.num_envs, # số lượng môi trường chạy song song
            self.num_steps_per_env, # Số lượng bước steps thu thập dữ liệu trong mỗi môi trường cho một iteration rollout
            [self.env.num_obs],# Shape của quan sát dành cho actor
            [self.env.num_privileged_obs], # Shape quan sát dành cho critic
            [self.env.num_actions], #Shape của hành động. Shape của hành động action 
        )
        
        # Log
        self.log_dir = log_dir # thư mục lưu trữ log
        self.writer = None # khởi tạo writer cho TensorBoard 
        self.tot_timesteps = 0 # khởi tạo biến đếm tổng số timestep đã thu thập từ tát cả môi trường trong toàn bộ quá trình học 
        self.tot_time = 0 #khởi tạo biến đếm tổng số thời gian đã trôi qua trong suốt quá trình học
        self.current_learning_iteration = 0 # khởi tạo biến đến số interation học hiện tại, bắt đầu từ 0, và sẽ tăng lên sau mỗi vòng lặp learn()
        # => nó giúp theo dõi tiến độ học. Khi lưu mô hình qua save(), iteration này được lưu cùng với state dict để có thể resume tranning từ điểm dừng.
        #   Nó cũng được dùng trong logging để hiển thị iteration hiện tại.

        _, _ = self.env.reset()

    # chịu trách nhiệm chạy vòng lặp học PPO trong nhiều iterations. Nó tích hợp rollout (thu thập từ môi trường), cập nhật mô hình, logging và lưu checkpoint
    # ==> num_learning_interations(int): Số lượng iteration học mà bạn muốn thực hiện. Mối iteration là một vòng lặp chính bao gồm
    #   + Thu thập dữ liệu từ rollout từ môi tường sử dung self.num_steps_per_env cho mỗi môi trường 
    #   + Tính toán return và advantage từ dữ liệu rollout
    #   + Cập nhật mô hình PPO bao gồm tối ưu hoá policy (actor) và value function (critic)
    #   Giá trị này được cộng vào self.current_learning để theo dõi tổng số iteration đã chạy. Ví dụ, nếu bạn gọi learn(100), nó sẽ chạy từ iteration hiện tại đến 99
    #   Điều này giúp resume training nếu cần 
    # ==> init_at_random_ep_len: hệ thống sẽ khởi tạo độ dài episode hiện tại cho tất cả môi trường. Điều này sẽ giúp cho việc bắt đầu trainning  từ một episode, thay vì từ đầu
    #       Hữu ích khi resume trainning lại một episode, thay vì từ đầu (độ dài = 0)
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        # Thiết lập summary writer từ tensorboard để ghi log các metric (như loss, reward) vào thư mục log dir 
        if self.log_dir is not None and self.writer is None:
            # wandb.init(
            #     project="XBot",
            #     sync_tensorboard=True,
            #     name=self.wandb_run_name,
            #     config=self.all_cfg,
            # )
            # Toạ một SummaryWriter để ghi log các metrics (như loss, reward, ) vào thư mục log_dir và ghi ra disk mỗi 10s để đảm bảo dữ liệu không bị mất
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # nếu thiết lập độ dại episode hiện tại của môi trường (episode_length_buf) thành một giá trị ngẫu nhiên, điều này làm cho việc bắt đàu trainning từ giữa một episode
        # thay vì từ đầu   
        if init_at_random_ep_len:
            # torch.randint_like(): tạo tensor ngẫu nhiên có cùng shape như episode_length_buf, độ dài từ 0 đến max_episode_length
            # Hữu ích khi resume training từ checkpoint, để tăng đa dạng hoá à tránh bắt đàu từ trạng thái giống nhau từ bất kì bước nào từ 0 đến 999
            # 
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        # lấy observation từ môi trường, đây là dữ liệu trạng thái hiện tại của môi trường( như vị trí, vận tốc của humanoid robot). Tensor này có shape num_envs, num_privileged_obs
        obs = self.env.get_observations()
        # lấy tensor quan sát dành cho critic, đây có thể là thông tin nội bộ đặc quyền không có sẵn
        privileged_obs = self.env.get_privileged_observations()
        # chọn quan sát cho critic. Nếu có privileged_obs, dùng nó nếu không dùng obs thông thường. Điều này đảm bảo critic có dữ liệu phù hợp để đánh giá giá trị
        critic_obs = privileged_obs if privileged_obs is not None else obs
        # Di chuyển tensors obs và critic_obs từ device của môi trường sang self.device (có thể là CPU hoặc GPU), được thiết lập trong _init_. 
        # Điều này đảm bảo dữ liệu tương thích với mô hình và tính toán trên GPU nếu cần, tăng tốc độ.
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        # Điều này bật các thành phần như dropout, batch normalization giúp mô hình học tốt hơn trong quá trình rollout và update. Ngược klaij trong inference, mô hình sẽ chuyển ang eval() để dropout
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"]
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                statistics.mean(locs["rewbuffer"]),
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                statistics.mean(locs["lenbuffer"]),
                self.tot_time,
            )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate
