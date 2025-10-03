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

import torch
import torch.nn as nn
from torch.distributions import Normal

# test kitare
class ActorCritic(nn.Module):
    def __init__(self,  num_actor_obs, # cung cấp chiều của observation cho actor (policy) :::int
                        num_critic_obs, # Số chiều của critic (value function) :::int
                        num_actions, # số chiều của action space :::int 
                        actor_hidden_dims=[256, 256, 256], # List kích thước của các hidden layers của actor :::List[int]
                        critic_hidden_dims=[256, 256, 256], #List Kích thước các hidden layer của critic :::List[int]
                        init_noise_std=1.0, # độ lệch chuẩn khởi tạo cho action noise :::float
                        activation = nn.ELU(), # hàm kích hoạt ELU :::nn.Module 
                        **kwargs):
        if kwargs: # kiểm tra các thông số truyền không đúng
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__() 


        mlp_input_dim_a = num_actor_obs 
        mlp_input_dim_c = num_critic_obs
        # Policy: xây dựng não cho robot
        actor_layers = [] # khởi tạo danh sách rỗng
        # nhận thông tin từ môi trường
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0])) #mlp_input_dim_a có 256 chiều
        actor_layers.append(activation)
        #Chèn cái ELU vào sau mỗi phần tử 
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        # mục đích nhận thông tin robot nhìn thấy gì, xử lý qua 3 tầng ẩn: suy nghĩ phức tạp qua các phép biến đổi, đưa ra quyết định, mỗi khớp nên xoay theo bao nhiêu độ

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        # mục đích xây dụng bộ não đánh giá để ước lượng giá trị của tình huống hiện tại
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        # Tạo sự ngẫu nhiên để robot có thể học được
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions)) # tạo standard deviation độ lệch chuẩn
        # torch.one => tạo vector toàn số 1 với kích thước bằng số lượng num_action
        # init_noise_std * torch.ones(num_actions) tạo ra ma trận với cơ số init_noise_std [1 1] * 2 => [2 2]

        # như một hộp xổ số để rút action ngẫu nhiên sẽ có giá trị khi sử dụng hàm update_distribution
        self.distribution = None

        # disable args validation for speedup 
        # tắt nhân viên kiểm tra chất lượng
        Normal.set_default_validate_args = False
        
    # giống phương thức của java không cần khởi tạo obj chỉ cần gọi từ class 
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):# sequence: mạng neural network cần khởi tạo. scales : danh sách hệ số gain cho từng layer
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
    # mod for mod in sequential if isinstance(mod, nn.Linear)) duyệt qua tất cả modules trong sequential chỉ lấy nhưng layer kiểu nn.Linear 
    # enumerate(mod for mod in sequential if isinstance(mod, nn.Linear)) tạo ra cặp (index module)
    # orthogonal initialization là ma trận trực giao W * W^T = I (ma trận đơn vị)
    # Giữ độ dài vector khi nhân ma trận


    # được gọi khi env rết tức là khi episode kết thúc và bắt đầu episode mới => chỉ là placeholder method 
    def reset(self, dones=None):
        pass

    # không implement từ class gốc
    def forward(self):
        raise NotImplementedError

    # truy cập giá trị mean của phân phối giá trị trung bình kỳ vọng mà policy kỳ vọng mà policy muốn thực hiện
    @property
    def action_mean(self):
        return self.distribution.mean

    # đại diện cho mức độ khám phá độ lệch chuẩn của phân phối action
    @property
    def action_std(self):
        return self.distribution.stddev
    
    #Tính entropy của phân phói action và sum theo chiều cuối cùng
    # std cao thì policy sẽ khám phá nhiều, action ngẫu nhiên hơn, std thấp policy tự tin hơn, action ổn định hơn 
    # thường được dùng trong loss function để khuyến kích exploration
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    #cập nhật phân phối xác xuất cho việc lấy mẫu action trong PPO 
    def update_distribution(self, observations):
        #actor nhận state đầu vào trả về giá trị trung bình mean của action cho mỗi dimention
        mean = self.actor(observations)
        # tạo phân phối chuẩn với mean, std
        self.distribution = Normal(mean, mean*0. + self.std)

    # lấy mẫu action từ phân phối
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    # tính log xác suất của action đã thực hiện
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # trả về chỉ mean từ actor network, không thêm noise
    # không cần distribution, nó không gọi update_distribution(), không sample
    # dùng cho deploy sau khi đã train xong
    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    # critic network đánh giá giá trị (value) của state hiện tại
    # trả về ước lượng tổng reward tích luỹ từ state này đến cuối episode
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value