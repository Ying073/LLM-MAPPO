"""
search_env.py —— M1 里程碑：多无人机动态目标搜索仿真环境（骨架版）

这是复现 LLM-MAPPO 的第一步。你的任务：把每个标了 ``TODO(你)`` 的函数体
补全，让 ``python search_env.py`` 跑出一个演示——UAV 在动、目标在动、
障碍物挡住路、区域不确定度随时间下降。

在动手前，先读懂下面这套「数据约定」，后面 MAPPO 的观测/动作都依赖它。

数据约定
--------
- 网格索引 ``(ix, iy)``，``ix, iy ∈ [0, LX)``，LX = LY = 20。
  每格 100 m（搜索区 2000 m / 20）。
- 高度只有 3 档 ``z ∈ {50, 100, 150}``，用下标 ``h ∈ {0, 1, 2}`` 表示。
- 动作空间（离散 6 个）：``0=北 1=东 2=南 3=西 4=升 5=降``。
  前 4 个改水平位置，后 2 个改高度（±一档）。
- 目标存在状态 ``zeta[iy, ix] ∈ {0, 1}``：该网格此刻是否有目标。
- 障碍占用 ``occ[iy, ix] ∈ {0, 1}``：该网格是否被障碍物占据（UAV 不能进入）。

参考公式（原文 + 中文说明见 LLM-MAPPO_Markdown_Reader/equations.md）
--------------------------------------------------------------------
- 公式 4：感知域 Φ_n(t)，半径 S_n = z·tan(θ/2)/L；表 I 给出结果 1/5/9 个网格
- 公式 5：检测概率 P^D、虚警概率 P^F（随高度）
- 公式 6：LTPM 的贝叶斯更新（检测到 / 未检测到 / 不在感知域 三种情形）
- 公式 7：LEUM = 目标概率的信息熵（概率 0.5 时最大，0 或 1 时最小）
- 公式 8：全局不确定度 = 各 UAV 局部不确定度的最小值
- 公式 10：全局目标概率 = 不确定度最低的 UAV 的估计（并列取最大 p）
- 公式 9：区域平均不确定度 = 所有网格全局不确定度的均值
"""

import numpy as np

# ---------------------------------------------------------------
# 论文表 I 的参数（直接照抄，不用改）
# ---------------------------------------------------------------
LX = 20                      # 网格列数
LY = 20                      # 网格行数
AREA = 2000.0                # 搜索区边长 2000 m
GRID = AREA / LX             # 每格 100 m

N_UAV = 7                    # 无人机数量
N_OBSTACLE = 20              # 静态障碍物数量
N_TARGET = 15                # 动态目标数量
TARGET_SPEED = 1.0           # 目标速度 1 m/s
UAV_SPEED = 10.0             # UAV 速度 10 m/s
MAX_STEPS = 500              # 每回合最大时间步

HEIGHTS = np.array([50.0, 100.0, 150.0])            # 三档高度
DET_PROB = np.array([0.9, 0.8, 0.7])                # 检测概率（随高度档位）
FALSE_PROB = np.array([0.1, 0.2, 0.3])              # 虚警概率
SENSE_SIZE = np.array([1, 5, 9])                    # 感知域大小（网格数）1/5/9

# 水平移动的方向向量，下标对应动作 0~3
DX = np.array([0, 1, 0, -1])   # 北(0) 东(1) 南(2) 西(3)：列方向变化
DY = np.array([-1, 0, 1, 0])   # 注意屏幕坐标 y 向下，所以「北」是 iy-1


class SearchEnv:
    """多 UAV 动态目标搜索环境。

    内部状态（均用 numpy 数组）：
        - ``uav_pos``  : (N_UAV, 3) 每架 UAV 的 (ix, iy, h)，h 是高度档位下标
        - ``zeta``     : (LY, LX)   int，目标存在状态 0/1
        - ``occ``      : (LY, LX)   int，障碍占用 0/1
        - ``obs_h``    : (LY, LX)   int，障碍物高度档位（仅 occ=1 处有意义）
        - ``ltpm``     : (N_UAV, LY, LX) 每架 UAV 的局部目标概率 p ∈ [0,1]
        - ``leum``     : (N_UAV, LY, LX) 每架 UAV 的局部不确定度 χ ∈ [0,1]
        - ``gtpm``     : (LY, LX)  全局目标概率
        - ``geum``     : (LY, LX)  全局不确定度
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    # -------------------------------------------------------------
    # 一、初始化
    # -------------------------------------------------------------
    def reset(self):
        """重置一局：随机摆放障碍物、目标、UAV，初始化感知地图。

        返回
        ----
        obs : 一个你自定义的观测结构（先用最简版本：返回 self._get_obs()）
        """
        # TODO(你):
        # 1. 随机选 20 个不重复网格放障碍物（含随机高度档位 h）
        # 2. 随机选 15 个不重复、且非障碍的网格放目标（zeta=1）
        # 3. 随机放 7 架 UAV 到非障碍网格，初始高度 h=1（100 m）
        # 4. 初始化 ltpm 全为 0.5，leum 由公式 7 的熵给出
        # 5. 调用 self._fuse_maps() 得到 gtpm / geum
        # 6. 记录 self.t = 0，返回 self._get_obs()
        raise NotImplementedError

    # -------------------------------------------------------------
    # 二、感知模型（公式 4、5）
    # -------------------------------------------------------------
    def _sensing_domain(self, n: int):
        """返回 UAV n 感知域内所有网格的 (ix, iy) 列表。

        提示：SENSE_SIZE[h] 给出感知域包含的网格数（1 / 5 / 9）。
        1 格 = 自身；5 格 = 自身 + 上下左右；9 格 = 3×3。以 UAV 所在网格
        为中心，超出边界的网格丢弃。这个解释可对照公式 4 验证。
        """
        # TODO(你)
        raise NotImplementedError

    def _detect(self, n: int, ix: int, iy: int):
        """按公式 5 返回 UAV n 对网格 (ix, iy) 的检测结果 D ∈ {0, 1}。

        规则：若该网格真的有目标（zeta=1），以概率 DET_PROB[h] 报 D=1；
        否则以概率 FALSE_PROB[h] 报 D=1（虚警）。用 self.rng 抽样。
        """
        # TODO(你)
        raise NotImplementedError

    # -------------------------------------------------------------
    # 三、感知地图更新与融合（公式 6–10）
    # -------------------------------------------------------------
    def _update_maps(self):
        """每步调用：先各 UAV 局部更新，再融合出全局地图。

        局部更新（对每架 UAV n、感知域内每个网格）：
          - 得到检测结果 D = self._detect(n, ix, iy)
          - 按公式 6 做贝叶斯更新 p：
              D=1:  p ← p·P^D / (p·P^D + (1-p)·P^F)
              D=0:  p ← p·(1-P^D) / (p·(1-P^D) + (1-p)·(1-P^F))
            不在感知域内的网格 p 不变。
          - 按公式 7 更新 leum：χ = -p·log2(p) - (1-p)·log2(1-p)
            （p=0 或 1 时取 0，避免 log 报错）
        然后调用 self._fuse_maps()。
        """
        # TODO(你)
        raise NotImplementedError

    def _fuse_maps(self):
        """融合出全局地图 gtpm / geum（公式 8、10）。

        - geum[iy, ix] = 所有 UAV 在该网格 leum 的最小值
        - gtpm[iy, ix] = 取「leum 最小」的那架 UAV 的 ltpm；
          若多架并列最小，取其中最大的 ltpm（提高该网格搜索优先级）。
        """
        # TODO(你)
        raise NotImplementedError

    def area_uncertainty(self) -> float:
        """返回区域平均不确定度（公式 9）：geum 的均值。"""
        # TODO(你)
        raise NotImplementedError

    # -------------------------------------------------------------
    # 四、仿真推进
    # -------------------------------------------------------------
    def step(self, actions):
        """执行一个时间步。

        参数
        ----
        actions : list[int]，长度 N_UAV，每架 UAV 的动作 0~5

        返回
        ----
        (obs, reward, done, info)
        reward 在 M1 阶段先随便给（比如返回 -area_uncertainty），
        真正的手写稠密奖励留到 M2。
        """
        # TODO(你)：
        # 1. 逐架执行动作：水平动作改 (ix, iy)，升/降改 h；
        #    越界或撞障碍则保持原位（简单版动作掩码）
        # 2. 移动目标：每个目标以 1 m/s 随机方向走（注意换算成网格；
        #    简单起见可让目标每步以一定概率向随机相邻格移动，或原地）
        # 3. self._update_maps() 更新感知
        # 4. self.t += 1，判断 done（t 达到 MAX_STEPS）
        # 5. 返回 (self._get_obs(), reward, done, info)
        raise NotImplementedError

    # -------------------------------------------------------------
    # 五、观测与可视化
    # -------------------------------------------------------------
    def _get_obs(self):
        """返回一个观测结构。M1 先用最简版：一个 dict，包含
        uav_pos / gtpm / geum / occ 等字段，供 render 使用即可。
        （MAPPO 真正用的局部观测 O_n(t) 到 M2 再精确定义。）
        """
        # TODO(你)
        raise NotImplementedError

    def render(self, ax=None):
        """用 matplotlib 画俯视图：网格、障碍（灰）、目标（红）、
        UAV（蓝）、背景为不确定度热图。至少能看清位置关系。
        """
        # TODO(你)
        raise NotImplementedError


# ---------------------------------------------------------------
# 演示脚本：补全上面的 TODO 后，运行 python search_env.py 应能出图
# ---------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = SearchEnv(seed=0)
    obs = env.reset()
    for t in range(MAX_STEPS):
        actions = [int(env.rng.integers(0, 6)) for _ in range(N_UAV)]
        obs, reward, done, info = env.step(actions)
        if done:
            break

    fig, ax = plt.subplots(figsize=(6, 6))
    env.render(ax=ax)
    ax.set_title(f"t={env.t}, area uncertainty={env.area_uncertainty():.3f}")
    plt.show()
