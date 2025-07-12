# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist, squareform


# シミュレーションの複数回実行と平均化の設定

NUM_RUNS = 10  # シミュレーションの実行回数

# === シミュレーションパラメータの定義 ===
WIDTH = 50
HEIGHT = 50
AGENTS_NUMBER = 420
BASE_AGENT_SPEED = 1.0
SIMULATION_TIME = 200
ENTRY_PERIOD = 50

# --- 幸福度関連のパラメータ ---
INITIAL_HAPPINESS = 50.0
COLLISION_RADIUS = 1.5
HAPPINESS_DECREASE = 1.0
STRESS_DECREASE_FROM_AVOIDANCE = 0.2
HAPPINESS_DECREASE_PROBABILITY = 0.6
DENSITY_STRESS_RADIUS = 4.0
DENSITY_THRESHOLD = 3
DENSITY_STRESS_FACTOR = 0.3
SPEED_DIFFERENCE_STRESS_RADIUS = 5.0
SPEED_DIFFERENCE_THRESHOLD = 0.5
SPEED_DIFFERENCE_STRESS_FACTOR = 0.7
FAST_AGENT_STRESS_MULTIPLIER = 3.2
SLOW_AGENT_AORI_STRESS_FACTOR = 0.5

# --- ストレス耐性に関するパラメータ ---
VULNERABLE_AGENT_RATIO = 0.2
VULNERABLE_DECREASE_MULTIPLIER = 2.0
RESILIENT_DECREASE_MULTIPLIER = 0.5

# --- 回避行動のパラメータ ---
AVOIDANCE_RADIUS = 5.0
AVOIDANCE_FACTOR = 0.5

# --- 特殊エージェント用のパラメータ ---
ODD_AGENT_NUMBER = 5
ODD_AVOIDANCE_FACTOR = 2.0
ODD_AGENT_SPEED_MULTIPLIER = 0.8
ODD_WANDER_FACTOR = 0.3

# --- 速度の多様性と影響に関するパラメータ ---
FAST_AGENT_RATIO = 0.3
SLOW_AGENT_RATIO = 0.2
FAST_SPEED_MULTIPLIER = 1.5
SLOW_SPEED_MULTIPLIER = 0.6
INFLUENCE_RADIUS = 7.0
SYNC_FACTOR = 0.25
MAVERICK_AGENT_RATIO = 0.1

# --- 現在の速度で色分けするための閾値 ---
SPEED_COLOR_FAST_THRESHOLD = BASE_AGENT_SPEED * (1 + (FAST_SPEED_MULTIPLIER - 1) * 0.7)
SPEED_COLOR_SLOW_THRESHOLD = BASE_AGENT_SPEED * (1 - (1 - SLOW_SPEED_MULTIPLIER) * 0.7)


class Agent:
    def __init__(
        self,
        is_odd=False,
        speed_type="normal",
        resilience_type="resilient",
        is_maverick=False,
    ):
        self.is_odd = is_odd
        self.is_maverick = is_maverick if not is_odd else False
        self.speed_type = speed_type if not is_odd else "normal"
        self.resilience_type = resilience_type if not is_odd else "normal"
        self.happiness = INITIAL_HAPPINESS
        self.is_active = True
        if self.is_odd:
            self.base_speed = BASE_AGENT_SPEED * ODD_AGENT_SPEED_MULTIPLIER
        elif self.speed_type == "fast":
            self.base_speed = BASE_AGENT_SPEED * FAST_SPEED_MULTIPLIER
        elif self.speed_type == "slow":
            self.base_speed = BASE_AGENT_SPEED * SLOW_SPEED_MULTIPLIER
        else:
            self.base_speed = BASE_AGENT_SPEED
        self.current_speed = self.base_speed
        self.x, self.y, self.target_x, self.target_y = 0, 0, 0, 0
        self.set_new_target(is_initial=True)

    def set_new_target(self, is_initial=False):
        edges = {
            0: (np.random.uniform(0, WIDTH), 0),
            1: (np.random.uniform(0, WIDTH), HEIGHT),
            2: (0, np.random.uniform(0, HEIGHT)),
            3: (WIDTH, np.random.uniform(0, HEIGHT)),
        }
        if is_initial:
            start_edge_key = np.random.choice(list(edges.keys()))
            self.x, self.y = edges[start_edge_key]
        else:
            start_edge_key = np.argmin(
                [self.y, HEIGHT - self.y, self.x, WIDTH - self.x]
            )
        possible_target_keys = [key for key in edges.keys() if key != start_edge_key]
        self.target_x, self.target_y = edges[np.random.choice(possible_target_keys)]

    def move(self, all_agents):
        if not self.is_active:
            return False

        if self.is_odd:
            distance_to_target = np.sqrt(
                (self.target_x - self.x) ** 2 + (self.target_y - self.y) ** 2
            )
            if distance_to_target < self.current_speed:
                self.is_active = False
                return False
            dest_vec_x, dest_vec_y = self.target_x - self.x, self.target_y - self.y
            norm_dest = np.sqrt(dest_vec_x**2 + dest_vec_y**2)
            if norm_dest > 0:
                dest_vec_x, dest_vec_y = dest_vec_x / norm_dest, dest_vec_y / norm_dest
            random_angle = np.random.rand() * 2 * np.pi
            rand_vec_x, rand_vec_y = np.cos(random_angle), np.sin(random_angle)
            final_vec_x = (
                1 - ODD_WANDER_FACTOR
            ) * dest_vec_x + ODD_WANDER_FACTOR * rand_vec_x
            final_vec_y = (
                1 - ODD_WANDER_FACTOR
            ) * dest_vec_y + ODD_WANDER_FACTOR * rand_vec_y
            norm_final = np.sqrt(final_vec_x**2 + final_vec_y**2)
            if norm_final > 0:
                final_vec_x, final_vec_y = (
                    final_vec_x / norm_final,
                    final_vec_y / norm_final,
                )
            self.x += final_vec_x * self.current_speed
            self.y += final_vec_y * self.current_speed
            self.x, self.y = max(0, min(self.x, WIDTH)), max(0, min(self.y, HEIGHT))
            return False

        if self.is_maverick:
            self.current_speed = self.base_speed
        else:
            surrounding_speeds = []
            for other in all_agents:
                if self == other or other.is_odd or not other.is_active:
                    continue
                if (
                    np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
                    < INFLUENCE_RADIUS
                ):
                    surrounding_speeds.append(other.current_speed)
            if surrounding_speeds:
                self.current_speed = (
                    1 - SYNC_FACTOR
                ) * self.base_speed + SYNC_FACTOR * np.mean(surrounding_speeds)
            else:
                self.current_speed = self.base_speed

        if (
            np.sqrt((self.target_x - self.x) ** 2 + (self.y - self.y) ** 2)
            < self.current_speed
        ):
            self.is_active = False
            return False
        dest_vec_x, dest_vec_y = self.target_x - self.x, self.target_y - self.y
        norm_dest = np.sqrt(dest_vec_x**2 + dest_vec_y**2)
        if norm_dest > 0:
            dest_vec_x, dest_vec_y = dest_vec_x / norm_dest, dest_vec_y / norm_dest
        avoid_vec_x, avoid_vec_y = 0, 0
        for other in all_agents:
            if self == other or not other.is_active:
                continue
            dist_x, dist_y = self.x - other.x, self.y - other.y
            distance = np.sqrt(dist_x**2 + dist_y**2)
            if 0 < distance < AVOIDANCE_RADIUS:
                repulsion = 1 / distance
                current_avoid_factor = AVOIDANCE_FACTOR
                if other.is_odd:
                    current_avoid_factor = ODD_AVOIDANCE_FACTOR
                    if np.random.rand() < HAPPINESS_DECREASE_PROBABILITY:
                        multiplier = (
                            VULNERABLE_DECREASE_MULTIPLIER
                            if self.resilience_type == "vulnerable"
                            else RESILIENT_DECREASE_MULTIPLIER
                        )
                        self.happiness = max(
                            0,
                            self.happiness
                            - STRESS_DECREASE_FROM_AVOIDANCE * multiplier,
                        )
                avoid_vec_x += (dist_x * repulsion) * current_avoid_factor
                avoid_vec_y += (dist_y * repulsion) * current_avoid_factor
        final_vec_x, final_vec_y = dest_vec_x + avoid_vec_x, dest_vec_y + avoid_vec_y
        norm_final = np.sqrt(final_vec_x**2 + final_vec_y**2)
        if norm_final > 0:
            final_vec_x, final_vec_y = (
                final_vec_x / norm_final,
                final_vec_y / norm_final,
            )
        self.x += final_vec_x * self.current_speed
        self.y += final_vec_y * self.current_speed
        self.x, self.y = max(0, min(self.x, WIDTH)), max(0, min(self.y, HEIGHT))
        return False


def initialize_simulation_blueprints():
    blueprints = []
    for _ in range(ODD_AGENT_NUMBER):
        blueprints.append({"is_odd": True})

    num_normal_agents_total = AGENTS_NUMBER - ODD_AGENT_NUMBER

    num_vulnerable = int(num_normal_agents_total * VULNERABLE_AGENT_RATIO)
    num_resilient = num_normal_agents_total - num_vulnerable

    num_fast = int(num_normal_agents_total * FAST_AGENT_RATIO)
    num_slow = int(num_normal_agents_total * SLOW_AGENT_RATIO)
    num_normal_speed = num_normal_agents_total - num_fast - num_slow

    num_maverick = int(num_normal_agents_total * MAVERICK_AGENT_RATIO)
    num_follower = num_normal_agents_total - num_maverick

    resilience_types = ["vulnerable"] * num_vulnerable + ["resilient"] * num_resilient
    speed_types = (
        ["fast"] * num_fast + ["slow"] * num_slow + ["normal"] * num_normal_speed
    )
    maverick_flags = [True] * num_maverick + [False] * num_follower

    # 各リストの要素数がnum_normal_agents_totalと一致するように調整
    if len(resilience_types) > num_normal_agents_total: resilience_types = resilience_types[:num_normal_agents_total]
    if len(speed_types) > num_normal_agents_total: speed_types = speed_types[:num_normal_agents_total]
    if len(maverick_flags) > num_normal_agents_total: maverick_flags = maverick_flags[:num_normal_agents_total]

    np.random.shuffle(resilience_types)
    np.random.shuffle(speed_types)
    np.random.shuffle(maverick_flags)

    for i in range(num_normal_agents_total):
        blueprints.append(
            {
                "is_odd": False,
                "speed_type": speed_types[i],
                "resilience_type": resilience_types[i],
                "is_maverick": maverick_flags[i],
            }
        )

    np.random.shuffle(blueprints)
    return blueprints


def check_collisions(agents):
    active_agents = [agent for agent in agents if agent.is_active]
    for i in range(len(active_agents)):
        for j in range(i + 1, len(active_agents)):
            agent1, agent2 = active_agents[i], active_agents[j]
            distance = np.sqrt((agent1.x - agent2.x) ** 2 + (agent1.y - agent2.y) ** 2)
            if distance < COLLISION_RADIUS:
                if np.random.rand() < HAPPINESS_DECREASE_PROBABILITY:
                    if agent1.resilience_type == "vulnerable":
                        multiplier1 = VULNERABLE_DECREASE_MULTIPLIER
                    elif agent1.resilience_type == "resilient":
                        multiplier1 = RESILIENT_DECREASE_MULTIPLIER
                    else:
                        multiplier1 = 1.0
                    if agent2.resilience_type == "vulnerable":
                        multiplier2 = VULNERABLE_DECREASE_MULTIPLIER
                    elif agent2.resilience_type == "resilient":
                        multiplier2 = RESILIENT_DECREASE_MULTIPLIER
                    else:
                        multiplier2 = 1.0
                    agent1.happiness = max(
                        0, agent1.happiness - HAPPINESS_DECREASE * multiplier1
                    )
                    agent2.happiness = max(
                        0, agent2.happiness - HAPPINESS_DECREASE * multiplier2
                    )


def check_density_stress(agents):
    active_agents = [agent for agent in agents if agent.is_active and not agent.is_odd]
    if not active_agents:
        return
    positions = np.array([[agent.x, agent.y] for agent in active_agents])
    dist_matrix = squareform(pdist(positions))
    for i, agent in enumerate(active_agents):
        neighbor_indices = np.where(dist_matrix[i] < DENSITY_STRESS_RADIUS)[0]
        neighbor_count = len(neighbor_indices) - 1
        if neighbor_count > DENSITY_THRESHOLD:
            multiplier = (
                VULNERABLE_DECREASE_MULTIPLIER
                if agent.resilience_type == "vulnerable"
                else RESILIENT_DECREASE_MULTIPLIER
            )
            stress = DENSITY_STRESS_FACTOR * (neighbor_count - DENSITY_THRESHOLD)
            agent.happiness = max(0, agent.happiness - stress * multiplier)


def check_speed_stress(agents):
    active_agents = [agent for agent in agents if agent.is_active and not agent.is_odd]
    for i in range(len(active_agents)):
        for j in range(i + 1, len(active_agents)):
            agent1, agent2 = active_agents[i], active_agents[j]
            distance = np.sqrt((agent1.x - agent2.x) ** 2 + (agent1.y - agent2.y) ** 2)
            if distance < SPEED_DIFFERENCE_STRESS_RADIUS:
                speed_diff = agent1.current_speed - agent2.current_speed

                if speed_diff > SPEED_DIFFERENCE_THRESHOLD:
                    stress_multiplier_fast = 1.0
                    if agent1.current_speed > SPEED_COLOR_FAST_THRESHOLD:
                        stress_multiplier_fast = FAST_AGENT_STRESS_MULTIPLIER
                    resilience_fast = (
                        VULNERABLE_DECREASE_MULTIPLIER
                        if agent1.resilience_type == "vulnerable"
                        else RESILIENT_DECREASE_MULTIPLIER
                    )
                    stress_fast = (
                        SPEED_DIFFERENCE_STRESS_FACTOR
                        * speed_diff
                        * stress_multiplier_fast
                    )
                    agent1.happiness = max(
                        0, agent1.happiness - stress_fast * resilience_fast
                    )

                    resilience_slow = (
                        VULNERABLE_DECREASE_MULTIPLIER
                        if agent2.resilience_type == "vulnerable"
                        else RESILIENT_DECREASE_MULTIPLIER
                    )
                    stress_slow = (
                        SPEED_DIFFERENCE_STRESS_FACTOR
                        * speed_diff
                        * SLOW_AGENT_AORI_STRESS_FACTOR
                    )
                    agent2.happiness = max(
                        0, agent2.happiness - stress_slow * resilience_slow
                    )

                elif -speed_diff > SPEED_DIFFERENCE_THRESHOLD:
                    stress_multiplier_fast = 1.0
                    if agent2.current_speed > SPEED_COLOR_FAST_THRESHOLD:
                        stress_multiplier_fast = FAST_AGENT_STRESS_MULTIPLIER
                    resilience_fast = (
                        VULNERABLE_DECREASE_MULTIPLIER
                        if agent2.resilience_type == "vulnerable"
                        else RESILIENT_DECREASE_MULTIPLIER
                    )
                    stress_fast = (
                        SPEED_DIFFERENCE_STRESS_FACTOR
                        * (-speed_diff)
                        * stress_multiplier_fast
                    )
                    agent2.happiness = max(
                        0, agent2.happiness - stress_fast * resilience_fast
                    )

                    resilience_slow = (
                        VULNERABLE_DECREASE_MULTIPLIER
                        if agent1.resilience_type == "vulnerable"
                        else RESILIENT_DECREASE_MULTIPLIER
                    )
                    stress_slow = (
                        SPEED_DIFFERENCE_STRESS_FACTOR
                        * (-speed_diff)
                        * SLOW_AGENT_AORI_STRESS_FACTOR
                    )
                    agent1.happiness = max(
                        0, agent1.happiness - stress_slow * resilience_slow
                    )


def run_simulation():
    history = []
    agent_blueprints = initialize_simulation_blueprints()
    agents = []

    for frame in range(SIMULATION_TIME):
        if frame < ENTRY_PERIOD:
            target_agent_count = int(AGENTS_NUMBER * (frame + 1) / ENTRY_PERIOD)
            num_to_add = target_agent_count - len(agents)
            for _ in range(num_to_add):
                if agent_blueprints:
                    bp = agent_blueprints.pop(0)
                    agents.append(Agent(**bp))

        for agent in agents:
            agent.move(agents)

        check_collisions(agents)
        check_density_stress(agents)
        check_speed_stress(agents)

        current_state = [
            (
                agent.x,
                agent.y,
                agent.happiness,
                agent.is_odd,
                agent.is_active,
                agent.speed_type,
                agent.current_speed,
                agent.is_maverick,
            )
            for agent in agents
        ]

        # 【幸福度計算】生まれつきのspeed_typeに基づいて集計
        all_normal_agents = [agent for agent in agents if not agent.is_odd]
        fast_h = [a.happiness for a in all_normal_agents if a.speed_type == "fast"]
        slow_h = [a.happiness for a in all_normal_agents if a.speed_type == "slow"]
        normal_h = [a.happiness for a in all_normal_agents if a.speed_type == "normal"]

        happiness_by_type = {
            "fast": np.mean(fast_h) if fast_h else np.nan,
            "slow": np.mean(slow_h) if slow_h else np.nan,
            "normal": np.mean(normal_h) if normal_h else np.nan,
        }

        # 【人数計算】現在の速度に基づいて集計
        active_normal_agents = [
            agent for agent in agents if agent.is_active and not agent.is_odd
        ]
        count_fast = 0
        count_slow = 0
        count_normal = 0
        for agent in active_normal_agents:
            if agent.current_speed > SPEED_COLOR_FAST_THRESHOLD:
                count_fast += 1
            elif agent.current_speed < SPEED_COLOR_SLOW_THRESHOLD:
                count_slow += 1
            else:
                count_normal += 1

        speed_counts = {"fast": count_fast, "slow": count_slow, "normal": count_normal}

        history.append((current_state, happiness_by_type, speed_counts))

        active_agents_exist = any(s[4] for s in current_state)
        if frame > ENTRY_PERIOD and not active_agents_exist:
            # print(f"All agents have become inactive at frame {frame}.")
            remaining_frames = SIMULATION_TIME - (frame + 1)
            last_h = history[-1][1]
            empty_counts = {"fast": 0, "slow": 0, "normal": 0}
            for _ in range(remaining_frames):
                history.append(([], last_h, empty_counts))
            break

    return history


#ここからがメインの実行ブロック 


def run_multiple_simulations(num_runs):
    """指定された回数シミュレーションを実行し、結果を集計・平均化する"""
    all_runs_histories = []
    
    # 指定回数シミュレーションを実行し、全履歴を保存
    for i in range(num_runs):
        print(f"シミュレーション実行中... ({i + 1}/{num_runs})")
        # 乱数のシードを毎回変えるために、初期化はしない
        history = run_simulation()
        all_runs_histories.append(history)

    print("\n全シミュレーション完了。結果を集計中...")

    # --- グラフデータの平均化 ---
    # 平均値を格納するためのリストを初期化
    avg_fast_h, avg_slow_h, avg_normal_h = [], [], []
    avg_fast_c, avg_slow_c, avg_normal_c = [], [], []

    # 各フレームごとに平均値を計算
    for frame in range(SIMULATION_TIME):
        # そのフレームの全実行結果から値を取得
        frame_fast_h = [run[frame][1].get("fast", np.nan) for run in all_runs_histories]
        frame_slow_h = [run[frame][1].get("slow", np.nan) for run in all_runs_histories]
        frame_normal_h = [run[frame][1].get("normal", np.nan) for run in all_runs_histories]
        
        frame_fast_c = [run[frame][2].get("fast", 0) for run in all_runs_histories]
        frame_slow_c = [run[frame][2].get("slow", 0) for run in all_runs_histories]
        frame_normal_c = [run[frame][2].get("normal", 0) for run in all_runs_histories]

        # nanを無視して平均を計算し、リストに追加
        avg_fast_h.append(np.nanmean(frame_fast_h))
        avg_slow_h.append(np.nanmean(frame_slow_h))
        avg_normal_h.append(np.nanmean(frame_normal_h))
        
        avg_fast_c.append(np.mean(frame_fast_c))
        avg_slow_c.append(np.mean(frame_slow_c))
        avg_normal_c.append(np.mean(frame_normal_c))

    # --- 最終幸福度の平均化 ---
    final_fast_h_runs = [run[-1][1].get("fast", np.nan) for run in all_runs_histories]
    final_slow_h_runs = [run[-1][1].get("slow", np.nan) for run in all_runs_histories]
    final_normal_h_runs = [run[-1][1].get("normal", np.nan) for run in all_runs_histories]
    
    avg_final_happiness = {
        "fast": np.nanmean(final_fast_h_runs),
        "slow": np.nanmean(final_slow_h_runs),
        "normal": np.nanmean(final_normal_h_runs)
    }

    # アニメーション用には最後の実行履歴を、グラフ用には平均データを返す
    return {
        "last_run_history": all_runs_histories[-1],
        "avg_happiness": { "fast": avg_fast_h, "slow": avg_slow_h, "normal": avg_normal_h },
        "avg_counts": { "fast": avg_fast_c, "slow": avg_slow_c, "normal": avg_normal_c },
        "avg_final_happiness": avg_final_happiness
    }


if __name__ == '__main__':
    # 複数シミュレーションを実行し、結果を取得
    results = run_multiple_simulations(NUM_RUNS)

    # 結果をそれぞれの変数に展開
    history_for_anim = results["last_run_history"]
    avg_h_data = results["avg_happiness"]
    avg_c_data = results["avg_counts"]
    avg_final_h = results["avg_final_happiness"]


    # --- 描画設定 ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.tight_layout(pad=5.0)

    # 1. エージェントの動き（最後の実行結果を表示）
    ax1.set_xlim(0, WIDTH); ax1.set_ylim(0, HEIGHT); ax1.set_aspect("equal")
    ax1.set_title("Agent Simulation (Last Run)"); ax1.set_xlabel("X-axis"); ax1.set_ylabel("Y-axis")
    scatter_fast = ax1.scatter([], [], s=25, color="crimson", label="Currently Fast")
    scatter_slow = ax1.scatter([], [], s=25, color="royalblue", label="Currently Slow")
    scatter_normal_speed = ax1.scatter([], [], s=25, color="forestgreen", label="Currently Normal")
    scatter_odd = ax1.scatter([], [], s=50, facecolor="orange", edgecolor="black", linewidths=1.0, label="Odd")
    ax1.legend(loc="upper right")

    # 2. 幸福度の平均値グラフ
    ax2.set_xlim(0, SIMULATION_TIME); ax2.set_ylim(-5, INITIAL_HAPPINESS * 1.05)
    ax2.set_title(f"Average Happiness by Base Type ({NUM_RUNS} runs)"); ax2.set_xlabel("Time (frames)"); ax2.set_ylabel("Average Happiness"); ax2.grid(True)
    line_h_fast, = ax2.plot(avg_h_data["fast"], color="crimson", lw=2, label="Fast Type")
    line_h_slow, = ax2.plot(avg_h_data["slow"], color="royalblue", lw=2, label="Slow Type")
    line_h_normal, = ax2.plot(avg_h_data["normal"], color="forestgreen", lw=2, label="Normal Type")
    ax2.legend(loc="upper right")

    # 3. エージェント数の平均値グラフ
    num_normal_agents_total = AGENTS_NUMBER - ODD_AGENT_NUMBER
    ax3.set_xlim(0, SIMULATION_TIME); ax3.set_ylim(0, num_normal_agents_total * 0.8)
    ax3.set_title(f"Average Agent Count by Current Speed ({NUM_RUNS} runs)"); ax3.set_xlabel("Time (frames)"); ax3.set_ylabel("Average Count"); ax3.grid(True)
    line_c_fast, = ax3.plot(avg_c_data["fast"], color="crimson", lw=2, label="Currently Fast")
    line_c_slow, = ax3.plot(avg_c_data["slow"], color="royalblue", lw=2, label="Currently Slow")
    line_c_normal, = ax3.plot(avg_c_data["normal"], color="forestgreen", lw=2, label="Currently Normal")
    ax3.legend()
    
    # 4. 空のプロット
    ax4.axis("off")


    # --- アニメーション更新関数 ---
    # アニメーションは最後の実行結果のみを描画
    def animate(frame):
        if frame >= len(history_for_anim): return ()

        current_state, _, _ = history_for_anim[frame]
        active_states = [s for s in current_state if s[4]]
        all_normal_states = [s for s in active_states if not s[3]]
        
        fast_now_states, slow_now_states, normal_now_states = [], [], []
        for s in all_normal_states:
            current_speed = s[6]
            if current_speed > SPEED_COLOR_FAST_THRESHOLD: fast_now_states.append(s)
            elif current_speed < SPEED_COLOR_SLOW_THRESHOLD: slow_now_states.append(s)
            else: normal_now_states.append(s)

        odd_states = [s for s in active_states if s[3]]

        scatter_fast.set_offsets(np.array([[s[0], s[1]] for s in fast_now_states]) if fast_now_states else np.empty((0, 2)))
        scatter_slow.set_offsets(np.array([[s[0], s[1]] for s in slow_now_states]) if slow_now_states else np.empty((0, 2)))
        scatter_normal_speed.set_offsets(np.array([[s[0], s[1]] for s in normal_now_states]) if normal_now_states else np.empty((0, 2)))
        scatter_odd.set_offsets(np.array([[s[0], s[1]] for s in odd_states]) if odd_states else np.empty((0, 2)))
        
        ax1.set_title(f"Agent Simulation (Last Run, Frame: {frame})")
        
        # グラフはすでにプロット済みなので、アニメーションでの更新は不要
        # 戻り値に含めないとエラーになることがある
        return (scatter_fast, scatter_slow, scatter_normal_speed, scatter_odd, 
                line_h_fast, line_h_slow, line_h_normal,
                line_c_fast, line_c_slow, line_c_normal)

    # --- アニメーション生成と結果表示 ---
    ani = animation.FuncAnimation(fig, animate, frames=SIMULATION_TIME, blit=True, repeat=False)
    plt.show()

    # --- 最終的な幸福度の平均値を表示 ---
    print("\n" + "=" * 50)
    print(f"--- Final Average Happiness by Base Type ({NUM_RUNS} runs) ---")
    print(f"  - Fast Type Agents:   {avg_final_h.get('fast', float('nan')):.2f}")
    print(f"  - Normal Type Agents: {avg_final_h.get('normal', float('nan')):.2f}")
    print(f"  - Slow Type Agents:   {avg_final_h.get('slow', float('nan')):.2f}")
    print("=" * 50)