# GMR 核心方法详解：General Motion Retargeting

文献：Araújo et al., *Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking*, arXiv:2510.02252 (2025)；阅读位置：Section IV, pp. 4–5（PDF）。

> **一句话总结：**GMR 是基于运动学优化的人体到人形机器人动作重定向方法，创新重点在**非均匀局部缩放**和**两阶段逆运动学优化**，以提高动作保真度并减少不合理参考轨迹。它不是 RL 控制器。

## 1. GMR 在完整机器人学习系统中的位置

![图 A：人体运动 → GMR 动作重定向 → 机器人参考运动 → RL 跟踪（讲解示意图）](GMR_Zotero_Images/gmr_learning_workflow.png)

图 A：人体运动 → GMR 动作重定向 → 机器人参考运动 → RL 跟踪（讲解示意图）

人体运动数据（BVH/SMPL/MoCap）与机器人存在关节数量、连杆比例和可动范围差异（embodiment gap）。直接复制人体关节角可能出现悬空、脚底滑动、穿地、末端漂移。GMR 输出机器人浮动基座位姿与关节轨迹；随后由 BeyondMimic 等 tracking policy 学习执行。

## 2. 论文整体流程与五个步骤

![图 B：论文 Fig. 2，GMR 的五步流程（原文 p.4）](GMR_Zotero_Images/paper_fig2_gmr_pipeline.png)

图 B：论文 Fig. 2，GMR 的五步流程（原文 p.4）

### Step 1｜Human–Robot Key Body Matching

从人体骨架和机器人 MJCF/URDF 提取主要刚体，构建映射 M={(i,j)}，如 pelvis↔root、torso↔torso、left hand↔left hand、feet↔feet。GMR 匹配的是关键刚体在任务空间中的位置和姿态，而不是简单复制每个关节角。

### Step 2｜Rest Pose Cartesian Alignment

人体与机器人即使处于同样的站姿，各刚体局部坐标系定义也可能不同。因此在静止姿态下调整人体刚体方向，必要时加入局部位置偏移，修正方向基准差异，减少脚尖内扣等姿态伪影。

### Step 3｜Human Data Non-Uniform Local Scaling（关键创新）

单一全局缩放假设“人体与机器人所有身体部位按同一比例变化”，而真实人形机器人往往有不同的腿长/臂长比例。GMR 为不同关键部位设置局部缩放系数，同时用单独系数处理根节点平移。

![图 C：非均匀局部缩放的概念示意图（本笔记重绘）](GMR_Zotero_Images/gmr_local_scaling_concept.png)

图 C：非均匀局部缩放的概念示意图（本笔记重绘）

论文式 (2)：关键部位目标位置 · LaTeX 源码（可复制）

$$
\mathbf p_b^{target}=\frac{h}{h_{ref}}s_b(\mathbf p_b^{source}-\mathbf p_{root}^{source})+\frac{h}{h_{ref}}s_{root}\mathbf p_{root}^{source}
$$

**拆解：**第一项将关键部位相对根节点的位置按 s\_b 缩放；第二项将根节点的全局位移按 s\_root 缩放；h/h\_ref 是源人体高度相对于基准高度的系数。对根节点 b=root，公式退化为下面的统一全局平移缩放：

论文式 (3)：根节点平移 · LaTeX 源码（可复制）

$$
\mathbf p_{root}^{target}=\frac{h}{h_{ref}}s_{root}\mathbf p_{root}^{source}
$$

例如把 h/h\_ref 设为 1，手部距离骨盆 0.8 m、s\_hand=0.75，则目标相对距离为 0.60 m；脚部距离骨盆 0.9 m、s\_foot=0.9，则目标为 0.81 m。这只是说明原理，并非论文给出的具体系数。

**为什么可以减少脚滑？**原始人体支撑脚在地面近似不动；不合理的根节点与局部关键点缩放会使目标支撑脚产生漂移。GMR 通过“统一根节点平移＋分部位局部缩放”降低这种冲突。但它没有显式施加支撑脚零速度约束，故不能理论保证完全消除脚滑。

### Step 4｜第一阶段 IK：姿态 + 手脚等末端位置

优化广义构型 q（浮动基座的位姿与关节角）。第一阶段对所有匹配刚体施加姿态匹配，但**位置项只涉及手、脚等末端执行器**；并施加关节范围约束。这有利于先找到较合理的局部解，避免直接同时精确匹配所有关键点导致优化陷入较差的局部极小值。

论文式 (4)：第一阶段目标函数 · LaTeX 源码（可复制）

$$
\min_q\;\sum_{(i,j)\in M}w^R_{1,i,j}\|e^R_{i,j}(q)\|^2+\sum_{(i,j)\in M_{ee}}w^p_{1,i,j}\|p_i^{target}-p_j(q)\|^2;\quad q^-\le q\le q^+
$$

其中姿态误差可写为 e^R=Log((R\_i^h)^T R\_j(q)) 的三维李代数向量表示（便于理解的记法）。

作者调用 Mink 进行迭代 Differential IK：在当前姿态附近线性化任务误差，并求解带关节范围约束的局部最小二乘问题。

论文式 (5)：Differential IK · LaTeX 源码（可复制）

$$
\min_{\dot q}\;\|e(q)+J(q)\dot q\|_W^2;\quad q^-\le q+\dot q\Delta t\le q^+
$$

求得广义速度增量后积分更新构型；论文最多迭代 10 次，或直到目标函数变化低于 0.001。这里 Δt 为求解器参数，未必等于原始动作的帧周期。

### Step 5｜第二阶段 IK：全身位置 + 姿态

以第一阶段结果为初值，将位置目标从仅末端扩展到所有匹配关键刚体，并允许第二阶段使用一套不同权重，进一步提高全身动作保真度。

论文式 (6)：第二阶段目标函数 · LaTeX 源码（可复制）

$$
\min_q\;\sum_{(i,j)\in M}w^R_{2,i,j}\|e^R_{i,j}(q)\|^2+\sum_{(i,j)\in M}w^p_{2,i,j}\|p_i^{target}-p_j(q)\|^2;\quad q^-\le q\le q^+
$$

| 对比项 | 第一阶段 | 第二阶段 |
| --- | --- | --- |
| 姿态误差 | 所有匹配刚体 | 所有匹配刚体 |
| 位置误差 | 只包含手脚等末端 | 全部匹配关键刚体 |
| 初始解 | 上一帧结果或初始值 | 第一阶段解 |
| 作用 | 更稳健地得到合理姿态 | 提高全身匹配精度 |

处理完整序列时，逐帧执行两阶段 IK，上一帧重定向结果作为下一帧 Step 4 的初值（warm start）。全序列完成后，再用机器人刚体的最小高度统一校正全局平移，减轻穿地或悬空。

## 3. 与 PHC / ProtoMotions 的区别

| 方法 | 人体数据缩放 | 优化方式 |
| --- | --- | --- |
| PHC | SMPL 人体形状拟合到机器人 | 梯度优化 |
| ProtoMotions | 按世界坐标轴使用缩放因子 | Mink 微分 IK |
| **GMR** | **身体部位非均匀局部缩放 + 根节点全局缩放** | **两阶段 Mink 微分 IK** |

关键判断：GMR 并非提出一种全新的 IK 理论，而是指出缩放阶段本身就是参考轨迹质量的关键来源，然后通过局部缩放与分阶段的目标函数改善其结果。

## 4. 实验：重定向质量如何影响 RL 跟踪？

论文将 LAFAN1 中的 21 段动作映射至 Unitree G1，使用 BeyondMimic 训练各动作的跟踪策略。评价 sim / sim-dr / MuJoCo+ROS sim2sim 中的成功率以及关键刚体位置、姿态跟踪误差。

![图 D：由论文 Table II 的全局关键刚体平均位置误差重绘；单位 mm；越低越好](GMR_Zotero_Images/gmr_tracking_error_chart.png)

图 D：由论文 Table II 的全局关键刚体平均位置误差重绘；单位 mm；越低越好

例如 Run (stop & go) 的 sim2sim 成功率：PHC 为 74%，ProtoMotions 为 26%，GMR 为 100%，Unitree 为 100%（论文 Table I）。跨动作平均全局位置误差为 PHC 247.8 mm、ProtoMotions 139.7 mm、GMR 104.1 mm、Unitree 77.2 mm（Table II）。

![图 E：论文 Fig. 3：PHC 穿地、ProtoMotions 自穿透、GMR 腰部关节突跳（原文 p.7）](GMR_Zotero_Images/paper_fig3_artifacts.png)

图 E：论文 Fig. 3：PHC 穿地、ProtoMotions 自穿透、GMR 腰部关节突跳（原文 p.7）

![图 F：论文 Fig. 4：感知保真度用户实验（原文 p.7）](GMR_Zotero_Images/paper_fig4_user_study.png)

图 F：论文 Fig. 4：感知保真度用户实验（原文 p.7）

从用户研究看，GMR 在感知保真度上明显优于 PHC 与 ProtoMotions，但闭源 Unitree 数据仍被认为更接近原始人体动作。

## 5. 技术局限与研究启示

**（1）没有完整动力学保证。** 公式 (4)–(6) 是运动学层面的匹配与关节范围约束，不是显式的全身动力学可行性、接触力或摩擦锥约束。最终轨迹仍可能需要 RL 跟踪策略处理物理执行。

**（2）没有显式的跨帧速度 / 加速度连续约束。** 使用 warm start 有利于保持连续性，但无法保证不突跳。论文报告了 Dance 5 中 GMR 腰部 roll/pitch 突然跳变的问题。

**（3）实验范围有限。** 主要验证 LAFAN1 与 Unitree G1，未系统覆盖复杂环境交互、物体操作和其他机器人形态。

> **记忆重点：**身体局部缩放解决 geometry/morphology mismatch；两阶段 Differential IK 改善非凸任务优化的求解质量；逐帧 warm start 改善序列处理的连续性。GMR 输出 reference motion，后续 tracking controller 再解决如何执行。

来源定位：论文 Section IV pp.4–5，Table I–II p.6，Fig. 3–4 p.7。图 A、C、D 为基于文献信息重新绘制；图 B、E、F 摘自用户上传的原文 PDF。