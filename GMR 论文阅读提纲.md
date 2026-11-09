# GMR 论文阅读提纲

> Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking Araujo et al., Stanford University, arXiv:2510.02252

------

## 阅读前：建立问题意识

在开始阅读前，先思考以下几个问题，读完后回来填写你的答案：

- 为什么retargeting的质量很重要？现有方法是怎么"绕过"这个问题的？
- 如何公平地衡量一个retargeting方法的好坏？
- GMR解决的核心矛盾是什么？

📝 **你的思考：**

1. retargeting的质量直接影响后期motion imitation policy训练的质量，现有方法都是用过reward tunning or domain randomization解决retargeting引入的artifacts.

2. 论文使用了3个量化指标和user直观感受相结合的方式，量化指标包括：1.全局坐标系下身体部位的位置偏差 2. 身体部位相对于根节点的位置偏差 3. 关节角度偏差

3. 在阐述核心矛盾之前，我想先梳理一遍文章的核心contributions，作者提供了一套novel retarget框架，能够实现所有种类机器人的retarget （general 属性），real-time(fast 属性)， 高保真（high performance 属性）。旨在解决人形机器人领域一个非常重要的问题，机器人运动的policy的训练依赖高质量的数值（难以获取），之前的retatget方法存在脚步滑动，脚步穿透，身体自穿透等问题。本文的提出的retarget方法有general、real-time、high-performance等属性，使得操作员采集到的动作能够很好地retarget到机器人上用于训练，本文对比了PHC和Protomotion两个经典方法，使用BeyondMimic作为同一个motion imitation policy用于评估retarget的效果。

   如果说GMR解决了什么核心矛盾，我认为有   1. 通过非均匀局部缩放精细化了不同部位的retarget过程，同时使用两阶段Mink IK高效而精细化地求取了一个较好的解，在这样的处理方式下，之前方法存在的artifacts被显著解决了。2.在之前的工作中，没有一个统一的retarget评价体系，作者在GMR的实验部分提出了一套retarget performance评价体系，以客观量化指标+ 用户主观评价相结合的方式，后续研究可在此评价体系的基础之上进行改进和优化。

------

## 第一章：Introduction

### 核心逻辑链

```
人形机器人需要学习动作
→ 人类动作数据是最好的来源
→ 但人和机器人形态不同（embodiment gap）
→ 需要retargeting做"翻译"
→ 现有retargeting会引入artifacts
→ 过去靠reward tuning掩盖问题
→ 本文：不tuning，直接测retarget质量的影响
```

### 需要重点关注

- [ ] 什么是 embodiment gap？包含哪些维度？
- [ ] 作者的核心假设（hypothesis）是什么？
- [ ] 本文三个贡献分别是什么？

<details> <summary>💡 参考答案</summary>
**embodiment gap 的维度：** 

论文明确列出了六个维度：

1. 骨骼长度（bone length）、

2. 关节运动范围（joint range of motion）

3. 运动学结构（kinematic structure）、
4. 体型（body shape）、
5. 质量分布（mass distribution）
6. 驱动机制（actuation mechanisms）。核心问题是人的关节是球关节（SO(3)），而机器人通常只有转动关节（revolute joints），这个结构差异是最根本的。

**作者的核心假设：** "通过足够的reward function设计和domain randomization，retargeting引入的artifacts基本上可以被掩盖或消除。但如果没有这些工程努力，retargeting的质量就会扮演重要角色。" 本文的目的正是在**压制**这些工程手段的情况下，验证这个假设，从而隔离出retargeting本身的影响。

**本文三个贡献：**

1. 提出GMR这个新的retargeting方法，解决了其他方法中的glaring artifacts
2. 对retargeted reference motion质量如何影响策略性能进行了系统性研究
3. （隐含）建立了一套相对完整的评估框架（定量指标+用户研究+多评估环境）

</details>

### 关键术语

| 术语                 | 含义                                                 |
| -------------------- | ---------------------------------------------------- |
| Motion Retargeting   | 将人类动作映射到机器人的过程                         |
| Embodiment Gap       | 人与机器人在骨骼长度、关节范围、质量分布等方面的差异 |
| Artifacts            | 重定向引入的瑕疵，如脚滑、穿透、关节突变             |
| Zero-shot deployment | 仿真训练后直接部署到真实世界，不做额外调试           |

> 📝 **读完本章后的思考：**
>
> - 作者批评prior work的点是否合理？
> - 这个研究问题有什么实际意义？
>
> （在此填写）

------

## 第二章：Related Work

### 阅读策略

> 这章不需要逐字精读，重点建立领域地图。

### 领域分支地图

```
Motion Retargeting
├── 计算机图形学方向
│   ├── 经典优化方法（PHC前身）
│   └── 数据驱动方法（需要配对数据）
└── 机器人学方向
    ├── 朴素方法（直接复制关节角）
    ├── Whole-body Geometric Retargeting (WBGR)
    │   ├── PHC（基于SMPL）
    │   └── ProtoMotions（全局轴对齐缩放）
    └── 本文：GMR（非均匀局部缩放）
```

### 需要重点关注

- [ ] 为什么data-driven retargeting在机器人上难以应用？
- [ ] PHC和ProtoMotions各自的局限在哪里？（读完后对比第四章）

<details> <summary>💡 参考答案</summary>
**data-driven retargeting 的困难：** 数据驱动方法通常需要配对数据（paired data）做监督学习，或者语义标签做无监督训练。在机器人上，很难收集"人做同一个动作、机器人也做同一个动作"这样的配对数据，因为机器人本身还没有能力稳定执行这些动作——这是一个先有鸡还是先有蛋的问题。此外已有的learning-based方法大多只处理上半身或简单手臂动作，全身locomotion的retargeting几乎没有被data-driven方法解决。

**PHC的核心局限：**

- SMPL是为人体设计的，对形态差异较大的机器人拟合效果差（机器人腿比例、关节位置都和人类有显著差异）
- 完全不考虑接触状态，导致脚部穿透和浮空
- 梯度下降IK速度慢，对实时应用不友好

**ProtoMotions的核心局限：**

- 全局轴对齐缩放（x/y/z三个轴各一个系数）太粗糙，上下半身缩放需求不同，强行统一导致比例失真

- 同样不考虑接触，腿部自相交是常见问题（尤其是快速运动如跑步）

  

  

  **Background**

  **PHC:**

  （Skinned Multi-Person Linear Model）是一个参数化三维人体模型。

  它主要通过两组参数描述人体：

  $V_h=f_{\mathrm{SMPL}}(\beta,\theta) $

  其中：

  - \($\beta$\)：Shape Parameters，人体形状参数，例如身高比例、躯干宽度和肢体粗细等。
  - \($\theta$\)：Pose Parameters，人体姿态参数，描述各个关节的旋转。
  - \($V_h$\)：由参数生成的三维人体网格顶点。

  我自己针对PHC和ProtoMotions的理解，PHC将原始基于人体的SMPL模型找到一组形状参数$\beta_h$，让SMPL模型在此参数下和人形机器人的形状尽可能一致，再把原始人体动作的姿态参数$\theta$ 应用到拟合以后SMPL模型上，这样得到匹配二者动作一致性的关键位置。第二阶段再通过IK求解机器人的关节角度
  $$
  \begin{aligned} \min_{\mathbf p_t,R_t,\mathbf q_t}\quad& \left\| \mathbf p_t^{target} - FK(\mathbf p_t,R_t,\mathbf q_t) \right\|_2^2 \\ \text{s.t.}\quad& \mathbf q^-\leq\mathbf q_t\leq\mathbf q^+ \end{aligned}
  $$

- 



**ProtoMotions:**

第一阶段ProtoMotions 不通过 SMPL 形状参数拟合机器人，而是直接在笛卡尔空间对人体关键点进行缩放。

首先，根据世界坐标系 X、Y、Z 三个方向分别设置缩放系数，对人体关键点的位置进行调整：
$$
\mathbf p_i^{target}=S\mathbf p_i^{source},
\quad
S=\operatorname{diag}(s_x,s_y,s_z)
$$
其中，$s_x,s_y,s_z$ 分别表示三个坐标轴方向的缩放系数。



第二阶段使用 Mink 微分逆运动学求解器，通过最小化机器人关键刚体与人体目标运动之间的位置和姿态误差，得到机器人关节构型。

与直接求解非线性 IK 不同，微分 IK 首先在当前构型附近利用雅可比矩阵对任务误差进行线性化：
$$
\mathbf e(\mathbf q^{k+1})
\approx
\mathbf e(\mathbf q^k)+J_e(\mathbf q^k)\dot{\mathbf q}\Delta t
$$
然后通过求解局部 QP 得到广义速度：
$$
\begin{aligned}
\min_{\dot{\mathbf q}}\quad&
\left\|
\mathbf e(\mathbf q^k)
+
J_e(\mathbf q^k)\dot{\mathbf q}\Delta t
\right\|_W^2\\
\text{s.t.}\quad&
\mathbf q^-
\leq\mathbf q^k+\dot{\mathbf q}\Delta t
\leq\mathbf q^+
\end{aligned}
$$
最后通过积分更新机器人构型，并重复优化：
$$
\mathbf q^{k+1}=\mathbf q^k+\dot{\mathbf q}^k\Delta t
$$
通过逐帧迭代优化，算法得到机器人浮动基座位姿及关节角度的参考序列 $\mathbf q_{\mathrm{ref}}(t_j)$。对于后续 Motion Tracking 或强化学习训练，可以根据相邻参考帧及采样周期计算对应的广义速度序列 $\mathbf v_{\mathrm{ref}}(t_j)$，构造完整的参考运动数据。

总结起来：传统非线性 IK 直接优化关节构型，而 Differential IK 在当前构型附近求解使任务误差减小的局部运动增量，再通过迭代逐步逼近目标构型。

**局限性：** ProtoMotions 对所有身体部位采用相同的全局坐标轴缩放系数，无法针对手臂、腿部和躯干的不同长度比例独立调整，因此可能引入足底滑动和自穿透等问题。

------

## 第三章：Evaluation Method

### 章节结构

```
A. 对比的四个Retargeting方法
   ├── PHC
   ├── ProtoMotions  
   ├── GMR（本文）
   └── Unitree（闭源基准）

B. 数据处理（LAFAN1数据集子集）

C. Motion Tracking评估方式
   ├── sim（无domain randomization）
   ├── sim-dr（有domain randomization）
   └── sim2sim（MuJoCo/ROS仿真）

D. 评估指标

E. 用户研究设计
```

### 需要重点关注

<details> <summary>💡 参考答案</summary>
**为什么选BeyondMimic：** 公平性的关键在于BeyondMimic是**独立于所有retargeting方法开发的**，且不需要reward tuning和extensive domain randomization。如果用需要大量调参的训练框架，实验者可以针对某种retarget方式专门调出好结果，从而掩盖retargeting本身的质量差异。BeyondMimic的"不tuning"特性让retargeting质量的影响能够直接暴露出来。

**成功率的严格性：** 定义是：策略必须**完整跑完整个参考动作序列**，且全程不能摔倒（anchor body的高度和朝向不超过阈值）。严格在于：只要中途有任何一帧失败就算失败，哪怕只差最后一秒。对于长达2分钟的Dance序列，这意味着策略需要持续稳定超过数千帧，任何一次细小扰动都可能导致失败。

**三个指标的区别：**

- Eg-mpbpe：全局坐标误差，反映机器人在空间中的绝对位置是否跟上了参考（对根节点漂移敏感）
- Empbpe：相对根节点的误差，反映身体各部位相对姿态是否正确（去除了根节点位置误差的影响）
- Empjpe：关节角度误差，最细粒度，直接衡量每个关节的旋转是否正确

三者结合可以区分"全局位置漂了但姿态对"和"姿态本身就错了"这两种不同的失败模式。

**用户研究的潜在偏差：**

- 样本量较小（N=20），统计显著性有限
- 视频渲染风格可能不同（GMR用机器人模型，其他方法类似），视觉差异可能干扰判断
- 用户不是领域专家，对"物理合理性"的感知可能与专家不同
- 5秒随机片段可能恰好选到各方法差异大或差异小的段落，结果有随机性

</details>

### 评估指标速览

| 指标         | 全称                                | 衡量内容                   |
| ------------ | ----------------------------------- | -------------------------- |
| Eg-mpbpe     | Global Mean Per Body Position Error | 全局坐标下的身体位置误差   |
| Empbpe       | Mean Per Body Position Error        | 相对根节点的身体位置误差   |
| Empjpe       | Mean Per Joint Position Error       | 关节旋转角度误差           |
| Success Rate | —                                   | 完整完成动作且不摔倒的比例 |

> 📝 **读完本章后的思考：**
>
> - 这套评估体系有哪些局限性？
> - 如果你来设计评估，你会改变什么？
>
> （在此填写）

------

## 第四章：General Motion Retargeting (GMR)

### ⭐ 核心章节，需要精读

### GMR五步流程

```
Step 1: Human-Robot Key Body Matching
        → 建立人体关键部位与机器人部位的对应关系

Step 2: Cartesian Space Rest Pose Alignment  
        → 对齐静止姿态，缓解toed-in等系统性偏差

Step 3: Non-Uniform Local Scaling  ← 核心创新点
        → 每个关键部位有独立缩放系数
        → Root平移用统一缩放（防止脚滑）

Step 4: IK with Rotation Constraints（第一阶段）
        → 只约束朝向 + 末端执行器位置
        → 得到全局合理的粗糙解

Step 5: IK with Rotation & Translation Constraints（第二阶段）
        → 以Step4结果为初始值
        → 加入所有关键部位的位置约束
        → 精细优化
```

### 关键公式理解

**step 3: 非均匀局部缩放公式（Eq.2）：** 
$$
p^{target}_b = \frac{h}{h{ref}} s_b (p^{source}_j - p^{source}_{root}) + \frac{h}{h_{ref}} s_{root} p^{source}_{root}
$$

- $h$：源人体身高，$h_{ref}$：参考身高，$h_{ref} $是设置缩放参数时的参考基准，$h$ 是当前输入数据的人体高度。两个高度相除，就是为了让预设的局部缩放系数能够随输入人体的高度变化。
- $s_b$：该部位的自定义缩放系数（**非均匀的关键**）
- 意义：先将关节位置相对根节点归一化，再按各自系数缩放
- 这里我的理解是匹配key bodies的同时，让人和机器人的坐标系朝向也一致，再按照不同位置的key bodies选择不同的缩放系数。

**经过 Step 3 缩放后，得到一组 Key Bodies 的目标位置和姿态：**
$$
\left\{ \mathbf p_i^{target},R_i^{target} \right\}_{i=1}^{m}
$$
**接下来是two-step IK:**

**step 4: IK with Rotation Constraints（第一阶段）**

假设原始人体做一个右手上举的动作。

我们最希望实现的是：

- 机器人右手到达对应的目标位置。
- 肩部、上臂、前臂等 Key Bodies 的朝向接近人体动作。
- 关节角度满足机器人运动学约束。

**但此时，不一定要求机器人肘部的空间位置与人体肘部完全匹配。这样就给优化留出了一定自由度，允许机器人根据自身的手臂长度和关节结构调整动作。**
$$
\boxed{
\begin{aligned}
\min_{\mathbf q}\quad&
\sum_{(i,j)\in\mathcal M}
w_{1,i,j}^{R}
\|\mathbf e_{i,j}^{R}(\mathbf q)\|_2^2\\
&+
\sum_{(i,j)\in\mathcal M_{ee}}
w_{1,i,j}^{p}
\|\mathbf p_i^{target}-\mathbf p_j(\mathbf q)\|_2^2\\
\text{s.t.}\quad&
\mathbf q^-\leq\mathbf q\leq\mathbf q^+
\end{aligned}
}
$$
第一项匹配所有key bodies的姿态，第二项匹配所有末端的位置。这样的好处是既可以让机器人的动作姿态和末端匹配上，又能保留一些自由度，避免因为优化指标过多而发生冲突。第一步类似于先粗略匹配动作形态

**step 4的求解利用迭代式Mink Differential IK 差分IK（Eq.5）：**
$$
\min_{\dot{q}} |e(q) + J(q)\dot{q}|^2_W \quad \text{s.t.} \quad q^- \leq q + \dot{q}\Delta t \leq q^+
$$

- 求解广义速度 $\dot{q}$ 而非直接求解 $q$
- 沿Jacobian方向迭代逼近目标，避免大步跳跃



**step 5: IK with Rotation & Translation Constraints（第二阶段）**

论文式（6）：
$$
 \boxed{ \begin{aligned} \min_{\mathbf q}\quad& \sum_{(i,j)\in\mathcal M} w_{2,i,j}^{R} \|\mathbf e_{i,j}^{R}(\mathbf q)\|_2^2\\ &+ \sum_{(i,j)\in\mathcal M} w_{2,i,j}^{p} \|\mathbf p_i^{target}-\mathbf p_j(\mathbf q)\|_2^2\\ \text{s.t.}\quad& \mathbf q^-\leq\mathbf q\leq\mathbf q^+ \end{aligned} } 
$$
针对全身key bodies，优化位置和姿态误差，这一步使用step 4的解作为step 5 Mink Differential IK求解器的初值，step 5的意义是利用step 4求解出的比较合理和接近的关节位置作为初值继续找一组更加匹配的关节空间的解，这样分布式的求解方法是为了降低非凸优化问题陷入局部最优解的风险

### 需要重点关注

- [ ] 为什么统一缩放root平移可以避免脚滑？（想清楚几何原因）
- [ ] 两阶段IK的必要性：第一阶段只用朝向约束，为什么能帮助避免局部最优？
- [ ] 与PHC的关键区别：不依赖SMPL，可以直接处理BVH格式，意味着什么？
- [ ] GMR的局限：什么情况下会出现关节值突变？

<details> <summary>💡 参考答案</summary>
**统一缩放root平移为何避免脚滑：** 脚滑的本质是：脚在地面接触时，世界坐标下的脚部位置发生了移动。如果root的水平位移被非均匀缩放（比如缩放比例随时间变化），就会改变脚相对于地面的绝对位置，即使脚在角色局部坐标里是"静止"的，在世界坐标里仍然会滑动。用统一的固定系数缩放root平移，保证了root的运动轨迹是原始轨迹的等比缩放，脚部在接触期间的世界坐标变化也会被一致地缩小，从而减少滑动。

**两阶段IK避免局部最优：** IK问题的优化landscape通常是非凸的，直接用全约束优化很容易陷入局部极小值（例如关节选择了"肘关节朝外"而非"朝内"这样的对称解）。第一阶段只约束朝向和末端执行器，自由度更大，更容易找到一个全局合理的构型（相当于先确定四肢的大致姿态）。第二阶段再以此为初始点加入所有位置约束做精细化，等于给第二阶段一个好的初始猜测，大大降低陷入坏局部最优的概率。这是一种coarse-to-fine的优化策略。

**不依赖SMPL的意义：** SMPL是一个参数化人体模型，用它来"套"机器人形状本质上是在用人体的先验约束机器人——对于腿长比例、关节位置和人类差异很大的机器人（如G1）效果会变差。GMR直接从BVH格式读取骨架结构，**对源数据格式更通用**（BVH是mocap的标准格式），对目标机器人也没有SMPL拟合步骤，因此理论上可以更灵活地适配形态差异较大的机器人，扩展性更好。

**GMR关节值突变的原因：** 差分IK本质上是一种局部线性化的迭代方法，当运动中出现**大幅度、快速的姿态变化**时（如Dance 5中的某些剧烈转体动作），从上一帧的解出发的线性近似可能跨越到另一个局部最优区域，导致某个关节（如腰部roll）在相邻帧之间出现跳变。这是差分IK的固有局限，本质上是因为Jacobian在这些点附近的线性化误差过大。作者也承认这需要针对特定动作进行权重调整。

</details>

> 📝 **读完本章后的思考：**
>
> - 非均匀缩放的想法是否直觉上合理？
> - 两阶段IK是一种通用技巧还是针对这个问题的特殊设计？
> - 你觉得GMR还有哪些可以改进的地方？
>
> （在此填写）

------

## 第五章：Evaluation Results

### 章节结构

```
A. retargeting方法对策略表现的影响（回答Q1）
B. retargeting artifacts的影响（回答Q2）
C. 用户研究结果（回答Q3）
D. 起始帧的影响
```

### 关键发现速览

**哪些动作所有方法都能成功（>98%）：** Turn 2, Walk (army), Walk (knees), Dance 3/4, Run (slow), Run, Hopscotch, Jump and rotate, Kung fu, Various sports

**区分方法优劣的关键动作：**

| 动作            | 问题方法               | 原因                 |
| --------------- | ---------------------- | -------------------- |
| Dance 1 & 2     | PHC完全失败(0%)        | 地面穿透（最深60cm） |
| Run (stop & go) | ProtoMotions低(20-26%) | 腿部自相交           |
| Dance 5         | GMR下降(51-96%)        | 腰部roll值突变       |

**Tracking Error结论（Tab. II）：**

- 排名：Unitree > GMR > ProtoMotions > PHC
- PHC的Max误差远高于其他方法（Mean被Dance失败拉高）

### 需要重点关注

- [ ] Table I：找出GMR和ProtoMotions表现差异最大的动作，思考原因
- [ ] Table II：为什么PHC的Mean误差远高于Median？（提示：Dance 1/2失败）
- [ ] Fig. 4用户研究：GMR vs Unitree的偏好差距有多大？
- [ ] Tab. III：起始帧如何影响Turn 1的Unitree结果？

<details> <summary>💡 参考答案</summary>

**GMR vs ProtoMotions 差异最大的动作：** 最显著的是 **Dance 5** 和 **Run (stop & go)**：

- Dance 5：GMR在sim2sim中只有51%，而ProtoMotions达到100%。原因是GMR的腰部roll突变artifact恰好在这个动作里出现，而ProtoMotions没有这个问题。
- Run (stop & go)：ProtoMotions只有20-26%（腿部自相交），GMR达到98-100%。 这两个案例说明两种方法各有"软肋"，但GMR的软肋覆盖面更小（只影响<2%的数据）。

**PHC的Mean远高于Median的原因：** Dance 1和Dance 2在PHC下成功率为0（策略完全无法学习），这意味着这两个序列中策略很快就倒下了。倒下之前的那几帧tracking error会被计入（论文说"tracking errors are computed for all frames that a policy is alive"），但由于策略根本没有学会跟踪，误差极大。这几个outlier序列把Mean拉得很高，而Median对异常值不敏感，所以Median保持在正常范围。这也说明用Mean来报告tracking error时需要注意是否有失败案例混入。

**GMR vs Unitree 用户研究差距：** 从Fig. 4来看，Unitree vs GMR的对比中，37%偏好GMR，38%无法区分，25%偏好Unitree。也就是说**多数用户要么选GMR要么无法区分**，只有1/4的用户认为Unitree更像原始动作。这个差距相当小，说明GMR的感知质量已经非常接近闭源的Unitree方案。对比PHC（74%偏好GMR）和ProtoMotions（63%偏好GMR），差距非常显著。

**起始帧对Turn 1的影响：** Turn 1从第0帧开始时，Unitree只有47%的成功率；从第49帧开始则达到100%。这说明即使是质量最好的Unitree retarget，如果起始帧的姿态让机器人难以从默认pose安全过渡到参考pose，策略也会在起步时立即失败。这个发现与retargeting方法本身无关，而是一个独立的工程问题——需要确保参考动作的起始帧姿态"友好"。

</details>

> 📝 **读完本章后的思考：**
>
> - 实验结果是否支持了Introduction中的假设？
> - 有哪些结果是你没有预料到的？
> - 这些结果对实际工程应用有什么指导意义？
>
> （在此填写）

------

## 第六章：Conclusion

### 核心结论

1. Retargeting质量确实显著影响策略表现（✅ Q1）
2. 三类致命artifacts：地面穿透、自相交、关节值突变（✅ Q2）
3. GMR在忠实度上优于PHC和ProtoMotions，接近Unitree（✅ Q3）
4. 起始帧的选择同样重要

### 作者承认的局限

- 只用了LAFAN1一个数据集
- 只测试了Unitree G1一款机器人
- 未涉及有物体交互的动作

> 📝 **读完本章后的思考：**
>
> - 这些局限性中，你认为哪个最需要被解决？
> - 如果做后续工作，你会选择哪个方向？
>
> （在此填写）

------

## 全文读完后：综合反思

### 论文整体评价

**优点：**

> （在此填写）

**不足或可质疑的地方：**

> （在此填写）

**对你自己研究方向的启示：**

> （在此填写）

### 遗留问题清单

> 记录读完后仍然不清楚或想继续深究的问题

- [ ] 问题1：作者宣称每个部位都采用了非均匀缩放系数sb，但作者却没有在论文中说明是如何获得的，如果在代码中这个系数是硬编码进去的，是否说明更换一种机器人后就需要重新调整？
- [ ] 问题2：
- [ ] 问题3：

### 相关论文延伸阅读

基于本文引用，推荐优先级：

| 论文              | 关键性 | 原因                                                         |
| ----------------- | ------ | ------------------------------------------------------------ |
| BeyondMimic [15]  | ⭐⭐⭐    | Motion imitation policy，本文使用的训练框架，理解它对理解实验设置至关重要 |
| PHC [38]          | ⭐⭐⭐    | 本文的主要对比方法，需要了解其细节                           |
| ProtoMotions [40] | ⭐⭐     | 另一个对比方法                                               |
| H2O/SMPL [37]     | ⭐⭐     | PHC方法的基础，理解SMPL模型                                  |
| Mink [41]         | ⭐      | GMR使用的IK求解器                                            |

------

*提纲版本：v1.0 | 基于 arXiv:2510.02252v1*