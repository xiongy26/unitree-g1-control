# Unitree G1 · ZMP-based Walking in MuJoCo（浏览器 WebAssembly 版）

在 **浏览器里** 用 MuJoCo WebAssembly + three.js 实现 Unitree G1 人形机器人的
ZMP 行走 —— 复现参考视频《ZMP-based Walking with Weighted
Whole-Body Control》的技术路线，并新增 **LIPM-ZMP 线性 MPC** 作为可切换的
CoM 轨迹生成器：

```
足步规划器  →  ZMP 参考轨迹  →  CoM 轨迹生成【二选一，UI 下拉框切换】
    ├─ ZMP 预观控制 (Kajita 2003, cart-table LQR)    —— 默认
    └─ LIPM-ZMP 线性 MPC (Wieber 2006 / Herdt 2010)  —— 支撑域约束滚动时域 QP
    →  CoM 轨迹  →  加权全身逆向运动学 (weighted whole-body IK)  →  位置执行器
```

## 运行

```bash
# 1. 生成 web 资源（模型 / 网格 / 预观控制增益 / 渲染描述）
py scripts/export_web_model.py

# 2. 启动本地静态服务器
py scripts/serve_web.py 8765

# 3. 浏览器打开
#    http://127.0.0.1:8765/
```

页面功能：

- **实时 3D 行走**：物理完全在浏览器内运行（`@mujoco/mujoco` WASM，dt=2ms），
  **按真实时间 1:1 推进**（物理步数由墙钟节流，见 main.js 的 ticker）
- **绿色折线** = 规划 ZMP 参考（支撑脚序列），**白色曲线** = 实际 CoM 轨迹，
  **黄点** = 落脚点，右下角为俯视图分析面板
- **参数面板**：步数 / 步长 / 单支撑时长 / 双支撑时长 / 控制器（ZMP 预观 或
  MPC）随时可改，点击"重置"生效
- **⬅推 / 推➡ 按钮**：行走中施加 40N×0.15s 的侧向推力（质心速度变化约
  0.17 m/s，在两种模式的恢复包络内；连点可叠加超出包络）——MPC 与预观
  控制的差异主要在扰动恢复时可见（名义行走时两者指令几乎重合）
- **橙色曲线**（MPC 模式）= MPC 在俯视图上画出的未来 1.6s 预测 ZMP 轨迹，
  恢复过程中可以看到它贴着支撑域边缘走
- **暂停 / 相机跟随 / 拖拽视角 / 滚轮缩放**
- **⏺ 录制**：把 3D 视口实时录成 WebM 视频并自动下载（MediaRecorder 采集
  WebGL 画布，30fps / 8Mbps）。点击开始（按钮变红并显示时长），再点一次
  停止并下载 `g1_walk_<时间戳>.webm`；录制中可以照常暂停 / 重置 / 调视角

## 控制器结构（web/js/controller.js）

| 模块 | 说明 |
| --- | --- |
| `GaitPlan` | 足步序列（步长 = 相邻落脚点间距）、相位时间轴（init-DS → SS → DS → … → final-DS）、ZMP 参考（SS 相位位于支撑脚中心，DS 相位 smoothstep 过渡）、摆动腿 minjerk + 正弦抬脚轨迹 |
| `KajitaPreviewControl` 增益 | cart-table 模型离散化 → 增广状态（CoM 状态 + 未来 ZMP 参考窗）→ DARE 求得状态反馈 `Kx` 与预观增益 `kr`（由 `scripts/export_web_model.py` 用 scipy 预计算，见 `web/gains.json`） |
| `OnlinePreview.resim` | **滚动时域重预测**：每个 IK 周期从*实测* CoM 状态出发，用预观 LQR 闭环再仿真 cart-table 模型 0.6s。重预测轨迹从实测状态收敛回参考，其与参考的偏差即"当前状态误差沿稳定模态的传播" |
| `MpcPreview`（MPC 模式） | **LIPM-ZMP 线性 MPC**（Wieber 2006 / Herdt 2010）：决策变量 = 未来 1.6s 的 ZMP 序列（dt=0.04s, N=40），逐点约束在支撑多面体内（SS 取支撑脚矩形、DS 取双脚凸包，留 1.5cm 边距）；CoM 通过线性倒立摆的**精确 ZOH 离散**（cosh/sinh）响应；代价 = 名义 CoM 计划位置+速度跟踪 + 弱 ZMP 跟踪 + ZMP 一阶差分平滑；箱约束 ADMM 求解（M = H+ρI 一次分解复用，每周期 40 次热启动迭代，浏览器内实时） |
| MPC 命令律 | 与预观控制同构：`cmd = plan(t+lead) + β·(plan(t+lead) − mpcTraj(t+lead))`，钳在 plan±9cm 安全管。**计划承载倒计时深度**（滚动重解从静止状态出发天然偏浅）；MPC 轨迹从实测状态出发且满足支撑域约束，因此修正项是约束感知的。lead x/y = 0.06/0.12，β = 1.2 |
| 捕获步（MPC 模式） | 推扰后武装 2.5 s 的恢复机制：① `updatePlanClock` 按横向速度误差压缩/拉伸单支撑相位（向摆动侧倒→提前落脚；向支撑侧倒→延长制动），时钟上限放宽至 1.5；② `footTargetsUpdate` 把摆动脚落点向倾倒侧平移至多 5 cm（`plan.footY` 活引用 → ZMP 参考与 MPC 支撑约束自动跟随）；③ 名义 CoM 计划**刻意不**跟随新落点——命令律的 β 项需要未被迁就的计划作为参考，否则修正信号消失 |
| CoM 目标律 | `cmd = plan(t+lead) + β·(plan(t+lead) − resim(t+lead))`——把重预测偏差**沿参考镜像**（落后→指令超前参考，超前→制动），得到带 LQR 模态整形的位置/速度/加速度负反馈；`lead` 补偿指令→执行的传输滞后（横向经髋滚转滞后更大），偏差限幅在参考周围 ±5cm 的安全管内 |
| `updatePlanClock` | **自适应步态时钟**：计划时间轴按实测 CoM 矢状进度放慢/加快（rate 0.6~1.3），防止位置执行链滞后使 CoM 与计划脱节 |
| 触地重定时 | 摆动完成 60% 后一旦摆动脚物理触地，把计划时钟快进到该步 DS 起点，使 ZMP 参考的重量转移与真实接触同步开始（计划 CoM 轨迹跨相位连续，指令无跳变） |
| 伺服滞后补偿 | 输出级 `cmd = target + (target − actual)·0.7`，消除位置执行器稳态滞后（等效刚度 ×1.7、阻尼比 ×0.77）；消融实验表明这是稳定行走的**必要**机制 |
| `WalkingController.updateIK` | 加权最小二乘全身 IK（解析雅可比 `mj_jacSite`/`mj_jacSubtreeCom`/`mj_jacBody`）：支撑脚 6D(400) > 摆动脚 6D(120) > CoM(x40/y80/z120) > 骨盆姿态(30) > 姿态Reg(5) + 关节限位保护；**姿态任务权重加大以维持屈膝标称（膝 ≥0.4rad），远离直腿奇异性**——腿部伸直时位置伺服失去对骨盆高度的 authority，机器人会被接触力顶起（"弹簧高跷"失稳模态） |
| settle | 两段式：直接以屈膝姿态落地（脚掌贴地校正）→ WBC 平衡整定，CoM 平衡点 = 脚掌中心（与 ZMP 参考起点重合，零初始瞬态） |

## 已验证的稳定步态包络（无头仿真，Node + 同一 WASM）

| 步数 | 步长 | 结果 |
| --- | --- | --- |
| 4 / 8 / 12 / 16 / 20 | 0.10 m 或 0.12 m | 全部走完，不跌倒 |
| 默认 16 步 × 0.12 m | ≈1.9 m / 14 s | 横向 CoM 误差 < 9 cm，全程稳定 |
| 极限 | 24 步以上 | 收尾阶段累积失稳跌倒 |

**MPC 模式包络**（同一 harness、`--ctrl mpc`；结果为 12 个用例的扫描）：

| 用例 | ZMP 预观 | MPC |
| --- | --- | --- |
| 8 步 × {0.10, 0.12} × {0.3, 0.4} s | 全部 PASS | 全部 PASS，且跟踪误差普遍更低（如 8×0.10×0.4：横 3.3/矢 7.2 cm vs 2.8/9.4 cm） |
| 16 / 20 步 × 0.12 m × 0.4 s | PASS | PASS（16 步 13.5 s、20 步 16.4 s 走完） |
| 16 / 20 步 × 0.10 m × 0.4 s | 16 步 PASS；**20 步 12.1 s 失稳** | 全部 PASS——**20 步用例 MPC 更优**（15.4 s 走完，横 8.0 cm） |
| 快速步态（单支撑 0.3 s） | 全部 FAIL | 全部 FAIL——单支撑过短是位置执行链的固有限制，与 CoM 生成器无关 |

**侧向推扰对比**（`--push F,dur,t0`，默认步态）——名义行走时两模式指令几乎
重合（MPC 的命令律本就沿用预观控制的形式），差异在扰动恢复时显现。MPC 模式
额外带**捕获步机制**（推扰后武装 2.5 s：横向速度误差压缩/拉伸单支撑相位 +
摆动脚落点向倾倒侧平移至多 5 cm，ZMP 参考与 MPC 支撑约束自动跟随新落点）：

| 推扰 | ZMP 预观 | MPC（含捕获步） |
| --- | --- | --- |
| 30 N × 0.15 s @ 2.5 / 3.0 / 3.25 s | @3.0 时 8.1 s 摔倒 | **三个时机全部恢复走完** |
| 50 N × 0.15 s @ 3.0 s | 4.9 s 摔倒 | **13.8 s 走完** |
| 60–70 N × 0.15 s | 摔倒 | 摔倒（固定落脚点架构的共同边界；完整恢复需落脚点自适应重规划） |

注意：捕获步机制只在真实推扰后武装——名义摇摆的偏差与中等推扰同量级，
不加门槛的捕获逻辑会破坏名义步态。

步态质量：摆动脚实际离地约 **5 cm**（目标 10 cm，IK 加权折衷 + 位置执行器
滞后各吃掉一半；进一步提高摆动权重/增益会进入 IK-执行器闭环的不稳定区），
矢状向 CoM 跟踪误差 < 12 cm，行进距离约为计划的 85-95%。

- 已知局限：步长 ≥ 0.14 m、单支撑 ≥ 0.5 s、或 24 步以上的超长行走会失稳。
- `scripts/sim_node.mjs` 是无头回归 harness：`node scripts/sim_node.mjs
  [--steps N] [--len L] [--ss T] [--ds T] [--ctrl zmp|mpc] [--push F,dur,t0]
  [--mpcleadx T] [--mpcleady T] [--mpcbeta B] [--mpctube M] [--mpcqcom Q]
  [--mpcqvel Q] [--mpcq Q] [--mpcr R] [--mpcmargin M] [--mpciter N]`，用与
  浏览器完全相同的 controller.js + 模型做快速定量验证，输出 PASS/FAIL。

## 调试过程中修掉的关键 bug（备忘）

1. **`minjerk` 笔误（致命）**：五次多项式应为 `s³(10−15s+6s²)`，
   原代码写成 `s³(10−15s+6s)`，`minjerk(0.84)=1.45 > 1`——摆动脚轨迹中后段
   冲到落点前方 40%、z 冲到地面以下 3.4 cm，摆动脚中途砸地拖行 → CoP 离开
   支撑脚 → 重量转移失败 → 横向发散。这是"走 3~4 步跌倒"的直接根因。
2. **mesh 渲染散架（两代修复）**：MuJoCo 编译时将 mesh 顶点重新居中并折叠进
   `geom_pos/mesh_pos/mesh_quat`。第一版用 three.js 加载原始 STL 再手工做
   `X_geom ∘ M⁻¹` 补偿——手臂部分补偿出错仍然散架。现方案（参照
   g1-kitchen-web / zalo/mujoco_wasm）：直接从编译后模型的 `model.mesh_vert`
   / `mesh_face` 缓冲构建几何体并直接施加 `geom_pos/geom_quat`，编译期重居中
   天然自洽，整类补偿 bug 不复存在。注意：WASM 堆视图 `.slice()` 后必须
   `new Float32Array(copy)` / `Array.from(faces)` 再交给 three.js（它需要
   拥有 `byteLength` 的真实 buffer），否则渲染报 createBuffer 错误。
3. WASM 绑定的缓冲区方法是 `GetView()`（README 写的 `getView` 有误），
   `DoubleBuffer` 构造用 `FromArray`。
4. 姿态任务误差符号写反 = "反姿态任务"，会把关节推离标称姿态。
5. 后台标签页的 window 定时器被节流到 ~1Hz —— 物理步进必须用 Web Worker 计时器。
6. MuJoCo 是 z 轴朝上，three.js 是 y 轴朝上：场景根节点需绕 X 轴 -90°，
   相机目标 (x, z, -y)。
7. 膝关节伸直（kneeMin=0.15 太小 + 姿态任务太弱 0.5）导致腿部接近奇异：
   直腿是刚性撑杆，位置伺服无法再控制骨盆高度，接触力把机器人顶起、双脚
   悬空（实测支撑脚抬起 7 cm 而指令在地面）。姿态任务 0.5→5 + kneeMin 0.4
   保持屈膝后彻底解决。

## 参考的开源项目

- [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) — Unitree G1（29 自由度）MJCF 模型与网格
- [google-deepmind/mujoco（wasm）](https://github.com/google-deepmind/mujoco/tree/main/wasm) / npm [`@mujoco/mujoco`](https://www.npmjs.com/package/@mujoco/mujoco) — 官方 WebAssembly 绑定
- [kevinzakka/mink](https://github.com/kevinzakka/mink) — 加权任务 QP-IK 的任务定义/权重范式
- Kajita 2003《Biped Walking Pattern Generation by using Preview Control of ZMP》及社区实现
  [chauby/ZMP_preview_control](https://github.com/chauby/ZMP_preview_control)、
  [ekorudiawan/ZMP-Preview-Control-WPG](https://github.com/ekorudiawan/ZMP-Preview-Control-WPG)、
  [zanppa/WPG](https://github.com/zanppa/WPG)、
  [rdesarz/lipm-walking-controller](https://github.com/rdesarz/lipm-walking-controller)
- MPC 模式：Wieber 2006《Trajectory Free Linear Model Predictive Control for
  Stable Walking in the Presence of Strong Perturbations》、
  Herdt et al. 2010《Iterative Predictive Control for Stable Interleaved
  Walking and Push Recovery》——决策变量取未来 ZMP 序列 + 支撑域约束的
  标准形式；[stephane-caron/lipm_walking_controller](https://github.com/stephane-caron/lipm_walking_controller)
  是同一思路的完整实现参考

## 目录结构

```
web/                     浏览器应用（index.html + js/{main,controller,viewer}.js）
  lib/package/           @mujoco/mujoco WASM 绑定
  lib/three/             three.js + OrbitControls
  model/                 自包含场景 XML + STL 网格（供 WASM 虚拟文件系统编译；
                         渲染几何体直接取自编译后模型的 mesh 缓冲，不用 STL）
  gains.json             ZMP 预观控制增益（Kx, kr；MPC 复用其中的 zc/g）
scripts/export_web_model.py   生成 web 模型/增益/渲染描述（内联增益计算）
scripts/serve_web.py     静态服务器（正确 .wasm MIME、no-store）
scripts/sim_node.mjs     Node 无头回归 harness（与浏览器同一 controller.js；
                         --ctrl zmp|mpc 选择 CoM 轨迹生成器）
models/g1_walk.xml       G1 模型源（足底摩擦 1.2、执行器增强版 kp=1600）
third_party/mujoco_menagerie   上游模型资产（仅 meshes 被引用）
```

web/js/controller.js 导出：`GaitPlan`（足步规划，两模式共用）、
`OnlinePreview`（Kajita 预观控制，默认）、`MpcPreview`（LIPM-ZMP 线性 MPC）、
`WalkingController`（加权全身 IK + 状态机，经 `params.controller` 分派）。
