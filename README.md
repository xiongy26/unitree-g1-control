# Unitree G1 · ZMP-based Walking in MuJoCo（浏览器 WebAssembly 版）

在 **浏览器里** 用 MuJoCo WebAssembly + three.js 实现 Unitree G1 人形机器人的
ZMP 预观控制行走 —— 复现参考视频《ZMP-based Walking with Weighted
Whole-Body Control》的技术路线：

```
足步规划器  →  ZMP 参考轨迹  →  ZMP 预观控制 (Kajita 2003, cart-table LQR)
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
- **参数面板**：步数 / 步长 / 单支撑时长 / 双支撑时长随时可改，点击"重置"生效
- **暂停 / 相机跟随 / 拖拽视角 / 滚轮缩放**

## 控制器结构（web/js/controller.js）

| 模块 | 说明 |
| --- | --- |
| `GaitPlan` | 足步序列（步长 = 相邻落脚点间距）、相位时间轴（init-DS → SS → DS → … → final-DS）、ZMP 参考（SS 相位位于支撑脚中心，DS 相位 smoothstep 过渡）、摆动腿 minjerk + 正弦抬脚轨迹 |
| `KajitaPreviewControl` 增益 | cart-table 模型离散化 → 增广状态（CoM 状态 + 未来 ZMP 参考窗）→ DARE 求得状态反馈 `Kx` 与预观增益 `kr`（由 `scripts/export_web_model.py` 用 scipy 预计算，见 `web/gains.json`） |
| `OnlinePreview.resim` | **滚动时域重预测**：每个 IK 周期从*实测* CoM 状态出发，用预观 LQR 闭环再仿真 cart-table 模型 0.6s。重预测轨迹从实测状态收敛回参考，其与参考的偏差即"当前状态误差沿稳定模态的传播" |
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

步态质量：摆动脚实际离地约 **5 cm**（目标 10 cm，IK 加权折衷 + 位置执行器
滞后各吃掉一半；进一步提高摆动权重/增益会进入 IK-执行器闭环的不稳定区），
矢状向 CoM 跟踪误差 < 12 cm，行进距离约为计划的 85-95%。

- 已知局限：步长 ≥ 0.14 m、单支撑 ≥ 0.5 s、或 24 步以上的超长行走会失稳。
- `scripts/sim_node.mjs` 是无头回归 harness：`node scripts/sim_node.mjs
  [--steps N] [--len L] [--ss T] [--ds T]`，用与浏览器完全相同的
  controller.js + 模型做快速定量验证，输出 PASS/FAIL。

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

## 目录结构

```
web/                     浏览器应用（index.html + js/{main,controller,viewer}.js）
  lib/package/           @mujoco/mujoco WASM 绑定
  lib/three/             three.js + OrbitControls
  model/                 自包含场景 XML + STL 网格（供 WASM 虚拟文件系统编译；
                         渲染几何体直接取自编译后模型的 mesh 缓冲，不用 STL）
  gains.json             ZMP 预观控制增益（Kx, kr）
scripts/export_web_model.py   生成 web 模型/增益/渲染描述（内联增益计算）
scripts/serve_web.py     静态服务器（正确 .wasm MIME、no-store）
scripts/sim_node.mjs     Node 无头回归 harness（与浏览器同一 controller.js）
models/g1_walk.xml       G1 模型源（足底摩擦 1.2、执行器增强版 kp=1600）
third_party/mujoco_menagerie   上游模型资产（仅 meshes 被引用）
```
