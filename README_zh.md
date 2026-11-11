# S1Pert 可复现工程：中文执行说明

本工程包含论文 Tables II–IV 与 Figures 1–3 的实验执行、逐样本结果、聚合和
作图链路。核心方法是当前 A4：冻结 DECIDER-4B 提取三路输入及联合表示，
从头训练小解码器，直接输出上调、不变、下调概率。

## 一、安装与统一路径

在解压后的仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,train,paper]'
export S1PERT_ROOT="$PWD/workspace"
python -m experiments.paper init --gpu 0
```

所有正式数据、缓存和结果写入 `S1PERT_ROOT`；也可以逐命令传 `--work-dir`。
已有原始数据时，在初始化时加 `--data-root /path/to/raw_data`。JSON 内路径相对
该 JSON 文件解析。按 `requirements-backbone-tested.txt` 安装编码环境依赖。
STATE、vLLM、经典基线建议各用独立环境，通过 `configs/interpreters.example.json`
指定各方法的 Python。不要把不同虚拟环境的解释器替换成系统 Python。

## 二、数据下载、处理和主模型

```bash
python -m experiments.paper download --execute
python -m experiments.paper data --execute
python -m experiments.paper prepare --execute
python -m experiments.run_section table3 --execute
python -m experiments.paper main --execute
```

第一条会下载四个公开 h5ad、公共注释和通用 DECIDER 权重。权重和数据不随源码包分发。
处理使用线性 CP10k、Welch 方向检验、TOST 等效检验和联合 BH，保留未裁决原因。
类别固定为 `0=Up, 1=Same, 2=Down`，不能把“没有显著变化证据”直接当作不变。
四个细胞系分别建模，五个种子 `17/29/43/61/89` 按靶点分组做 7:1:2 划分。
这是五次随机留出，不是五折互斥交叉验证。训练证据在 TRAIN 内做五折交叉拟合。

主流程包括数据准备、三路/联合缓存、完整 epoch 训练、验证集选模、独立进程恢复、
最后 TEST 评价。A4 默认有效 batch 4096，10–100 epochs，早停 patience 10。
资源配置在 `workspace/configs/a4.json`；默认缓存空间门槛较高，需要按机器明确配置。

## 三、八个对比方法

包括 GenePT-Gene、GenePT-Prot、Qwen3-4B-LoRA、GEARS、scGPT、STATE、ALIGNED、SUMMER。
资产下载与构建见 [ASSETS.md](baselines/ASSETS.md)，具体适配见
[BASELINES.md](docs/BASELINES.md)。下载外部资产时统一加
`--root "$S1PERT_ROOT"`，避免落到另一个默认目录。

ALIGNED 已纳入实际两轮训练、独立恢复和最终概率导出源码；公共有符号图谱与 GO
分别由 `data_prep/build_public_knowledge.py` 从公开原始文件构建。
其原始核心代码哈希已核验，科学实现保留，替换的是环境路径与输入/输出接口。
Qwen 比较项保留历史小预算配置；STATE 是 ST 适配版本；SUMMER 是本地 Qwen3-8B
推理版本。这些差异有明确记录，不冒充各原论文未经修改的完整系统。

```bash
# 全新工作区可以执行全部九个方法（替代前面单独运行主模型）：
bash experiments/run_main_table2.sh --python-map configs/interpreters.example.json
bash experiments/run_main_table2.sh --python-map configs/interpreters.example.json --execute
# 若已经完成前面的主模型，只运行八个基线：
bash experiments/run_main_table2.sh --python-map configs/interpreters.example.json \
  --methods GenePT-Gene GenePT-Prot Qwen3-4B-LoRA GEARS scGPT STATE ALIGNED SUMMER \
  --registry-out "$S1PERT_ROOT/paper/baseline_predictions.csv" \
  --table-out "$S1PERT_ROOT/paper/baseline_table2" --execute
# 已有当前工作区完整预测时，仅重新登记和汇总：
bash experiments/run_main_table2.sh --evaluate-only --execute
```

Table II 默认要求九个方法×四个细胞系×五个种子齐全。每折预测按 target/query/label
逐行对齐，汇总均值与样本标准差。硬标签不能被伪造成连续概率来计算排序 AUC。
全九方法路线与“已有主模型＋八基线”路线二选一。主模型总流程不自动续训已有目录；
再次汇总请用 `--registry-out`、`--table-out` 指定新位置，保留原结果。

## 四、论文补充实验

```bash
bash experiments/run_ablation_table4.sh --execute
bash experiments/run_prior_stress_fig1.sh --execute
bash experiments/run_cell_pool_fig2.sh --execute
bash experiments/run_runtime_fig3.sh --python-map configs/interpreters.example.json --execute
```

- Table IV：K562 五个种子的五类从头训练消融，外加固定模型靶点置换诊断。
- Fig. 1：从逐样本预测重算跨靶点 AUC、先验熵、逆倾向和训练支持度。
- Fig. 2：重新划分 K562 不相交 A/B 细胞池、统计裁决与冻结模型边界评价。
- Fig. 3：实际测量冷缓存、训练和推理耗时；这一步会新训练计时用模型，较耗时。

计时通过解释器映射分别运行 A4、ALIGNED、SUMMER，避免其不同依赖版本冲突；
在线时间取各子进程内部同步后的测量区间，进程启动开销单列并计入端到端时间。

所有表图读取本次执行产生的文件；没有固定的成绩数组。历史先验熵是归一化熵的
等宽三档，不是样本三分位。历史图里的参考样本数也由数据重算，不写死。
去掉 `--execute` 只显示计划；程序不常驻监测。

## 五、交付前检查

```bash
python -m pytest -q
python scripts/audit_paper_coverage.py
python scripts/check_release.py
python scripts/package_release.py --out ../S1Pert_source.zip
```

验收范围见 `VALIDATION_REPORT.json`。CPU 合成测试验证工程逻辑与恢复，不等于重新
跑完全部正式实验。源码包排除原始数据、所有模型权重、大缓存、用户路径、凭据与
Git 历史；第三方许可证和来源保留。原作者需确定自有代码许可证后再公开发布。
完整命令、参数、资源和复现边界以英文 [README](README.md) 及对应专题文档为准。
