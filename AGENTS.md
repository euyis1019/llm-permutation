# AGENTS.md

## 适用范围

本文件适用于整个仓库。若子目录以后增加更具体的 `AGENTS.md`，以距离目标文件最近的说明为准。

## 对话约定

- 默认使用中文与用户沟通，路径、命令和代码标识符保留原文。
- 所有面向用户的过程更新、说明、总结和最终答复都不得展示数学公式、方程或推导过程，也不得用 LaTeX、纯文本或代码块变相展示。
- 需要说明数学内容时，只用自然语言概述。仓库文档可按任务需要保留或新增公式，但对话中只能给出路径和文字摘要。
- 先说明结果、风险或阻塞，再补充必要细节；不要把未经验证的推测写成实验结论。

## 仓库定位

这是一个研究 Transformer FFN 通道置换及其有限精度影响的实验仓库。仓库同时保存实验代码、冻结配置、预注册计划、机器可读结果、复审记录、报告和可执行 Notebook。整理工作的首要目标是提高可读性与可复现性，同时保持证据链不被破坏。

建议阅读顺序：

1. `README.md`：仓库入口、主要结论和复现概览；
2. `docs/reports/overall_report.md`：整体叙述；
3. `docs/reports/experiment_index.md`：逐实验索引与证据地图；
4. `docs/reports/current_findings.md`：当前结论的适用范围；
5. 目标实验目录中的计划、README、结果报告和机器可读结果。

## 目录职责

- `notebooks/`：核心阅读入口。`build_core_notebook.py` 是 Notebook 叙事与代码单元的可编辑来源，`ffn_permutation_core.ipynb` 是构建并执行后的交付物。
- `experiments/`：按研究问题拆分的代码、计划、配置、结果和复审材料。改动前先阅读对应目录的 README 与计划文档。
- `docs/reports/`：跨实验报告、阶段结论和索引。
- `docs/plans/`：跨实验设计文档；已执行计划应视为冻结记录。
- `docs/index.html`、`docs/experiment-atlas.html`、`docs/math-equivalence.html`：静态阅读页面；修改时检查相对链接和 `docs/assets/` 依赖。
- `references/`：文献综述、外部来源登记和轻量概念代码。
- `scripts/`：独立的最小推理入口，不代表完整实验环境。
- `ARTIFACTS.md`：大文件保留、忽略与重建策略的权威说明。

特殊边界：

- `experiments/ffn_benchmark_eval/bench/src/eval/infra/reference_evalplus/` 是外部参考实现。避免无关格式化、批量重命名或把本仓库逻辑混入其中。
- 外部代码与数据的来源、固定版本和许可信息应同步登记到 `references/SOURCES.md`。

## 证据保全规则

- 预注册计划、修订案、失败报告、决策记录和执行报告属于实验历史，不得为了让叙述更整齐而追溯修改。新解释应写入新的修订、复审或报告文件，并链接原记录。
- `results/`、`results_base/` 和 `reviewer_analysis/` 中的落盘数据默认是不可手工改写的证据。只有在运行对应生成或分析脚本后，才更新其派生产物。
- 保留未通过的判据、负结果、异常与勘误；不得只保留支持当前结论的结果。
- 区分预注册结论、复审后的事后分析和待复现观察。跨文档搬运结论时保留原有证据等级和适用范围。
- 不伪造运行记录、环境信息、哈希、样本量、性能数字或成功状态。没有运行的验证应明确写成“未运行”。
- 移动或重命名实验文件时，使用 `rg` 检查并更新 README、报告、Notebook 构建脚本、HTML 页面和脚本中的路径引用。
- 不提交模型权重、派生 checkpoint、二进制 logits、缓存或日志。新增大产物前先核对 `.gitignore` 与 `ARTIFACTS.md`。

## 实现与整理原则

- 优先做小而可审阅的改动；结构调整、行为修改和结果再生成尽量分开。
- 保留现有脚本的命令行入口、随机种子、默认模型、冻结配置和断点续跑语义，除非任务明确要求改变它们。
- 不要把机器相关的绝对路径静默替换成另一个机器路径。若要提高可移植性，应增加显式参数或环境变量，并保留兼容默认值与来源说明。
- 修改分析逻辑时，同时检查它读取的原始数据、写出的汇总文件和引用该汇总的报告。
- 修改 Notebook 内容时先编辑 `notebooks/build_core_notebook.py`，再重建并执行 Notebook；提交时二者应同步。
- Python 代码沿用现有的 `pathlib`、类型提示和小型脚本风格。新增公共逻辑前先搜索已有实现，尤其是多个实验目录中的 `permutation.py`。
- 不为单次整理引入大型框架、全仓格式化工具或新的运行时依赖。若确有必要，先说明迁移范围和对历史复现的影响。
- 不对数据集、JSONL、已执行 Notebook 或外部参考源做无关的批量格式化。

## 环境与运行约束

仓库目前没有统一的 `pyproject.toml`、锁文件或全局测试命令。不要假设根目录的轻量依赖足以运行 GPU 实验。

- Notebook 轻量依赖在 `requirements-notebook.txt` 中，只用于读取已有结果、重建图表和执行核心 Notebook。
- 完整实验通常依赖本机 `qwen3` Conda 环境、PyTorch、Transformers、vLLM、CUDA、指定 GPU 和本地模型。
- 本地模型通过根目录 `models` 链接或实验参数提供，不进入版本控制。
- 未经明确要求，不安装或升级 PyTorch、Transformers、vLLM、CUDA 相关包，也不运行耗时 GPU 实验。
- GPU、推理后端、数据类型或确定性配置发生变化时，旧结果不能直接当作新环境的验证结果；先运行目标实验规定的最小 smoke test。
- 某些历史脚本固定了 GPU 编号和绝对路径。执行前必须阅读相应计划与 README，不要仅凭脚本名启动。

## 验证方式

按改动范围选择最小充分验证，并在交付时说明实际运行了哪些命令。以下命令假定已经激活带 `python` 的合适环境；`pytest` 当前不在根目录依赖清单中，不要只为文档改动临时安装它。

轻量概念框架测试：

```bash
PYTHONPATH=references/minimal_perturbation_framework \
  python -m pytest -q references/minimal_perturbation_framework/tests
```

Python 语法检查可对本次修改的文件使用：

```bash
python -m py_compile path/to/changed_file.py
```

单个 JSON 文件可使用：

```bash
python -m json.tool path/to/file.json >/dev/null
```

重建并执行核心 Notebook：

```bash
python notebooks/build_core_notebook.py
jupyter nbconvert \
  --to notebook \
  --execute notebooks/ffn_permutation_core.ipynb \
  --output ffn_permutation_core.ipynb
```

完整 GPU 实验没有统一入口。以各 `experiments/*/README.md`、冻结配置和实验计划中的命令为准。只改文档或索引时，不需要为了形式完整而重跑 GPU 实验。

## 交付检查

- 查看 `git status --short`，只报告并保留与当前任务有关的改动；不要覆盖用户已有修改。
- 检查新增或移动文件的相对链接、路径引用和大小写。
- 对照 `ARTIFACTS.md`，确认没有把可重建的大文件纳入版本控制。
- 说明已运行的验证、未运行的高成本验证及原因。
- 若结论文字发生变化，指出其证据文件和证据等级是否也发生变化。
