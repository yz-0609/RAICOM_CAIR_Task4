# RAICOM CAIR 国赛任务四

UGOT 机器人的任务四工程，包含智能仓储、智能陪伴、智能侦察、智能解算四个场景。入口 `task4.py` 接收现场语音，调用云端模型生成五步计划，经本地校验后执行。

## 安装

在 Windows PowerShell 中进入本目录，使用 Python 3.14：

```powershell
uv venv --python 3.14 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
Copy-Item .env.example .env
```

在本地 `.env` 填写 `TASK4_LLM_*` 云端模型配置，按机器人主控屏幕地址修改场景配置中的 `robot_ip`。`.env`、虚拟环境和日志不会提交到 Git。

## 现场语音运行

每次仅运行要测试的一个场景，先将机器人与物料复位到该场景已验证的起点：

```powershell
# 智能仓储
.\.venv\Scripts\python.exe task4.py --config .\tmp\warehouse_trial.json

# 智能陪伴
.\.venv\Scripts\python.exe task4.py

# 智能侦察
.\.venv\Scripts\python.exe task4.py --config .\tmp\recon_site_verified.json

# 智能解算
.\.venv\Scripts\python.exe task4.py --config .\tmp\math_site_verified.json
```

固定计划复测可加 `--plan-json .\samples\warehouse.json` 等计划文件；`--dry-run` 只检查规划，不执行动作。运行日志写入 `logs/`。

仓储按队伍方案主动放弃“放到 Y 号位置”得分点，抓取后用 51 号舵机在原地释放色块。追色、夹取和 Tag 0 追踪直接调用任务一的原函数。程序优先只读加载相邻的任务一工程；独立克隆本仓库时使用 `vendor/task1/` 内的未改动副本。其余场景不使用该副本。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

现场标定与任务说明见 `CALIBRATION.md`、`INTELLIGENT_WAREHOUSE.md` 和 `INTELLIGENT_MATH_FINAL.md`。
