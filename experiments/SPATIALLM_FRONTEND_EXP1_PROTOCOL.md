# SceneCOT × SpatialLM EXP1 协议：SpatialLM 作为感知前端的 MSQA 三臂评测（PoC）

> **本文档的主要读者是服务器上的 AI Agent。**
> 背景：SceneCOT（`repos/scenecot`，ICLR 2026）的 grounded CoT 推理在 MSQA 主设定下感知前端是 oracle（GT 实例分割 + GT 类别 + GT 位置，`pc_type: gt`）；代码里另有 `pc_type: pred` 通路（外部分割器 mask，即 `scene-verse-pred-all`）。本实验第一次把**我方的 SpatialLM（EXP4 best）**作为感知前端接入，回答：LLM 结构化感知前端喂 grounded CoT 推理器，端到端掉多少分？是否不弱于外部分割前端？
> 设计：**三臂单变量对照**——A（gt oracle）/ B（官方分割前端 scene-verse-pred）/ C（SpatialLM 前端），唯一差异是 pred mask 三件套（`mask.npz / score.npy / label.npy`）的来源；模型 checkpoint、subset、超参、解码、MoE 状态逐字段一致。

## 0. 已知信息（代码事实，写报告时引用行号）

| 事实 | 出处 |
|---|---|
| pred 通路只消费三件套：`{obj_feat_base}/mask/{scene}.mask.npz`（M×N 稀疏阵，N=该场景点云点数）、`.score.npy`（NMS 排序用）、`.label.npy`（ScanNet **raw label id**） | `data/datasets.py:353-360` |
| mask 定义在 `${SCENECOT_DATA_ROOT}/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment/{scene}.pth` 的**固定点序**上；`obj_locs_pred` 由 mask 内点云现算（center=均值，size=max−min），不读外部框 | `data/datasets.py:318-322, 482-490` |
| pred 物体的 2D 图像经 `gt_to_pred_id`（mask-IoU>0 最佳匹配）回到 **GT 实例的多视图投影**取图；匹配不上用零图 | `data/datasets.py:456-472, 1147-1156` |
| 物体特征文件（`image_obj_feat_pred/`、`voxel_obj_feat_pred/`）缺失时**自动 zeros，不报错** → C 臂无需生成特征文件 | `data/datasets.py:551-588`（`os.path.exists` 保护） |
| `pc_type == 'gt'` 时 pred mask 三件套**同样被加载**（条件是 `hasattr(self,'pc_type')` 而非值判断）→ scene-verse-pred-all 是三臂共同依赖，缺了 A 臂也跑不起来 | `data/datasets.py:353` |
| 官方发布 test 脚本本身就是 `data.cotqa.msr3d.pc_type=pred` + `nms_iou_threshold=1`（NMS 实际不抑制）——B 臂即官方发布设定 | `scripts/test/full_training_msqa_beacon3d_test_moe.sh` |
| **MoE 断链**：脚本设 `moe.enable=True` 但模型读的是 `cfg.grounding.moe_flag`，且 `obj_prob_dict_path` 无 yaml 定义、无发布产物 → 三臂统一**不开 MoE**（公平且绕开坑） | `model/scenecot_agent.py:222-228, 813-819` |
| 评测指标：`em_overall` / `em_refined_overall` + 分题型（counting / existence / spatial relationship / attribute / ...） | `evaluator/msqa_eval_cot.py:24-32` |
| B 臂分割器是 **Mask3D**（Schult et al. 2022；论文 §3.2/§5.1/Table 3 注明主结果用 Mask3D 提供 mask 与语义标签）；其输出特征与代码吻合（≤100 个带 score 的实例 mask 提议 + ScanNet raw id + NMS） | 论文 arXiv:2510.16714 §3.2 |
| **校准锚点**（论文 Table 3，全量 test，GPT-score）：Mask3D mask + 预测概率 = 55.6（≈ B 臂 / 论文主表设定）；完美 mask/标签 + 预测概率 = 64.9（≈ A 臂，`pc_type: gt` 正是该设定）；A−B ≈ 9pt，counting 差约 25pt。300 题子集上 A−B 严重偏离此量级 → 先查 harness | 论文 Table 3 |
| SpatialLM 前端模型：`saves/scannet_exp4/checkpoint-2500`（EXP4 best，macro@.25=0.5825，中心误差中位 0.060m，18 类词表完全收敛） | `experiments/FINETUNE_SCANNET_EXP4_REPORT.md` |

我方 18 类词表（转换脚本 §5.3 需要）：`chair, door, otherfurniture, cabinet, table, window, painting, desk, sofa, sink, bookcase, bed, curtain, toilet, refrigerator, counter, shower_curtain, bathtub`。

## 1. 实验设计

### 1.1 三臂定义

| 臂 | `pc_type` | pred mask 来源 | 说明 |
|---|---|---|---|
| A | `gt` | —（GT 实例） | oracle 上界；同时也是 harness 正确性验证 |
| B | `pred` | `scene-verse-pred-all/ScanNet/mask/`（官方发布，随数据资产下载） | 官方分割前端；隔离「pred 通路固有掉分」与「我们的转换 bug」 |
| C | `pred` | 本实验生成：`spatiallm-pred-all/ScanNet/mask/` | **主实验**；`data.obj_feat_base.ScanNet` 指向新 root |

三臂共同固定：MSQA test **前 300 题**子集、同一 pretrained checkpoint、`rng_seed=42`、`num_beams=5`、`nms_iou_threshold=1`、MoE 关闭、`cot.*` 全部同官方 test 脚本。

### 1.2 子集定义（三臂共用，保证可复现）

- 取 `${SCENECOT_COT_DATA_ROOT}/MSQA/situated_qa_test_pure_txt.json` **前 300 条**（不采样，直接截断）为子集 test split；train/val json 原样复制（数据集构建时会加载，缺文件直接崩）。
- 子集注释目录：`/media/drive1_3TB/chenle/scenecot/subset300/MSQA/`（GQA3D 同目录平级复制，见 §2.4）。
- 记录子集题型分布（`type` 字段计数）——counting/existence 不得为 0，否则换种子重新截断并报告。

## 2. 判读逻辑（写报告时必须对照此表下结论）

以三臂子集 `em_overall`（MSQACOTEvaluator 输出）为准，主判读看 **Δ(A→C)**：

| Δ(A→C) | 结论 | 下一步 |
|---|---|---|
| **≤ 5pt** | SpatialLM 前端基本可用，grounded CoT 推理器对感知误差鲁棒 | EXP2：全量 test + `use_pred_for_train` 训练侧实验 |
| **5 ~ 15pt** | 前端可用但有明显损耗 | 必须完成 §6 归因（感知缺 object / 词表粒度 / 线索质量三选一或组合），EXP2 针对性修复 |
| **> 15pt** | 先怀疑转换 bug 或感知覆盖不足 | §6.2 匹配率正常（题目锚定物体覆盖 ≥ 85%）才承认是真实差距；否则修转换重跑 C 臂 |

**必做的辅助判读**（不依赖总分档位）：
- **C vs B**（同为 pred 通路，仅前端不同）：若 C ≥ B − 2pt，则「LLM 感知前端不弱于外部分割前端」成立——这是本实验最有发表价值的结论，即使 C 掉分也在中档也要回答
- **分题型掉分结构**：counting/existence 依赖 grounding 候选完整性；spatial relationship 依赖线索坐标质量——两族掉分模式不同指向不同归因
- **重叠 caveat**（§2.5）：MSQA test 场景与 SpatialLM 训练集（1201 场景）的重叠会**高估** C 臂感知质量，报告中必须给出重叠率并（若 > 20%）补一版仅含非重叠场景的分题型数字

## 3. 前置检查（Stage 0；任一失败 → 记录后停止并报告用户，不要自行找替代数据源）

### 3.1 代码与环境

```bash
# SceneCOT 代码（服务器若无则从 GitHub 克隆；README 的 env-var 化配置已在发布版里）
ls ~/SpatialLM/repos/scenecot/launch.py   # 不存在则: git clone https://github.com/SceneCOT/scenecot ~/SpatialLM/repos/scenecot

# 环境（python 3.9 + torch 2.4.1 cu118，README §Get Started 逐条执行）
conda create -n scenecot python=3.9 -y && conda activate scenecot
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
cd ~/SpatialLM/repos/scenecot && pip install -r requirements.txt && pip install spconv-cu118
cd model/pointnetpp && python setup.py install && cd ../..
python -c 'from model.pointnetpp.pointnetpp import PointNetPP'   # 预期无报错
```

### 3.2 数据资产 inventory

统一放 `/media/drive1_3TB/chenle/scenecot/data_assets`（下称 `$ASSETS`，即 `SCENECOT_DATA_ROOT`）。下载统一用 hf-download skill（HF 源：`EricLHK/SceneCOT` dataset repo，布局见其 README §Data preparation）。

```bash
export ASSETS=/media/drive1_3TB/chenle/scenecot/data_assets
# 逐项检查（最小集合：仅 MSQA test 子集涉及的 ScanNet 场景也需要整目录结构就位）
ls $ASSETS/scenecot_cot_data/MSQA/situated_qa_test_pure_txt.json          # 注释
ls $ASSETS/scenecot_cot_data/GQA3D/gqa3d_test.json                        # 注释（fallback 用，见 §2.4）
ls $ASSETS/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment/ | head # 场景点云 .pth
ls $ASSETS/scan_family/annotations/meta_data/scannetv2-labels.combined.tsv
ls $ASSETS/scene-verse-pred-all/ScanNet/mask/ | head                      # 官方 pred 三件套（三臂共同依赖，见 §0）
ls $ASSETS/scenecot_imgs/imgs/scannet/ | head                             # 物体图像
ls $ASSETS/LEO-2_feature/ScanNet/ | head                                  # obj 2D 特征
```

**失败处理**：哪项缺就停在哪项，向用户报告缺失清单与 HF repo 中对应路径，不要用别的数据集版本顶替（点序对齐依赖 SceneVerse 原版 pth）。

### 3.3 模型资产与 checkpoint 探针（重要：主模型 checkpoint 在发布物里有歧义）

```bash
export HF_HOME=/home/chenle/hf_home_spatiallm
export MODEL_ROOT=/media/drive1_3TB/chenle/scenecot/model_assets
# 从 EricLHK/SceneCOT 下载: pointnet_tokenizer.pth, query3d_pretrain.bin, expert1_checkpoint0/, expert2_best.pth/
# 从 HF 下载: liuhaotian/llava-v1.5-7b, openai/clip-vit-large-patch14-336, openai/clip-vit-large-patch14
ls $MODEL_ROOT
```

发布 README 只把 `expert1_checkpoint0` / `expert2_best.pth` 标注为 MOE expert，**主模型 checkpoint 是哪一个没有明说**。处理：
1. 先 inventory HF repo 根目录，若有名字对应训练脚本 `note=main_msqa_gqa3d` 的目录/文件则直接用它；
2. 否则候选 = `expert1_checkpoint0`（其训练脚本名即 `full_training_msqa_gqa3d`，最可能是主模型）与 `expert2_best.pth`，用 **20 题微子集**（§3.4 的前 20 条）各跑一次 A 臂，选输出 CoT 结构完整（含 `<answer>`、无全空）且 em 更高者，**结论写进报告**；
3. 两候选均不完整 → 停止并报告用户。

### 3.4 子集构建

```bash
python - <<'EOF'
import json, os, collections
src = "/media/drive1_3TB/chenle/scenecot/data_assets/scenecot_cot_data/MSQA"
dst = "/media/drive1_3TB/chenle/scenecot/subset300/MSQA"
os.makedirs(dst, exist_ok=True)
for split in ["train", "val"]:
    os.system(f"cp {src}/situated_qa_{split}_pure_txt.json {dst}/")
data = json.load(open(f"{src}/situated_qa_test_pure_txt.json"))
sub = data[:300]
json.dump(sub, open(f"{dst}/situated_qa_test_pure_txt.json", "w"))
scenes = sorted({d["scene_id"] for d in sub})
open("/media/drive1_3TB/chenle/scenecot/subset300/scenes.txt", "w").write("\n".join(scenes))
print("questions:", len(sub), "scenes:", len(scenes))
print("types:", collections.Counter(d["type"] for d in sub))
EOF
# 同步 GQA3D（防 mode=[] 覆盖失败时缺文件）：整目录复制，test 截断到 50 条以兜底运行时间
```

**预期**：300 题、约 100~250 场景、counting/existence 均非 0。GQA3D 场景不需要，因为我们尽力把它从评测里排除（§4 命令里的 `task.qacot.QACOTScanNetGQA3D.mode=[]`）。

### 3.5 场景重叠统计（判读 caveat 的输入）

```bash
python - <<'EOF'
import json
slm_train = set()
for it in json.load(open("/home/chenle/SpatialLM/data/scannet_spatiallm/scannet_train.json")):
    slm_train.add(it.get("scene_id") or it.get("scan_id"))   # 按实际字段名调整
scenes = set(open("/media/drive1_3TB/chenle/scenecot/subset300/scenes.txt").read().split())
print("overlap:", len(scenes & slm_train), "/", len(scenes))
EOF
```

## 4. Stage 1：Arm A（gt oracle）——同时是 harness 验证

```bash
conda activate scenecot && cd ~/SpatialLM/repos/scenecot
export SCENECOT_EXP_ROOT=/media/drive1_3TB/chenle/scenecot/outputs
export SCENECOT_DATA_ROOT=/media/drive1_3TB/chenle/scenecot/data_assets
export SCENECOT_MSR3D_ANNO_DIR=/media/drive1_3TB/chenle/scenecot/subset300/MSQA
export SCENECOT_MODEL_ROOT=/media/drive1_3TB/chenle/scenecot/model_assets
export HF_HOME=/home/chenle/hf_home_spatiallm WANDB_MODE=disabled
export CKPT=$SCENECOT_MODEL_ROOT/expert1_checkpoint0   # §3.3 探针结论替换

CUDA_VISIBLE_DEVICES=0 python launch.py --name scene_cot --mode python \
  --config configs/default_cot.yaml --port 2021 --gpu_per_node 1 --num_nodes 1 \
  model=SceneCOTAgent task=scenecot_scanent_msqa_gqa3d note=exp1_armA_gt \
  grounding.enable=True grounding.loss_type=ce grounding.grd_loss_weight=0.1 \
  grounding.grd_text_hidden_states=average_embedding grounding.use_region_mask=True \
  cot.mask_obj_prob_loc_token=True cot.use_oracle_obj_content=False cot.cot_no_scene_tokens=True \
  cot.msr3d_val_set_size=1000 \
  llm=llava1.5-7b llm.max_out_len=512 llm.max_context_len=1024 \
  vision3d.name=PQ3D vision3d.use_embodied_token=False \
  task.qacot.QACOTScanNetMSR3D.evaluator=MSQACOTEvaluator \
  task.qacot.QACOTScanNetGQA3D.mode=[] \
  data.cotqa.msr3d.pc_type=gt data.cotqa.gqa3d.pc_type=gt \
  data.nms_iou_threshold=1 data.cotqa.use_pred_for_train=False \
  data.cotqa.msr3d.anno_dir=$SCENECOT_MSR3D_ANNO_DIR \
  pretrained_ckpt_path=$CKPT mode=test 2>&1 | tee /media/drive1_3TB/chenle/scenecot/logs/exp1_armA.log
```

- **预期**：单卡 30~90 分钟；结束日志含 `em_overall`；结果落 `$SCENECOT_EXP_ROOT/scene_cot/exp1_armA_gt/.../eval_results/`（以日志实际路径为准，供离线复核 `evaluator/msqa_evaluator_offline.py`）。
- **失败处理**：
  - `task.qacot.QACOTScanNetGQA3D.mode=[]` 报 hydra 错 → 去掉该 override 重跑（此时 gqa3d test 也会跑，用 §3.4 截断的 50 条版兜底时间；gqa3d 数字不进报告）
  - 首个 batch 就崩在 mask 加载 → 对照 §0「gt 模式也加载 pred 三件套」，确认 scene-verse-pred-all 完整
  - OOM → `dataloader.eval.batchsize=1`
- **A 臂 sanity 门**：`em_overall` 应明显高于瞎猜且输出含可解析 `<answer>`；若接近 0，先修 harness（对照 §3.3 探针记录），**不得进入 B/C 臂**。

## 5. Stage 2 & 3：Arm B（官方分割前端）与 Arm C（SpatialLM 前端）

### 5.1 Arm B：只改两处

复制 §4 命令，改：`note=exp1_armB_svpred`、`data.cotqa.msr3d.pc_type=pred`，日志 `exp1_armB.log`。其余不动（B 臂即官方发布设定，其 mask 已在数据资产里）。

### 5.2 Arm C 第一步：点云导出 + SpatialLM 推理

```bash
conda activate spatiallm && cd ~/SpatialLM
# 导出 subset300 场景为 .ply（SpatialLM 输入；colors 反归一化 ×127.5+1 → uint8）
python - <<'EOF'
import torch, numpy as np, os
from pathlib import Path
src = "/media/drive1_3TB/chenle/scenecot/data_assets/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment"
out = "/media/drive1_3TB/chenle/scenecot/spatiallm_in"; os.makedirs(out, exist_ok=True)
for sid in open("/media/drive1_3TB/chenle/scenecot/subset300/scenes.txt").read().split():
    pts, cols, _ = torch.load(f"{src}/{sid}.pth", weights_only=False)[:3][0], None, None
    d = torch.load(f"{src}/{sid}.pth", weights_only=False)
    points, colors = d[0].numpy(), (d[1].numpy() * 127.5 + 1).clip(0, 255).astype(np.uint8)
    np.save(f"{out}/{sid}.npy", np.concatenate([points, colors], 1))   # 或写 .ply，与 inference.py 支持格式一致
EOF
# SpatialLM 推理（EXP4 best；照 EXP4 §4 分片做法，4 卡）
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g python inference.py -d object \
  -p <按场景分片目录> -o /media/drive1_3TB/chenle/scenecot/spatiallm_preds_g$g \
  --model_path saves/scannet_exp4/checkpoint-2500 --seed 42 > /media/drive1_3TB/chenle/scenecot/logs/infer_slm_g$g.log 2>&1 & done; wait
```

**预期**：每场景一个含 `Bbox(class, x, y, z, angle_z, sx, sy, sz)` 行的 txt；总场景数 = `scenes.txt` 行数。

### 5.3 Arm C 第二步：转换脚本（新写 `scripts/spatiallm_to_scenecot_masks.py`，放 SceneCOT repo）

I/O 契约（写脚本时逐条落实，全部为硬 assert）：

1. 输入：`--scenes`（scenes.txt）、`--slm_preds`（5.2 输出目录，文件名 `{scene_id}.txt`）、`--sceneverse_pcd_dir`、`--label_tsv`（scan_family 的 `scannetv2-labels.combined.tsv`）、`--out_dir`。
2. 每场景：载入 pth 得 points(N,3) → 解析 Bbox 行 → **point-in-OBB**（点平移到框中心、按 `-angle_z` 旋回、`|dx|≤sx/2` 三轴）→ 二值 mask(N,)。
3. 过滤：<5 点的 mask 丢弃；**mask 数截到 ≤100**（loader 只读前 100 行，`datasets.py:356`）；按 mask 点数降序排列行序。
4. `label.npy`：18 类名 → ScanNet raw name（`shower_curtain→shower curtain`；`bookcase→bookshelf`；`otherfurniture` 无同名 raw 类，用 tsv 中 benchmark 列映射到任一代表 raw id，**写脚本时打印 18 类最终映射表进日志并 assert 全覆盖**）→ `LabelConverter` raw_name→raw_id。
5. `score.npy`：全 1.0（官方 `nms_iou_threshold=1` 下 NMS 不抑制，排序无影响）。
6. 输出 `spatiallm-pred-all/ScanNet/mask/{scene}.mask.npz`（**sparse csr，shape (M,N)，与 pth 点数 N 严格相等 assert**）、`.score.npy`、`.label.npy`。
7. 附带打印每场景：pred 数、点覆盖数——供 §6 用。

```bash
# Arm C 第三步：特征软链（gt 特征保持可得，pred 特征缺失自动 zeros，见 §0）+ 评测
ln -s /media/drive1_3TB/chenle/scenecot/data_assets/scene-verse-pred-all/ScanNet/image_obj_feat_gt  /media/drive1_3TB/chenle/scenecot/spatiallm-pred-all/ScanNet/
ln -s /media/drive1_3TB/chenle/scenecot/data_assets/scene-verse-pred-all/ScanNet/voxel_obj_feat_gt   /media/drive1_3TB/chenle/scenecot/spatiallm-pred-all/ScanNet/
# 评测命令 = §5.1 基础上再改两处: note=exp1_armC_spatiallm, data.obj_feat_base.ScanNet=/media/drive1_3TB/chenle/scenecot/spatiallm-pred-all/ScanNet
```

**C 臂已知 caveat（写报告）**：pred 物体 CLIP 特征为 zeros（B 臂有真实文件）；若 PQ3D+grounding 路径确实不消费 `obj_fts_img_pred`（`scenecot_agent.py:801-806` 走 `scene_tokens`）则无影响——报告里给出验证方法与结论（如打印一个 batch 中该张量是否进入 forward）。

## 6. 感知质量辅助分析（判读钥匙，不依赖三臂跑分）

### 6.1 对象级指标（脚本可并入 5.3）

对 subset300 全部场景，B/C 两臂各算：

| 指标 | 定义 | 用途 |
|---|---|---|
| GT 覆盖率 | 存在 IoU≥0.25 pred mask 的 GT 实例占比 | 感知 recall |
| **题目锚定覆盖率** | 注释 `obj_ids` 中被 IoU≥0.25 pred 覆盖的比例 | **与 QA 掉分最直接相关**（锚定物体丢了 grounding 候选必然错） |
| 类别一致率 | 匹配对中 pred label == GT label 占比 | 词表归因 |
| 平均物体数 | pred 数 vs GT 数 | 欠/过分割印象 |

### 6.2 归因规则（报告必答）

- 题目锚定覆盖率高（≥85%）但 QA 仍掉分 → 掉分在**线索质量/词表粒度**，不在漏检；
- 锚定覆盖率低（<85%）→ 掉分主要是**感知漏检**，counting/existence 型应显著更差（分题型表验证）。

## 7. 报告要求

写 `repos/scenecot/experiments/SPATIALLM_FRONTEND_EXP1_REPORT.md`（风格沿用 `experiments/FINETUNE_SCANNET_EXP*_REPORT.md`），**必含**：

1. **三臂汇总表**：em_overall / em_refined_overall + 分题型（至少 counting / existence / spatial relationship）+ 每臂运行时长
2. **§2 三档判读**结论 + C vs B 辅助判读结论
3. **感知质量表**（§6.1 四指标 × B/C 两臂）+ §6.2 归因结论
4. **checkpoint 探针结论**（§3.3：最终用哪个、判据）
5. **重叠 caveat**：§3.5 重叠率；若 >20%，附仅非重叠场景的分题型对比
6. **C 臂 zeros 特征验证**结论（§5.3 caveat）
7. Caveat 汇总（MoE 关闭、nms_iou_threshold=1、子集 300 题非官方全量、GQA3D 处理方式）
8. 复现命令与产物清单

完成后向用户汇报并**停止**（不要自行发起 EXP2）。

## 8. 产物清单约定

| 路径（`/media/drive1_3TB/chenle/scenecot/` 下） | 内容 |
|---|---|
| `subset300/` | 子集注释（MSQA + GQA3D）+ scenes.txt |
| `data_assets/`、`model_assets/` | 下载的数据与模型资产（§3.2/3.3） |
| `spatiallm_in/`、`spatiallm_preds_g*/` | SceneVerse 点云导出 + SpatialLM 推理输出 |
| `spatiallm-pred-all/ScanNet/mask/` | 转换后三件套（C 臂前端） |
| `outputs/scene_cot/exp1_arm{A,B,C}_*/` | 三臂评测输出（eval_results + results.json） |
| `logs/` | 全部 tee 日志 |
| `repos/scenecot/scripts/spatiallm_to_scenecot_masks.py` | 转换脚本（随报告一并交付） |
| `repos/scenecot/experiments/SPATIALLM_FRONTEND_EXP1_REPORT.md` | 实验报告 |
