# YOLO 方法：Bring Your Own Model（BYOM）

本项目**不随仓库提供 YOLO 权重**。  
YOLO 只是一种可选的“变化定位”方法：代码已经写好，但模型需要你自己训练。

> 一句话：**方法实现看代码，模型权重靠你训练。**  
> 不配置权重时，YOLO 会自动变成“未配置（可选）”，其余六种方法照常工作。

---

## 1. 包里有什么，没有什么

| 内容 | 是否包含 | 说明 |
| --- | --- | --- |
| `core/methods/yolo.py` | ✅ | 加载权重、推理、按类别匹配、元素内像素差 |
| 方法元信息 / 前端卡片 | ✅ | 未配置时会显示“需自备权重” |
| 训练接入文档 | ✅ | 本文件 |
| `best.pt` / `last.pt` | ❌ | 训练产物，体积大且涉及数据隐私 |
| 微信截图 / 标注 / 数据集 | ❌ | 属于个人数据，不应进仓库 |
| `runs/`、`*.onnx` | ❌ | 训练中间产物 |

YOLO 在项目里只负责一件事：

```text
两张图 → 检测 UI 元素 → 同类元素按 IoU 配对 → 比较同一元素裁剪内的像素差
```

它不负责判断“新增还是消失”；这个语义仍然交给 HSV 或人工。

---

## 2. 快速开始

### 2.1 安装可选依赖

YOLO 依赖 Ultralytics（**AGPL-3.0**，见第 6 节）：

```bash
pip install -r requirements-yolo.txt
```

如果你只关心前六种方法，可以完全不装。

### 2.2 准备数据集

推荐 YOLO 标准目录：

```text
dataset/
├─ images/
│  ├─ train/
│  │  ├─ 001.png
│  │  └─ ...
│  └─ val/
│     ├─ 101.png
│     └─ ...
├─ labels/
│  ├─ train/
│  │  ├─ 001.txt
│  │  └─ ...
│  └─ val/
│     ├─ 101.txt
│     └─ ...
└─ data.yaml
```

标注文件是 YOLO 格式，每行：

```text
class_id cx cy w h
```

坐标都是相对图片宽高的 0–1 浮点数。

`data.yaml` 示例（类别可以按你的界面自己改）：

```yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: session_item
  1: nav_tab
  2: search_entry
  3: chat_title
  4: msg_bubble_other
  5: msg_bubble_self
  6: msg_input
  7: send_button
```

本项目代码**不写死类别名**：它从 `model.names` 动态读取。
你用自己的 UI 类别训练也可以，只要推理时模型能返回类别和框。

### 2.3 训练

以 Ultralytics 为例：

```bash
yolo detect train \
  data=dataset/data.yaml \
  model=yolo11n.pt \
  imgsz=960 \
  epochs=100 \
  batch=8
```

Windows PowerShell 一行写法：

```powershell
yolo detect train data=dataset/data.yaml model=yolo11n.pt imgsz=960 epochs=100 batch=8
```

训练完成后，权重通常在：

```text
runs/detect/train/weights/best.pt
```

### 2.4 放到项目里

任选一种方式：

```text
image-diff-lab/
└─ models/
   └─ wechat-ui-best.pt
```

```bash
# 方式 A：环境变量
export YOLO_WEIGHTS=models/wechat-ui-best.pt     # Linux/macOS
$env:YOLO_WEIGHTS = "models\wechat-ui-best.pt"   # Windows PowerShell

# 方式 B：命令行参数
python server.py --weights models/wechat-ui-best.pt
```

也可以复制 `.env.example` 为 `.env`：

```ini
YOLO_WEIGHTS=models/wechat-ui-best.pt
```

服务启动时会按以下顺序查找权重：

```text
--weights
  → 环境变量 / .env 的 YOLO_WEIGHTS
  → models/*.pt
  → 兼容旧路径 ../wechat-vision/training/runs/*/weights/best.pt
```

---

## 3. 如何确认配置成功

启动服务后访问：

```http
GET http://127.0.0.1:8791/api/meta
```

查看 `methods` 里 `key = "yolo"` 的项：

```json
{
  "key": "yolo",
  "available": true,
  "unavailableReason": null,
  "status": {
    "ready": true,
    "available": true,
    "model": "wechat-ui-best.pt",
    "device": "0",
    "hint": "docs/yolo-byom.md"
  }
}
```

也可以在页面右上角看状态：

```text
模型就绪
```

未配置时会显示：

```text
模型就绪 · YOLO 可选
```

YOLO 卡片会显示“需自备权重”，且不会被默认勾选。

---

## 4. 未配置权重时会发生什么

这是设计好的行为，不是报错：

- 前端方法列表里 YOLO 置灰，标注“可选”；
- 默认只运行其他六种方法；
- 综合投票不把 YOLO 算进去；
- 如果 API 显式请求 `methods: ["yolo"]`，会返回一条中性结果：

```json
{
  "key": "yolo",
  "ok": false,
  "error": "YOLO 权重未配置（BYOM）",
  "details": {
    "unavailable": true,
    "reason": "未配置 YOLO 权重（BYOM）……",
    "docs": "docs/yolo-byom.md"
  }
}
```

它不会让整个请求失败，也不会影响其他方法。

---

## 5. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 页面 YOLO 置灰 | 没找到权重 | 设置 `YOLO_WEIGHTS` 或 `--weights` |
| `未安装 ultralytics` | 只装了基础依赖 | `pip install -r requirements-yolo.txt` |
| 有权重但检测不到目标 | 模型类别/界面不适配 | 用自己的界面数据重新训练 |
| 红点变化 YOLO 没报 | 红点不在任何检测框内 | 增加类别/标注，或改用 HSV |
| GPU 没生效 | CUDA 环境不匹配 | 检查 `torch.cuda.is_available()` |
| 启动很慢 | YOLO/OCR 首次加载 | 正常；可加 `--no-warmup` 跳过预热 |
| 权重文件很大 | `best.pt` 含模型参数 | 不要提交到 Git；用 Release 或手动放置 |

---

## 6. 许可证：Ultralytics 是 AGPL-3.0

Ultralytics 的代码和官方模型默认使用 **AGPL-3.0**。  
这意味着如果你分发一个使用它的项目，通常需要按 AGPL 提供源代码，或购买 Ultralytics 商业许可证。

本项目的处理方式：

- YOLO 放在**可选依赖** `requirements-yolo.txt`；
- 基础依赖 `requirements.txt` 不包含 Ultralytics；
- 本仓库不包含 Ultralytics 的权重；
- 使用者自行安装、训练和部署时，自行遵守 AGPL。

> 这不是法律意见。如果用于商业闭源产品，请先确认 Ultralytics 的授权方式。

如果你希望整个项目使用 MIT / Apache-2.0，建议不要默认集成 YOLO，
而是把 `core/methods/yolo.py` 作为可选插件发布。

---

## 7. 隐私

- 不要提交真实微信截图、联系人名、头像、消息内容；
- 不要提交用私人数据训练出来的权重，除非你确认有权公开；
- 示例图请使用合成的假界面（假名字、纯色头像）；
- 训练数据、`runs/`、`*.pt`、`.env` 都已在 `.gitignore` 里排除。

---

## 8. 方法实现细节（给二次开发者）

`core/methods/yolo.py` 的关键步骤：

1. `discover_weight()`：按优先级找权重；
2. `_load()`：懒加载 `ultralytics.YOLO`，自动选 CUDA/CPU，并做一次空跑预热；
3. `_predict()`：对两张图分别检测，输出 `cls / score / bbox`；
4. 同类别目标按 IoU ≥ 0.3 贪心配对；
5. 对每一对目标，裁剪前后图同一区域，计算灰度绝对差；
6. 变化像素比例超过阈值 → 生成 `Region`；
7. 只返回变化目标作为“证据区域”，未变化目标仅画在叠加层里作参考。

方法本身**不依赖特定类别名**，所以你训练自己的 UI 模型也可以直接接入。
