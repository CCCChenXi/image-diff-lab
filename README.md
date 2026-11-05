# 图像变化检测实验台（image-diff-lab）

一个独立小实验：**输入两张微信截图，一个接口调用七种方法，输出统一判定、证据与 AOP 实测耗时。**

场景就是截图里的问题：第一张联系人旁边有红点，第二张红点消失——程序要判断
**是新增、消失、位置变化，还是没变**；同时观察七种方法各自“看得见什么、看不见什么、要花多久”。

> 与 `yolo-lab` 一样：**前端零构建、单用户、只监听 127.0.0.1**。
> 图像能力可以直接复用已有的 Python 环境，也可以按下面的通用方式新建环境。
>
> 实验目的、方法对比、观察耗时与结论见 [`docs/实验报告.md`](docs/实验报告.md)。
> YOLO 权重不随仓库提供，接入方式见 [`docs/yolo-byom.md`](docs/yolo-byom.md)。

---

## 安装

```powershell
cd image-diff-lab

python -m venv .venv
.venv\Scripts\activate

# 核心 + OCR（本实验完整跑前六种方法至少需要 OCR）
pip install -r requirements-ocr.txt

# 可选：YOLO（⚠️ Ultralytics 是 AGPL-3.0；权重需自行训练）
pip install -r requirements-yolo.txt
```

> 本机已有可用 venv 时，也可以直接：
>
> ```powershell
> <your-venv>\Scripts\python.exe server.py --port 8791
> ```

## 跑起来

```powershell
python server.py --port 8791

# 浏览器
http://127.0.0.1:8791/
```

| 参数 | 说明 |
| --- | --- |
| `--port` | 默认 `8791`（避开 yolo-lab 的 8790） |
| `--host` | 默认 `127.0.0.1`，不要改成 `0.0.0.0` 暴露公网 |
| `--weights` | YOLO 的 `best.pt`；不传时依次读 `YOLO_WEIGHTS`、`models/*.pt`（BYOM，权重不随仓库提供） |
| `--no-warmup` | 启动时不开后台预热；第一次 OCR/YOLO 会把模型加载算进方法耗时 |

启动后会在后台预热 OCR / YOLO。页面右上角：

- 有权重：`模型预热中 x/2` → `模型就绪`；
- 没权重：`模型就绪 · YOLO 可选`，YOLO 方法置灰，不参与默认对比。

**OCR 首次初始化约 10–20 秒**，预热完成后每张图约 8–12 秒（CPU）；YOLO 有 CUDA 时几十毫秒。

## YOLO 权重（BYOM）

仓库**不上传** `best.pt`、训练集和真实微信截图。YOLO 是可选方法：

```powershell
# 方式一：环境变量 / .env
$env:YOLO_WEIGHTS = "models\wechat-ui-best.pt"

# 方式二：命令行
python server.py --weights models/wechat-ui-best.pt
```

未配置时：

- 前端 YOLO 卡片显示“需自备权重”，默认不勾选；
- 其他六种方法正常运行；
- 综合投票不会把 YOLO 算进去。

自己训练、数据格式、类别约定、许可证和排错，见 [`docs/yolo-byom.md`](docs/yolo-byom.md)。

内置示例图对是由 `tools/make_samples.py` 从本机真实截图生成的红点新增 / 消失 / 移位 / 无变化四组。
**这些截图属于个人数据，`.gitignore` 默认不提交 `web/samples/*.png`**；公开仓库请改用合成的假界面示例：

```powershell
python tools\make_samples.py --force
```

---

## 架构

```
浏览器（零构建 HTML/CSS/JS）
   │  POST /api/compare
   │  { before: dataURL, after: dataURL, methods: ["pixel_diff", ...] }
   ▼
server.py（标准库 ThreadingHTTPServer）
   ▼
CompareService.compare                 ← AOP 切面：serviceMs / totalMs
   ├─ decode_data_url ×2                imdecode（兼容中文路径）
   ├─ align_pair                       长边 ≤1600；后图缩放到前图尺寸
   ├─ for method in selected:          ← AOP pointcut: execution(*.detect(..))
   │     pixel_diff.detect             → MethodResult
   │     red_hsv.detect                → MethodResult
   │     ssim.detect                   → MethodResult
   │     ocr.detect                    → MethodResult
   │     yolo.detect                   → MethodResult
   │     gray_fingerprint.detect       → MethodResult
   │     orb_align.detect              → MethodResult
   ├─ build_consensus                  多数投票（只做方法间一致度，不冒充准确率）
   └─ timing.aspect.records            本次 trace 的全部切面记录
   ▼
JSON：consensus + methods[] + images + timing
```

### 七种方法

| key | 方法 | 回答什么 | 关键实现 | 典型耗时 |
| --- | --- | --- | --- | --- |
| `pixel_diff` | OpenCV 图像差异 | 哪些区域变了 | 灰度绝对差 → 阈值 22 → 形态学 → 连通域 | 20–60 ms |
| `red_hsv` | HSV 红色检测 | 红点有没有、几个、在哪、新增/消失/移位 | HSV 双区间红掩码 → 圆形小块过滤 → 色块配对 | 25–40 ms |
| `ssim` | SSIM 结构相似度 | 整体结构变了多少 | 11×11 高斯窗口 SSIM，阈值 0.995 | 100–140 ms |
| `ocr` | OCR 文字识别 | 文字内容变了吗 | PaddleOCR PP-OCRv5 mobile → 精确 + 同位置模糊匹配 | 8–12 s/图（CPU） |
| `yolo` | YOLO 目标检测 | 变化落在哪个 UI 元素里 | 自定义 UI 模型框元素 → 元素内像素差；**权重需自备（BYOM）** | 30–60 ms（GPU）/ 数百 ms（CPU） |
| `gray_fingerprint` | 灰度指纹差分 | 还是不是同一张画面 | dHash / aHash / pHash 感知哈希 + 16×16 分块灰度差 | 20–45 ms |
| `orb_align` | ORB 特征对齐差分 | 是否只是整体平移/滚动 | ORB + BFMatcher + RANSAC 单应性，对齐后做残差 | 40–150 ms |

**这四个内置示例上的预期表现**（也是 `tools/selftest.py` 的断言）：

| 图对 | 像素差分 | HSV | SSIM | OCR | YOLO | 灰度指纹 | ORB 对齐 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 红点新增 | 变化（content） | **新增** | 无变化 | 无变化 | 变化（content） | 无变化 | 变化（content） |
| 红点消失 | 变化（content） | **消失** | 无变化 | 无变化 | 变化（content） | 无变化 | 变化（content） |
| 红点移位 | 变化（content） | **位置变化** | 无变化 | 无变化 | 变化（content） | 无变化 | 变化（content） |
| 无变化 | 无变化 | 无变化 | 无变化 | 无变化 | 无变化 | 无变化 | 无变化 |
| 整图平移 25px | 大面积误报 | 红点不变则无变化 | 可能变化 | 无变化 | 元素框仍能匹配 | 可能变化 | **对齐后判无内容变化** |

这张表就是实验结论的雏形：

- **像素差分 / HSV** 是红点场景的主力。差分定位强但不知道语义；HSV 能直接说出新增/消失/移位。
- **SSIM / 灰度指纹** 都是全局方法：一个看结构相似度，一个看感知哈希。
  它们能把“整页换图 / 滚动”快速筛出来，但一个 100 多像素的红点会被缩图或平均抹掉。
- **OCR** 只对联系人名、消息、时间等文本变化敏感；红点不是文字，它不会报。
- **YOLO** 的价值是把变化缩到“哪一个会话行”，但红点不是独立类别，仍需颜色/人工判断类型。
- **ORB 对齐** 解决像素差分的另一个大坑：两张截图窗口挪动/页面滚动时，
  它会先估计单应性，把后图对齐到前图，再比较残差；纯位移不会再被整屏误报。

### 还可以考虑的方法（未内置）

| 方法 | 适合什么 | 为什么这次没默认加 |
| --- | --- | --- |
| Lab ΔE 颜色差分 | 亮度接近、只改颜色的状态（红→绿、置灰/高亮） | 与 HSV / 像素差分有重叠，会让“红点”实验的方法数量继续膨胀 |
| 模板匹配 / NCC | 判断某个图标/按钮是不是“移动了”，而不只是“变了” | 需要先选模板；HSV 已覆盖红点移位，通用元素移位可后续加 |
| 光流（Farneback） | 滚动、动画、连续帧之间的位移场 | 单张截图对之间意义有限，更适合录屏/连续帧 |
| 直方图 / 颜色分布 | 整体色调、主题切换、滤镜变化 | 全局统计，定位能力弱，和指纹/SSIM 定位重复 |
| Siamese U-Net / ChangeFormer | 像素级变化分割，精度上限最高 | 需要训练数据和模型权重，超出“七种实现类对比”的实验范围 |

如果只再加一种，我会优先补 **Lab ΔE 颜色差分**：它能抓住“亮度几乎一样但颜色变了”的情况，
和现有灰度差分形成互补。

---

## 统一输出契约

七种实现类都返回同一个 `MethodResult`（`core/models.py`）：

```ts
interface MethodResult {
  key: string;                 // pixel_diff | red_hsv | ssim | ocr | yolo | gray_fingerprint | orb_align
  name: string;
  ok: boolean;
  error: string | null;
  elapsedMs: number;           // ← AOP 切面在方法返回时写入，业务代码不计时

  hasChange: boolean;          // 是否发生变化
  changeType: "none" | "added" | "removed" | "moved" | "content" | "unknown";
  changeTypeLabel: string;     // 新增 / 消失 / 位置变化 / 内容变化 / 无变化
  confidence: number;          // 0–1，方法对自身判定的置信度（不是数据集准确率）
  summary: string;             // 一句人话结论

  regions: Array<{             // 证据框：对齐后画布的像素坐标
    x: number; y: number; w: number; h: number;
    label: string;
    kind: "added" | "removed" | "moved" | "changed" | "object";
    score?: number;
  }>;

  metrics: Array<{             // 给页面直接渲染的指标
    key: string; label: string; value: string | number;
    unit: string; hint: string;
  }>;

  details: Record<string, unknown>; // 原始值：阈值、红点列表、OCR 行、YOLO 框……
  overlay: string | null;            // JPEG data URL，证据已画在图上
}

interface CompareResponse {
  ok: true;
  requestId: string;           // 同时是本次 traceId
  startedAt: string;
  images: {
    width: number; height: number;       // 对齐后的画布
    aligned: boolean; warning: string | null;
    before: { originalWidth; originalHeight; scale };
    after:  { originalWidth; originalHeight; scale; resizedToBefore };
  };
  methods: MethodResult[];
  consensus: {
    hasChange: boolean | null;
    changeType: string; changeTypeLabel: string;
    agreement: number;                    // 方法一致度 0–1
    votes: { change: number; noChange: number };
    typeVotes: Record<string, number>;
    summary: string;
  };
  timing: {
    totalMs: number; serviceMs: number;
    methodMs: Record<string, number>;
    aspect: {
      enabled: true;
      pointcut: string;
      traceId: string;
      records: Array<{
        seq: number; traceId: string; target: string;
        startedAt: string; elapsedMs: number;
        status: "ok" | "error"; error: string | null; thread: string;
      }>;
    };
  };
}
```

坐标口径：**对齐后画布的像素**。前端把叠加图直接展示，不需要再换算框坐标。

---

## AOP 切面怎么实现的

Python 没有 Spring AOP，这里用**动态代理 + pointcut 正则**做等价物（`core/aspect.py`）：

| Spring AOP | 本项目 |
| --- | --- |
| `@Aspect` | `TimingAspect` |
| `@Pointcut("execution(*.detect(..))")` | `TimingAspect(pointcut=r"\.detect$")` |
| `@Around` | `wrap()`：`perf_counter_ns()` 进入，`finally` 结算（异常也记） |
| `ProceedingJoinPoint.proceed()` | 调用被包裹的 `fn(*args, **kwargs)` |
| 织入 | 服务初始化时 `for m in methods: aspect.weave(m)`，按 pointcut 扫描 `detect` |

服务入口 `CompareService.compare` 也套了一层，因此：

- `methodMs` = 每个 `detect` 从进入到返回的时间；
- `totalMs` = 整个 `compare` 从进入到返回的时间，含解码、对齐、串行调度；
- 两者之差就是框架/服务开销；
- 页面底部「AOP 切面日志」展示每条记录（含 `#seq`、目标、耗时、状态、开始时间），
  也可直接访问 `GET /api/traces?limit=60`。

> 计时不受业务代码影响：实现类里**没有**任何 `time.time()`；新增方法只要方法名匹配
> `.detect`，切面自动生效。

---

## 接口

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| GET | `/api/meta` | 方法清单、限制、切面配置、OCR/YOLO 预热状态 |
| GET | `/api/samples` | 内置示例图对 |
| GET | `/api/traces?limit=60` | 最近的切面记录（倒序） |
| GET | `/api/health` | 探活 |
| POST | `/api/compare` | JSON：`{before,after,methods}`；返回统一契约 |

`methods` 省略时默认跑**当前可用的方法**（YOLO 未配置权重会自动跳过）；传空数组会 400。
显式请求未配置的 YOLO 会返回一条 `details.unavailable=true` 的中性结果，不影响其他方法。单张图 ≤12 MB，请求体 ≤40 MB。

```powershell
# 用示例图直接打接口（PowerShell）
$b = [Convert]::ToBase64String([IO.File]::ReadAllBytes("web\samples\added-before.png"))
$a = [Convert]::ToBase64String([IO.File]::ReadAllBytes("web\samples\added-after.png"))
$body = @{ before = "data:image/png;base64,$b"; after = "data:image/png;base64,$a";
           methods = @("pixel_diff","red_hsv","ssim","ocr","yolo") } | ConvertTo-Json -Depth 4
Invoke-RestMethod -Uri http://127.0.0.1:8791/api/compare -Method Post -ContentType "application/json" -Body $body
```

---

## 关于“准确率”

按你的选择，这一版**只做单次对比，不构建数据集、不计算准确率**：

- 没有真值就没有准确率；页面给出的是每个方法的**判定 + 置信度**，以及方法之间的**一致度**（多数投票）；
- 置信度是方法内部证据的强度（像素占比、红点配对质量、SSIM 差值、OCR 文本差），
  **不等于**“在数据集上对了多少”；
- 如果之后要算准确率：把 `changeType` 的真值加进请求（例如 `truth: "added"`），
  在 `CompareService` 里对比 `hasChange/changeType`，再跑一批标注图对即可；
  四种内置示例已经带了 `expect` 元数据，可以直接作为最小评测集。

---

## 验证

```powershell
# 先生成示例图对（本地真实截图；公开仓库请用合成图）
python tools\make_samples.py --force

# 端到端冒烟：4 个示例对 × 7 个方法，断言结构、判定、切面耗时
python tools\selftest.py
```

脚本会检查：

- 七种方法都返回统一结构；YOLO 未配置权重时自动跳过其判定断言；
- 像素差分 / YOLO 在红点变化的三组上都判“变化”，无变化组判“无变化”；
- HSV 分别给出 `added / removed / moved / none`；
- SSIM / OCR / 灰度指纹对红点不敏感，稳定报“无变化”（盲区符合预期）；
- ORB 对齐在红点变化组上保留残差，在无变化组上判“无变化”；
- 额外构造整图平移 25px / 12px，断言 ORB 能对齐并判“无内容变化”；
- 每条 `detect` 的 `elapsedMs > 0`，切面记录条数 ≥ 方法数。

---

## 文件

```
image-diff-lab/
├─ server.py                  # HTTP 服务 + 路由 + 静态文件
├─ requirements.txt           # 核心依赖（不含 OCR/YOLO）
├─ requirements-ocr.txt       # OCR 可选依赖
├─ requirements-yolo.txt      # YOLO 可选依赖（AGPL-3.0）
├─ requirements-all.txt       # 全量依赖
├─ .env.example               # YOLO_WEIGHTS 示例
├─ core/
│  ├─ models.py               # 统一输出契约 + 多数投票
│  ├─ aspect.py               # TimingAspect / AspectStore（AOP）
│  ├─ imaging.py              # 解码、对齐、叠加层绘制（中文字体）
│  ├─ config.py               # .env 读取
│  ├─ service.py              # 一个接口串起七种方法
│  └─ methods/
│     ├─ base.py              # ChangeMethod 抽象类
│     ├─ pixel_diff.py
│     ├─ red_hsv.py
│     ├─ ssim.py
│     ├─ ocr.py
│     ├─ yolo.py              # BYOM：权重自备
│     ├─ fingerprint.py       # 灰度指纹差分
│     └─ orb_align.py         # ORB 对齐差分
├─ models/                    # 放自己的 best.pt（不提交）
├─ web/                       # 零构建前端
│  ├─ index.html / app.css / app.js
│  └─ samples/                # 本地示例图对（真实截图默认不提交）
├─ tools/
│  ├─ make_samples.py         # 从本机截图生成示例
│  └─ selftest.py             # 端到端冒烟
├─ docs/
│  ├─ 实验报告.md
│  └─ yolo-byom.md            # 如何自己训练/接入 YOLO
└─ logs/aspect.jsonl          # 运行时切面日志（每次调用追加）
```

## 许可证

本项目使用 **GNU Affero General Public License v3.0（AGPL-3.0）**，见 [`LICENSE`](LICENSE)。

原因是可选的 YOLO 方法依赖 **Ultralytics（AGPL-3.0）**。如果你希望使用 MIT / Apache-2.0，
需要把 YOLO 从核心依赖中移除，做成独立插件或替换为许可证兼容的推理方案。
第三方组件与权重说明见 [`NOTICE`](NOTICE) 和 [`docs/yolo-byom.md`](docs/yolo-byom.md)。

> 以上不构成法律意见。

## 已知限制

- **单用户、无鉴权**，只监听 `127.0.0.1`；不要放公网。
- 七个方法**串行**执行，避免 YOLO / OCR / ORB 争抢 CPU/GPU；所以总耗时约等于各方法之和。
- OCR 是 CPU 推理，**秒级**；首次还要加载模型。想快就换 GPU 版 Paddle 或更小的识别模型。
- YOLO 是 **BYOM**：仓库不附带权重；未配置时该方法自动置灰，其他方法照常。接入见 `docs/yolo-byom.md`。
- HSV 会把红色头像、红包、红色图标当成候选；两张图中不变的红块会配对忽略，但整页红色元素很多时噪声会上升。
- 灰度指纹是**全局**判据：红点这类局部小改会被缩图抹平；整图平移也会改变指纹，所以“指纹变了”不等于“内容变了”。
- ORB 对齐依赖足够且分布均匀的特征点；纯色/重复纹理界面可能估计失败。对齐后会把整体滚动/平移视为“无内容变化”，并在 `details.translation` 里报告位移。
- 两张图尺寸不一致时后图会被拉伸到前图尺寸；宽高比差异 >2% 会在响应 `images.warning` 里提示。
- 上传图不落盘：解码 → 对齐 → 分析 → 只在内存里生成叠加图返回。
