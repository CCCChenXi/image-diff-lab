"""端到端冒烟测试：四个内置示例对 × 七种方法。

不是数据集评测，只验证：
  1. 七种方法都能返回统一结构；
  2. 示例对上的判定符合方法定位（像素/HSV/YOLO 应看到红点，OCR/SSIM 看不到）；
  3. AOP 切面确实记录了每一次 detect 的进入→返回耗时。

    ..\\wechat-vision\\.venv\\Scripts\\python.exe tools\\selftest.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.service import CompareService  # noqa: E402

SAMPLES = ROOT / "web" / "samples"
EXPECTED = {"added": "added", "removed": "removed", "moved": "moved", "none": "none"}


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    meta = json.loads((SAMPLES / "samples.json").read_text(encoding="utf-8"))
    service = CompareService()
    failures: list[str] = []

    print("=" * 96)
    print(f"{'图对':<12}{'方法':<18}{'判定':<8}{'类型':<10}{'置信度':<8}{'AOP 耗时':>10}  摘要")
    print("=" * 96)

    for item in meta:
        pid = item["id"]
        payload = {
            "before": data_url(SAMPLES / item["before"]),
            "after": data_url(SAMPLES / item["after"]),
            "methods": [m.key for m in service.methods],
        }
        response = service.compare(payload)
        data = response.to_dict()

        if not data.get("ok"):
            failures.append(f"{pid}: 响应失败")
        records = data["timing"]["aspect"]["records"]
        if len(records) < len(service.methods):
            failures.append(f"{pid}: 切面记录只有 {len(records)} 条")

        by_key = {m["key"]: m for m in data["methods"]}
        for method in data["methods"]:
            unavailable = bool(method.get("details", {}).get("unavailable"))
            verdict = "未配置" if unavailable else ("变化" if method["hasChange"] else "无变化")
            print(
                f"{item['label']:<10}{method['name']:<18}{verdict:<8}"
                f"{method['changeTypeLabel']:<10}{method['confidence']:<8.2f}"
                f"{method['elapsedMs']:>8.1f}ms  {method['summary'][:42]}"
            )
            if not method["ok"]:
                failures.append(f"{pid}/{method['key']}: {method['error']}")
            if method["elapsedMs"] <= 0:
                failures.append(f"{pid}/{method['key']}: 耗时为 0，切面未生效")

        # 像素差分：四个示例都应给出正确的“有无变化”
        px = by_key["pixel_diff"]
        if px["hasChange"] != (pid != "none"):
            failures.append(f"{pid}: 像素差分判定错误 hasChange={px['hasChange']}")

        # HSV：应给出具体的新增/消失/移位/无变化
        hsv = by_key["red_hsv"]
        if hsv["changeType"] != EXPECTED[pid]:
            failures.append(f"{pid}: HSV 类型 {hsv['changeType']} != {EXPECTED[pid]}")

        # YOLO：红点变化应能被元素内像素差捕获；未配置权重时跳过（BYOM）
        yolo = by_key["yolo"]
        if yolo.get("details", {}).get("unavailable"):
            print("  └ YOLO 未配置权重（BYOM），跳过该方法的判定断言")
        elif yolo["hasChange"] != (pid != "none"):
            failures.append(f"{pid}: YOLO 判定错误 hasChange={yolo['hasChange']}")

        # SSIM / OCR / 灰度指纹：全局或文本方法，本来就看不到小红点，应稳定报“无变化”
        for key in ("ssim", "ocr", "gray_fingerprint"):
            if by_key[key]["hasChange"]:
                failures.append(
                    f"{pid}: {key} 本应对红点不敏感，却报了变化 —— {by_key[key]['summary']}"
                )

        # ORB 对齐：红点变化应保留为残差
        if by_key["orb_align"]["hasChange"] != (pid != "none"):
            failures.append(
                f"{pid}: ORB 判定错误 hasChange={by_key['orb_align']['hasChange']}"
            )

        print(
            f"  └ 切面 {len(records)} 条，服务总耗时 {data['timing']['totalMs']:.1f}ms，"
            f"综合：{data['consensus']['summary']}"
        )
        print("-" * 96)

    # 额外验证 ORB 的看家本领：整张图平移 25px / 12px 后，应能对齐并判“无内容变化”
    base_path = SAMPLES / "none-before.png"
    base = cv2.imdecode(np.fromfile(str(base_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = base.shape[:2]
    shifted = cv2.warpAffine(
        base,
        np.float32([[1, 0, 25], [0, 1, 12]]),
        (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )
    ok, buf = cv2.imencode(".png", shifted)
    payload = {
        "before": data_url(base_path),
        "after": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
        "methods": ["orb_align"],
    }
    shift_data = service.compare(payload).to_dict()
    orb = shift_data["methods"][0]
    tx, ty = orb["details"].get("translation") or [0, 0]
    print(
        f"{'整图平移':<10}{orb['name']:<18}"
        f"{('变化' if orb['hasChange'] else '无变化'):<8}"
        f"{orb['changeTypeLabel']:<10}{orb['confidence']:<8.2f}"
        f"{orb['elapsedMs']:>8.1f}ms  估计位移 ({tx:+.1f}, {ty:+.1f})"
    )
    if orb["hasChange"]:
        failures.append("整图平移：ORB 对齐后仍误报变化")
    if not orb["details"].get("aligned"):
        failures.append("整图平移：ORB 未成功对齐")
    if abs(tx + 25) > 3 or abs(ty + 12) > 3:
        failures.append(f"整图平移：ORB 位移估计偏差过大 ({tx}, {ty})")
    print("-" * 96)

    print("=" * 96)
    if failures:
        print(f"❌ {len(failures)} 项断言失败：")
        for f in failures:
            print("  -", f)
        return 1
    print("✅ 全部通过：七种方法输出结构一致，切面计时生效，示例判定符合预期。")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Paddle 的 C++ 清理阶段偶尔改退出码，用 os._exit 保留断言结果
    os._exit(code)
