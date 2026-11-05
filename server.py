"""图像变化检测实验台 —— 一个接口、七种方法、一个 AOP 计时切面。

    # 安装依赖（OCR 可选；YOLO 可选且权重需自备）
    pip install -r requirements-ocr.txt

    # 启动
    python server.py --port 8791

    # 浏览器
    http://127.0.0.1:8791/

    # YOLO 权重（BYOM）
    python server.py --weights models/your-best.pt
    # 或设置 YOLO_WEIGHTS，详见 docs/yolo-byom.md

接口
----
    GET  /api/meta                 → 方法清单 + 限制 + 切面配置 + 模型状态
    GET  /api/samples              → 内置示例图对
    GET  /api/traces?limit=60      → 最近的切面调用记录
    GET  /api/health               → 探活
    POST /api/compare              → body: {before:dataURL, after:dataURL, methods:[...]}
                                     → {ok, consensus, methods[], timing{aspect{records}}, images}
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from core.service import VERSION, CompareService

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
SAMPLES = WEB / "samples"
MAX_BODY = 40 * 1024 * 1024

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def _json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = f"ImageDiffLab/{VERSION}"
    protocol_version = "HTTP/1.1"

    # ---------------------------------------------------------------- 基础
    def _send(self, code: int, body: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code: int, data: Any) -> None:
        self._send(code, _json_bytes(data), "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"ok": False, "error": message})

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self._error(404, "文件不存在")
            return
        ctype = MIME.get(path.suffix.lower()) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        cache = "public, max-age=3600" if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") else "no-cache"
        self._send(200, path.read_bytes(), ctype, cache)

    # ---------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in ("/", "/index.html"):
            self._serve_file(WEB / "index.html")
            return
        if path == "/api/health":
            self._json(
                200,
                {
                    "ok": True,
                    "version": VERSION,
                    "service": "image-diff-lab",
                },
            )
            return
        if path == "/api/meta":
            self._json(200, {"ok": True, **self.server.service.meta()})  # type: ignore[attr-defined]
            return
        if path == "/api/samples":
            self._json(200, {"ok": True, "samples": self._samples()})
            return
        if path == "/api/traces":
            qs = parse_qs(parsed.query)
            try:
                limit = int((qs.get("limit") or ["60"])[0])
            except ValueError:
                limit = 60
            limit = max(1, min(200, limit))
            records = self.server.service.store.recent(limit)  # type: ignore[attr-defined]
            self._json(200, {"ok": True, "count": len(records), "records": records})
            return
        if path.startswith("/web/"):
            target = (WEB / path[len("/web/") :]).resolve()
            try:
                target.relative_to(WEB.resolve())
            except ValueError:
                self._error(403, "非法路径")
                return
            self._serve_file(target)
            return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        self._error(404, "接口不存在")

    # ---------------------------------------------------------------- POST
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/compare":
            self._error(404, "接口不存在")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(400, "Content-Length 不合法")
            return
        if length <= 0:
            self._error(400, "请求体为空")
            return
        if length > MAX_BODY:
            self._error(413, f"请求体超过 {MAX_BODY // 1024 // 1024} MB")
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON 对象")
        except Exception as exc:
            self._error(400, f"JSON 解析失败：{exc}")
            return

        try:
            response = self.server.service.compare(payload)  # type: ignore[attr-defined]
            self._json(200, response.to_dict())
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:  # 保留完整堆栈到服务端日志
            traceback.print_exc()
            self._error(500, f"服务端错误：{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- 杂项
    def _samples(self) -> list[dict[str, Any]]:
        meta_path = SAMPLES / "samples.json"
        if not meta_path.is_file():
            return []
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            before = SAMPLES / str(item.get("before", ""))
            after = SAMPLES / str(item.get("after", ""))
            if before.is_file() and after.is_file():
                out.append(
                    {
                        "id": item.get("id"),
                        "label": item.get("label"),
                        "note": item.get("note", ""),
                        "expect": item.get("expect", "unknown"),
                        "before": f"/web/samples/{before.name}",
                        "after": f"/web/samples/{after.name}",
                    }
                )
        return out

    def log_message(self, fmt: str, *args: Any) -> None:
        # 只留一行，别把 BaseHTTPRequestHandler 的默认日志刷满终端
        print(f"[http] {self.address_string()} {fmt % args}")


class LabServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], service: CompareService) -> None:
        super().__init__(addr, Handler)
        self.service = service


def main() -> None:
    parser = argparse.ArgumentParser(description="图像变化检测实验台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument(
        "--weights",
        default=None,
        help="YOLO best.pt 路径；留空时依次读 YOLO_WEIGHTS、models/*.pt（BYOM，权重不随仓库提供）",
    )
    parser.add_argument("--no-warmup", action="store_true", help="启动时不后台预热 OCR/YOLO")
    args = parser.parse_args()

    service = CompareService(weights=args.weights)
    server = LabServer((args.host, args.port), service)

    print("=" * 64)
    print(" 图像变化检测实验台")
    print(f" 页面   http://{args.host}:{args.port}/")
    print(" 方法   " + " / ".join(m.name for m in service.methods))
    print(" AOP    execution(*.detect(..)) + CompareService.compare")
    print("=" * 64)
    yolo = next((m for m in service.methods if m.key == "yolo"), None)
    if yolo and getattr(yolo, "available", False):
        print(f" YOLO 权重：{yolo.weight_path}")
    else:
        reason = getattr(yolo, "unavailable_reason", "") if yolo else ""
        print(f" YOLO  ：未配置权重（BYOM），已跳过。{reason}")

    if not args.no_warmup:
        print(" 后台预热 OCR / YOLO…（页面上的模型状态会逐步变成 ready）")
        service.warmup_async()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n 收到 Ctrl+C，正在关闭…")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
