"""Chạy thực hành rồi bài tập, lưu stdout/stderr và ảnh kết xuất log thực tế.

Ảnh PNG là bản kết xuất văn bản đầu ra, không phải ảnh chụp terminal.
Chạy lại file này sau khi thay CSV để cập nhật mọi kết quả.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parent


def write_report():
    """Gộp lại các log có thật; ghi dấu vân tay dữ liệu để đối chiếu lần chạy."""
    import importlib.metadata
    generated = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M %z")
    sources = {"Lab07_DanhGiaMoHinh": "sinh_vien_bo_hoc.csv",
               "Lab08_KMeans": "khach_hang.csv", "Lab09_MLP": "xor.csv"}
    synthetic = sources[ROOT.name]
    parts = [f"# Kết quả chạy {ROOT.name}\n",
             f"Cập nhật: {generated}. Python {sys.version.split()[0]}.\n",
             "Các giá trị dưới đây được lấy trực tiếp từ stdout/stderr, không chép số mẫu trong PDF.\n",
             f"**Nguồn:** `{synthetic}` đi kèm hiện là dữ liệu mô phỏng thay thế có seed cố định. "
             "Xem README trước khi đối chiếu với PDF. Nếu đã thay CSV gốc, cập nhật ghi chú nguồn trong README.\n",
             "Ảnh trong `ket_qua/` là kết xuất văn bản đầu ra, không phải ảnh chụp terminal.\n",
             "## Môi trường\n"]
    for package in ["numpy", "pandas", "scikit-learn", "matplotlib", "Pillow"]:
        parts.append(f"- {package}: {importlib.metadata.version(package)}")
    parts.append("\n## Dấu vân tay dữ liệu (SHA-256)\n")
    for path in sorted((ROOT/"data").glob("*.csv")):
        parts.append(f"- `{path.name}`: `{hashlib.sha256(path.read_bytes()).hexdigest()}`")
    parts.append("\n## Đầu ra từng chương trình\n")
    scripts = sorted((ROOT/"code").glob("*.py"))
    scripts += sorted(p for folder in ROOT.glob("baitap*") for p in folder.glob("bai*.py"))
    for script in scripts:
        log = ROOT/"ket_qua"/(script.stem+".txt")
        if log.exists():
            parts.extend([f"\n### {script.relative_to(ROOT).as_posix()}\n",
                          f"[Log TXT](./ket_qua/{script.stem}.txt) · [Ảnh kết quả](./ket_qua/{script.stem}.png)\n",
                          "```text\n"+log.read_text(encoding="utf-8").strip()+"\n```\n"])
    (ROOT/"KET_QUA.md").write_text("\n".join(parts).rstrip()+"\n", encoding="utf-8", newline="\n")


def render_log(content, target):
    from PIL import Image, ImageDraw, ImageFont
    candidates = [Path("C:/Windows/Fonts/consola.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")]
    font_path = next((p for p in candidates if p.exists()), None)
    font = ImageFont.truetype(str(font_path), 17) if font_path else ImageFont.load_default()
    lines = []
    for line in content.splitlines():
        lines.extend(textwrap.wrap(line.expandtabs(4), width=108, replace_whitespace=False,
                                   drop_whitespace=False) or [""])
    line_height = 24
    image = Image.new("RGB", (1160, max(160, 55+line_height*len(lines))), "#101827")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((24, 20+i*line_height), line, fill="#e6edf3", font=font)
    image.save(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chi", nargs="*", help="Chi chay cac stem, vi du bai1 bai3")
    args = parser.parse_args()
    scripts = sorted((ROOT/"code").glob("*.py"))
    scripts += sorted(p for folder in ROOT.glob("baitap*") for p in folder.glob("bai*.py"))
    if args.chi:
        known = {p.stem for p in scripts}
        if set(args.chi)-known:
            parser.error("Khong tim thay: " + ", ".join(set(args.chi)-known))
        scripts = [p for p in scripts if p.stem in args.chi]
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", MPLBACKEND="Agg", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1")
    out = ROOT / "ket_qua"; out.mkdir(exist_ok=True)
    summary = []
    for script in scripts:
        started = time.perf_counter()
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding="utf-8")
        elapsed = time.perf_counter()-started
        header = (f"Tran An Ky - 2474802010460 | {ROOT.name}\n"
                  f"LENH: python {script.relative_to(ROOT).as_posix()}\n"
                  f"MA THOAT: {result.returncode}\n"
                  "NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.\n"
                  "BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL\n\n")
        content = header+result.stdout
        if result.stderr:
            content += "\nSTDERR (canh bao/loi):\n"+result.stderr
        content = "\n".join(line.rstrip() for line in content.splitlines())+"\n"
        (out/(script.stem+".txt")).write_text(content, encoding="utf-8", newline="\n")
        render_log(content, out/(script.stem+".png"))
        summary.append({"file": script.relative_to(ROOT).as_posix(), "exit_code": result.returncode,
                        "seconds": round(elapsed, 2), "stderr": bool(result.stderr)})
        print(f"[{result.returncode}] {script.relative_to(ROOT)} ({elapsed:.1f}s)", flush=True)
    # Giữ kết quả gần nhất của các bài khác khi chạy --chi.
    summary_path = out/"tong_hop.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    merged = {row["file"]: row for row in previous}
    merged.update({row["file"]: row for row in summary})
    summary_path.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    write_report()
    if any(row["exit_code"] for row in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
