"""Compose a recorded auto-weld grid run into a captioned demo video (1920x1080 MP4).

    python main.py ... --seam-source auto-weld --record-demo --output-dir outputs/runs/b2_e10l_grid
    python scripts/make_demo_video.py outputs/runs/b2_e10l_grid
"""

import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1920, 1080, 30
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
FONT_INDEX = 2  # Noto Sans CJK SC
BACKGROUND = (18, 22, 28)
# Playback speed per controller phase; frames were recorded at 30 per simulated second.
PHASES = {
    "settle": ("①", "站立与初始化", "B2 足端接触站稳，机械臂回到 home，焊枪抬离前置相机视野", 2),
    "transfer": ("⑥", "移到起焊点后方", "关节空间运动，终点在接近点后方 0.15 m，远离板面", 4),
    "approach": ("⑦", "沿焊枪轴接近", "直线接近；执行中按躯干实时位姿补偿机身晃动", 2),
    "descend": ("⑦", "下降到焊接高度", "焊尖沿 45° 角平分线停在离焊缝 1 cm 处", 2),
    "trace": ("⑧", "沿焊缝焊接", "平角焊（2F）：工作角 45°，沿 DA3 定出的交线匀速前进", 2),
    "retreat": ("⑨", "退回", "沿焊枪轴抬离焊缝", 2),
    "done": ("⑨", "完成", "蓝线：规划的焊尖轨迹 · 橙线：实际焊尖轨迹 · 绿线：焊缝段", 1),
}


def font(size):
    return ImageFont.truetype(FONT, size, index=FONT_INDEX)


def find_ffmpeg():
    found = shutil.which("ffmpeg") or next(iter(sorted(glob.glob(str(
        Path.home() / "miniconda3/envs/*/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")))), None)
    if not found:
        raise FileNotFoundError("ffmpeg not found; pass --ffmpeg")
    return found


def fit(image, width, height):
    image = image.copy()
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    return image


def caption_box(frame, number, title, subtitle, corner="left"):
    """A translucent caption in the top-left corner; returns the frame."""
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    big, small = font(46), font(28)
    text = f"{number} {title}"
    width = max(draw.textlength(text, font=big), draw.textlength(subtitle, font=small)) + 64
    x = 40 if corner == "left" else W - 40 - width
    draw.rounded_rectangle((x, 36, x + width, 176), radius=18, fill=(10, 14, 20, 190))
    draw.text((x + 32, 50), text, font=big, fill=(255, 255, 255, 255))
    draw.text((x + 32, 120), subtitle, font=small, fill=(200, 215, 230, 255))
    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")


def tag(frame, text, xy):
    draw = ImageDraw.Draw(frame)
    small = font(26)
    width = draw.textlength(text, font=small)
    x, y = xy
    draw.rounded_rectangle((x - width - 24, y, x, y + 46), radius=12, fill=(10, 14, 20))
    draw.text((x - width - 12, y + 6), text, font=small, fill=(255, 210, 120))
    return frame


def still(image, number, title, subtitle, seconds):
    frame = Image.new("RGB", (W, H), BACKGROUND)
    content = fit(image, W - 80, H - 230)
    frame.paste(content, ((W - content.width) // 2, 210 + (H - 230 - content.height) // 2))
    frame = caption_box(frame, number, title, subtitle)
    return [frame] * int(seconds * FPS)


def card(lines, seconds, title=None):
    frame = Image.new("RGB", (W, H), BACKGROUND)
    draw = ImageDraw.Draw(frame)
    y = 220
    if title:
        draw.text((140, 120), title, font=font(64), fill=(255, 255, 255))
        y = 300
    for size, color, text in lines:
        draw.text((140, y), text, font=font(size), fill=color)
        y += int(size * 1.9)
    return [frame] * int(seconds * FPS)


def labelled(image, text):
    frame = image.copy()
    draw = ImageDraw.Draw(frame)
    draw.rounded_rectangle((12, 12, 24 + draw.textlength(text, font=font(26)), 58), radius=10, fill=(10, 14, 20))
    draw.text((18, 16), text, font=font(26), fill=(255, 255, 255))
    return frame


def depth_view(run, scale):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps

    with np.load(run / "auto_weld/data/auto_results.npz", allow_pickle=False) as arrays:
        depth = arrays["image_000_depth_m"] * scale
    low, high = np.percentile(depth, [2, 98])
    rgb = colormaps["turbo"](np.clip((depth - low) / (high - low), 0, 1))[:, :, :3]
    image = Image.fromarray((rgb * 255).astype(np.uint8)).resize((640, 480), Image.Resampling.BILINEAR)
    return image, low, high


def seam_view(run, seam_report):
    """Every plate-intersection candidate in orange; the selected interval in red."""
    report = json.loads((run / "auto_weld/data/auto_report.json").read_text())["results"][0]
    image = Image.open(run / "body_rgb.png").convert("RGB").resize((1280, 960), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    with np.load(run / "auto_weld/data/auto_results.npz", allow_pickle=False) as arrays:
        for path in report["paths"]:
            xy = (arrays[path["xy_key"]] + .5) * 2 - .5
            draw.line([tuple(p) for p in xy], fill=(255, 170, 0), width=4)
        chosen = next(p for p in report["paths"] if p["id"] == seam_report["path_id"])
        run_range = seam_report["geometry_run"]
        xy = (arrays[chosen["xy_key"]][run_range["start_sample"]:run_range["stop_sample"]] + .5) * 2 - .5
        draw.line([tuple(p) for p in xy], fill=(255, 40, 40), width=9)
    return image


def execution_frames(run, timeline):
    kept, previous, count = [], None, 0
    for item in timeline["frames"]:
        phase = item["phase"]
        if phase not in PHASES:
            continue
        if phase != previous:
            count, previous = 0, phase
        if count % PHASES[phase][3] == 0:
            kept.append(item)
        count += 1
    for item in kept:
        number, title, subtitle, speed = PHASES[item["phase"]]
        frame = Image.open(run / f"demo/overview/{item['index']:05d}.jpg").convert("RGB").resize((W, H))
        closeup = Image.open(run / f"demo/closeup/{item['index']:05d}.jpg").convert("RGB").resize((640, 360))
        frame.paste(Image.new("RGB", (648, 368), (240, 240, 240)), (W - 688, H - 408))
        frame.paste(labelled(closeup, "焊枪近景"), (W - 684, H - 404))
        frame = caption_box(frame, number, title, subtitle)
        tag(frame, f"×{speed}  仿真时间 {item['sim_time_s']:.1f} s", (W - 40, 40))
        yield item["phase"], frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Default: <run_dir>/demo.mp4")
    parser.add_argument("--ffmpeg", help="ffmpeg executable with libx264")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = (args.output or run / "demo.mp4").resolve()
    timeline = json.loads((run / "demo/timeline.json").read_text())
    seam = json.loads((run / "seam_report.json").read_text())
    execution = json.loads((run / "execution_report.json").read_text())
    if not (run / "grid_evaluation.json").is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_grid_weld.py"), str(run)], check=True,
                       stdout=subprocess.DEVNULL)
    evaluation = json.loads((run / "grid_evaluation.json").read_text())
    motions = len(np.load(run / "trajectory.npz")["phases"]) - 1

    encoder = subprocess.Popen([args.ffmpeg or find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                str(output)], stdin=subprocess.PIPE)
    written = 0

    def emit(frames):
        nonlocal written
        for frame in frames:
            encoder.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
            written += 1

    emit(card([(40, (200, 215, 230), "仅用 B2 前置 RGB 相机 · auto_weld：DA3 单目深度 + SAM3 语义 + 板面交线几何检测"),
               (40, (200, 215, 230), "宇树 B2 背负华言 E10L-Pro 与 TRM351 焊枪 · Isaac Sim 6 物理仿真"),
               (34, (150, 165, 180), "QuadWeld · 船板格架中央格，肋板立板与足板的平角焊缝")],
              4, title="四足机器人船板格架角焊：完整流程演示"))
    stream = execution_frames(run, timeline)
    first_transfer = None
    for phase, frame in stream:
        if phase != "settle":
            first_transfer = frame
            break
        emit([frame])

    rgb = Image.open(run / "body_rgb.png").convert("RGB")
    emit(still(rgb.resize((1280, 960), Image.Resampling.LANCZOS), "②", "前置 RGB 相机拍摄",
               "B2 头部相机（URDF f_dc_link，俯视 45°）拍一张 640×480 彩色图，不使用深度相机", 4))
    depth, low, high = depth_view(run, seam["depth_scale"])
    pair = Image.new("RGB", (1800, 680), BACKGROUND)
    pair.paste(labelled(rgb.resize((880, 660)), "RGB 输入"), (0, 0))
    pair.paste(labelled(depth.resize((880, 660)), f"DA3 深度（校正后 {low:.2f}–{high:.2f} m）"), (920, 0))
    emit(still(pair, "③", "DA3 单目深度 + 支撑面尺度校正",
               f"足端支撑面离相机 {seam['support_plane_known_distance_m']:.2f} m，DA3 原估 "
               f"{seam['support_plane_monocular_distance_m']:.2f} m → 深度尺度 ×{seam['depth_scale']:.2f}", 5))
    emit(still(seam_view(run, seam), "④", "auto_weld 检出板面交线",
               f"橙：全部 {len(json.loads((run / 'auto_weld/data/auto_report.json').read_text())['results'][0]['paths'])} 条交线候选 · "
               f"红：选中的立板-足板交角（{seam['seam_length_m']:.2f} m）", 5))
    if first_transfer is not None:
        plan = caption_box(first_transfer, "⑤", "角焊规划",
                           f"焊段 {seam['weld_length_m']:.2f} m · 工作角 45° · 焊枪绕喷嘴轴 {seam['torch_roll_deg']:.0f}° · "
                           f"关节离限位 ≥ {seam['min_joint_limit_margin_rad']:.2f} rad · 绿/蓝线为焊缝段与焊尖轨迹")
        emit([plan] * int(4 * FPS))
        emit([first_transfer])
    for _, frame in stream:
        emit([frame])

    seam_offsets = evaluation["perceived_seam"]
    executed = evaluation.get("executed_trace", {})
    emit(card([
        (40, (140, 230, 150), f"执行完成：{motions} 个运动目标，"
                              f"沿线跟踪误差最大 {execution.get('trace_max_path_error_m', float('nan')) * 1000:.2f} mm"),
        (40, (255, 210, 120), f"DA3 定出的焊缝：在真实交角前方 {seam_offsets['offset_from_web_face']['mean_m'] * 100:.1f} cm、"
                              f"上方 {seam_offsets['offset_from_flange_top']['mean_m'] * 100:.1f} cm"),
        (40, (255, 210, 120), f"焊尖离真实交角 {executed.get('tcp_distance_to_corner_line', {}).get('mean_m', float('nan')) * 100:.1f} cm"
                              f"（设计 1 cm），{'未' if not executed.get('touched_a_face') else ''}接触钢板"),
        (34, (200, 215, 230), "交线在图像中准确（约 4 px），误差来自 DA3 在深色立板处的局部深度偏短"),
        (34, (200, 215, 230), "下一步：用深度相机或焊丝触碰寻位校正焊缝位置"),
    ], 7, title="结果（仿真真值评估）"))
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("ffmpeg failed")
    print(f"Saved {output} ({written / FPS:.1f} s, {written} frames)")


if __name__ == "__main__":
    main()
