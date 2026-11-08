#!/usr/bin/env python3
"""Encode the traced pipe frames; compare decoded candidates and curate public masters."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def call(args):
    r = subprocess.run(args, text=True, capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT/"work/ray-pipe-final")
    p.add_argument("--validation", type=Path, default=ROOT/"work/ray-pipe-validation")
    a = p.parse_args()
    src, val = a.input.resolve(), a.validation.resolve()
    m = json.loads((src/"manifest.json").read_text())
    v = json.loads((val/"manifest.json").read_text())
    assert v["cpu_comparison"] is not None
    n, fps = m["frames"], m["fps"]
    assert (src/"frame-0000.png").read_bytes() == (src/f"frame-{n:04d}.png").read_bytes()
    base = ["ffmpeg","-v","error","-y","-filter_complex_threads","2","-threads","4","-framerate",str(fps),"-i",str(src/"frame-%04d.png"),"-frames:v",str(n),"-threads","4"]
    choices = [
        ("mp4",["-c:v","libx264","-crf","22","-pix_fmt","yuv420p","-movflags","+faststart"]),
        ("vp9.webm",["-c:v","libvpx-vp9","-b:v","0","-crf","32","-cpu-used","4","-row-mt","1","-pix_fmt","yuv420p"]),
        ("av1.webm",["-c:v","libaom-av1","-b:v","0","-crf","35","-cpu-used","6","-row-mt","1","-pix_fmt","yuv420p"]),
        ("webp",["-c:v","libwebp_anim","-quality","70","-loop","0"]),
        ("gif",["-filter_complex","split[a][b];[a]palettegen[p];[b][p]paletteuse","-loop","0"]),
    ]
    metrics = []
    for suffix, options in choices:
        path = src/("candidate."+suffix)
        try:
            call(base+options+[str(path)])
            probe = json.loads(call(["ffprobe","-v","error","-count_frames","-show_streams","-show_format","-of","json",str(path)]).stdout)
            s = probe["streams"][0]
            count = int(s.get("nb_read_frames",0))
            if count != n:
                raise RuntimeError(f"decoded frame count {count}, expected {n}")
            # Align by decoded frame number so GIF timestamp quantization does not
            # change which reference frame is compared; duration is checked too.
            duration = float(probe["format"].get("duration",s.get("duration",0)))
            assert abs(duration-n/fps) <= 1/fps+.01, f"decoded duration {duration}, expected {n/fps}"
            result = call(["ffmpeg","-hide_banner","-i",str(path),"-framerate",str(fps),"-i",str(src/"frame-%04d.png"),"-filter_complex",f"[0:v]setpts=N/({fps}*TB),format=yuv420p[d];[1:v]setpts=N/({fps}*TB),format=yuv420p[r];[d][r]psnr","-frames:v",str(n),"-f","null","-"])
            match = re.search(r"average:([0-9.]+|inf)",result.stderr)
            psnr = float(match[1])
            metrics.append(dict(file=path.name,bytes=path.stat().st_size,codec=s["codec_name"],frames=count,duration=duration,psnr_db=psnr,eligible=psnr>=30))
        except (RuntimeError,AssertionError,ValueError,KeyError,TypeError) as e:
            metrics.append(dict(file=path.name,bytes=path.stat().st_size if path.exists() else 0,eligible=False,reason=(str(e) or type(e).__name__)[-500:]))
        print(metrics[-1],flush=True)
    winner = min((x for x in metrics if x["eligible"]),key=lambda x:x["bytes"])
    dst = ROOT/"site-assets/ray/pipes-rtx"
    dst.mkdir(parents=True,exist_ok=True)
    ext = "webm" if winner["file"].endswith("webm") else winner["file"].split('.')[-1]
    selected = "flythrough."+ext
    shutil.copyfile(src/winner["file"],dst/selected)
    if selected != "flythrough.mp4":
        shutil.copyfile(src/"candidate.mp4",dst/"flythrough.mp4")
    shutil.copyfile(src/"frame-0040.png",dst/"poster.png")
    for name in ["geometry.xtl","function.xtl","scene.xir","triangles.txt","materialids.txt","timings.txt"]:
        shutil.copyfile(src/name,dst/name)
    shutil.copyfile(val/"manifest.json",dst/"validation.json")
    shutil.copyfile(val/"function.xtl",dst/"validation-function.xtl")
    shutil.copyfile(val/"validation-cpu.out",dst/"validation-cpu.out")
    shutil.copyfile(val/"batches/batch-0000.out",dst/"validation-gpu.out")
    m.update(selected_animation=selected,selected_codec=winner["codec"],candidates=metrics,quality_floor_psnr_db=30,
             loop_seam="GPU-rendered frame at index framecount matches frame zero byte for byte; endpoint excluded from encoded playback",
             validation=v["cpu_comparison"],renderer_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ["components/cli/src/main.rs","components/cli/src/batches.rs"]},renderer_git_base=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
             masters=[],raw_float_storage=f"All raw GPU batches and frame TIFFs retained in {src.relative_to(ROOT)}; public archives contain every TIFF frame and source, not full Float streams")
    for start in range(0,n,30):
        name=f"masters-{start//30:02d}.zip"
        with zipfile.ZipFile(dst/name,"w",compression=zipfile.ZIP_STORED) as z:
            for f in range(start,min(start+30,n)):
                z.write(src/f"frame-{f:04d}.tif",f"frame-{f:04d}.tif")
            z.write(src/"frame-0000.png","loop-start.png")
            z.write(src/f"frame-{n:04d}.png","loop-end.png")
            for file in ["geometry.xtl","function.xtl","scene.xir","triangles.txt","materialids.txt"]:
                z.write(dst/file,file)
            z.write(ROOT/"LICENSE","LICENSE")
        m["masters"].append(dict(file=name,first_frame=start,frames=min(30,n-start)))
    m["artifact_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.iterdir() if p.name!="manifest.json"}
    (dst/"manifest.json").write_text(json.dumps(m,indent=2)+"\n")
    print(f"Selected {selected}, {winner['bytes']} bytes; {winner['psnr_db']:.2f} dB; every native TIFF frame archived",flush=True)


if __name__ == "__main__":
    main()
