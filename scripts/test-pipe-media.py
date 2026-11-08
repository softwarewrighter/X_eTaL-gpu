#!/usr/bin/env python3
"""Independently check traced pipe media, complete masters and resident execution evidence."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pipe_master_check",ROOT/"scripts/test-ray-media.py")
media = importlib.util.module_from_spec(spec)
spec.loader.exec_module(media)
p=ROOT/"site-assets/ray/pipes-rtx"
m=json.loads((p/"manifest.json").read_text())
w,h,n,fps=m["width"],m["height"],m["frames"],m["fps"]
assert (w,h,n,fps)==(640,360,180,15)
assert m["batch_frames"]>1 and m["resident_scene"]
assert m["camera_height"]>m["outer_radius"]-m["inner_radius"]
assert abs(m["camera_height"]-m["outer_radius"])<m["inner_radius"]
assert m["sphere_right"][0]<0 and m["pyramid_left"][0]>0 # +z view has screen right at -x.
assert m["pipe_starts"][1]-m["pipe_length"]>0
for name,digest in m["artifact_sha256"].items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((p/"function.xtl").read_bytes()).hexdigest()==m["function_sha256"]
assert hashlib.sha256((p/"geometry.xtl").read_bytes()).hexdigest()==m["geometry_sha256"]
assert hashlib.sha256((p/"scene.xir").read_bytes()).hexdigest()==m["ir_sha256"]
times=(p/"timings.txt").read_text().splitlines()
repeats=[s for s in times if s.startswith("repeat ")]
assert len(repeats)==n//m["batch_frames"]+1
assert all("setup 0.000 ms" in s for s in repeats[1:])
assert all(float(re.search(r"upload ([0-9.]+) ms",s)[1])>=0 for s in repeats)
v=json.loads((p/"validation.json").read_text())
assert v["cpu_comparison"]["rgb8_different_channels"]==0
for key in ["geometry_sha256","renderer_binary_sha256","xetal","libraries"]:
    assert m[key]==v[key]
a,b=[list(map(float,(p/f"validation-{backend}.out").read_text().split())) for backend in ["gpu","cpu"]]
assert len(a)==len(b)==v["width"]*v["height"]*v["batch_frames"]*6
assert all(math.isfinite(x) and abs(x-y)<=1e-5*(1+abs(y)) for x,y in zip(a,b))
seen=[]
for item in m["masters"]:
    with zipfile.ZipFile(p/item["file"]) as z:
        assert all(f.compress_type==zipfile.ZIP_STORED for f in z.infolist())
        assert z.read("loop-start.png")==z.read("loop-end.png")
        for frame in range(item["first_frame"],item["first_frame"]+item["frames"]):
            rgb=media.tiff_pixels(z.read(f"frame-{frame:04d}.tif"),w,h)
            assert len(rgb)==w*h*3
            seen.append(frame)
assert seen==list(range(n))
chosen=next(c for c in m["candidates"] if c["file"].endswith(m["selected_animation"].split('.')[-1]) and c.get("codec")==m["selected_codec"] and c["eligible"])
assert chosen["bytes"]==min(c["bytes"] for c in m["candidates"] if c["eligible"])
for name in set([m["selected_animation"],"flythrough.mp4"]):
    r=subprocess.run(["ffprobe","-v","error","-count_frames","-show_streams","-show_format","-of","json",str(p/name)],capture_output=True,text=True,check=True)
    data=json.loads(r.stdout); s=data["streams"][0]
    assert int(s["nb_read_frames"])==n and (s["width"],s["height"])==(w,h)
    assert abs(float(data["format"]["duration"])-n/fps)<.07
print("pipe media: complete native TIFFs, GPU loop endpoint, resident batches, CPU/GPU pixels and smallest quality-qualified codec pass")
