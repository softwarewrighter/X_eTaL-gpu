#!/usr/bin/env python3
"""Check the built gallery in Chromium through local WebDriver, without Python browser packages."""
import argparse
import base64
from contextlib import suppress
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from threading import Thread
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="project or private preview root containing pages/")
    root = parser.parse_args().root.resolve()
    assert shutil.which("chromedriver") and shutil.which("chromium"), "Chromium and chromedriver required"
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root / "pages")))
    Thread(target=server.serve_forever, daemon=True).start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    site = f"http://127.0.0.1:{server.server_port}"
    screenshots = root / "work/ray-live/browser-check"
    screenshots.mkdir(parents=True, exist_ok=True)
    def request(path, payload=None, method=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as result:
            return json.load(result)["value"]
    driver, session = None, None
    try:
        with tempfile.TemporaryDirectory(prefix="ray-browser-") as profile:
            with (screenshots / "chromedriver.log").open("w") as log:
                driver = subprocess.Popen(["chromedriver", f"--port={port}"], stdout=log, stderr=log)
                for _ in range(100):
                    try:
                        request("/status")
                        break
                    except OSError:
                        time.sleep(.1)
                result = request("/session", {"capabilities": {"alwaysMatch": {"browserName": "chrome", "goog:chromeOptions": {
                    "binary": shutil.which("chromium"), "args": ["--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
                    "--disable-gpu", "--autoplay-policy=no-user-gesture-required", f"--user-data-dir={profile}"]}}}})
                session = "/session/" + result["sessionId"]
                def script(code):
                    return request(session + "/execute/sync", {"script": code, "args": []})
                def wait(code):
                    for _ in range(150):
                        if script(code):
                            return
                        time.sleep(.1)
                    raise AssertionError("browser condition timed out: " + code)
                request(session + "/window/rect", {"width": 1100, "height": 900})
                request(session + "/url", {"url": site + "/"})
                wait("return document.querySelectorAll('#ray-tracing a.card').length===4")
                wait("const i=document.querySelector('#ray-tracing img');return i.complete&&i.naturalWidth>0")
                (screenshots / "root-desktop.png").write_bytes(base64.b64decode(request(session + "/screenshot")))
                wait("const v=document.querySelector('video.pipe-loop');return v&&v.readyState>=2&&v.videoWidth===640&&v.loop&&v.muted")
                wait("return document.querySelector('video.pipe-loop').currentTime>0.1")
                assert script("return document.querySelector('video.pipe-loop').currentSrc.endsWith('/flythrough.webm')"), "smallest animation must actually decode"
                script("document.querySelector('[data-pipe-open=pipe-home-dialog]').click()")
                wait("const d=document.getElementById('pipe-home-dialog');return d.open&&d.querySelector('video').videoWidth===640")
                script("document.getElementById('pipe-home-dialog').close()")
                request(session + "/url", {"url": site + "/ray-tracing/"})
                wait("const i=document.querySelector('#gpu-still img');return i&&i.complete&&i.naturalWidth>=1024")
                script("document.getElementById('gpu-still').scrollIntoView({block:'center'})")
                (screenshots / "gpu-desktop.png").write_bytes(base64.b64decode(request(session + "/screenshot")))
                assert script("return document.querySelectorAll('#image-pair a[download]').length") == 3
                script("document.getElementById('cpu-still').open=true")
                wait("const i=document.querySelector('#native-still img');return i&&i.complete&&i.naturalWidth>=1024")
                script("document.getElementById('native-still').scrollIntoView({block:'center'})")
                (screenshots / "native-desktop.png").write_bytes(base64.b64decode(request(session + "/screenshot")))
                script("document.querySelector('video').scrollIntoView({block:'center'});document.querySelector('video').play()")
                wait("const v=document.querySelector('video');return v.readyState>=2&&v.videoWidth>0")
                wait("return document.querySelector('video').currentTime>0.05")
                assert script("return document.querySelectorAll('.planned').length") == 1
                wait("const v=document.querySelector('#pipes video');return v&&v.readyState>=2&&v.currentTime>0.1")
                script("document.querySelector('[data-pipe-open=pipe-gallery-dialog]').click()")
                wait("const d=document.getElementById('pipe-gallery-dialog');return d.open&&d.querySelector('video').readyState>=2")
                script("const v=document.querySelector('#pipe-gallery-dialog video');v.currentTime=v.duration-0.15;v.play()")
                wait("const v=document.querySelector('#pipe-gallery-dialog video');return v.currentTime<1&&!v.paused")
                script("document.getElementById('pipe-gallery-dialog').close()")
                wait("const i=document.querySelector('#glass img');return i.complete&&i.naturalWidth===1024")

                script("document.getElementById('frame-inspector').open=true;const r=document.getElementById('frame');r.value=6;r.dispatchEvent(new Event('input',{bubbles:true}));")
                wait("const i=document.getElementById('frame-image');return i.complete&&i.currentSrc.endsWith('frame-006.png')")
                assert script("return document.getElementById('angle').value") == "90 degrees"
                assert script("return document.querySelector('video').paused")
                for label, width in [("desktop", 1100), ("mobile", 390)]:
                    request(session + "/window/rect", {"width": width, "height": 900})
                    assert script("return document.documentElement.scrollWidth<=innerWidth"), "horizontal overflow"
                    (screenshots / (label + ".png")).write_bytes(base64.b64decode(request(session + "/screenshot")))
                    script("document.getElementById('gpu-still').scrollIntoView({block:'center'})")
                    (screenshots / ("gpu-" + label + ".png")).write_bytes(base64.b64decode(request(session + "/screenshot")))
                    script("document.getElementById('native-still').scrollIntoView({block:'center'})")
                    (screenshots / ("native-" + label + ".png")).write_bytes(base64.b64decode(request(session + "/screenshot")))
                print("ray page: native still, four thumbnails and looping full-size pipe dialog, real video playback, frame selection and mobile layout pass")
    finally:
        if session:
            with suppress(Exception):
                request(session, method="DELETE")
        if driver:
            driver.terminate()
            driver.wait(timeout=10)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
