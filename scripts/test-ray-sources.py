#!/usr/bin/env python3
"""Check source navigation against the actual frozen recordings and emitted site."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent.parent


class Sources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.divs=[]
        self.sections=[]
        self.groups={}
        self.inside={}

    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag=="section":
            self.sections.append(a.get("id"))
        if tag=="div":
            group=a.get("data-recording", self.divs[-1] if self.divs else None)
            self.divs.append(group)
            if "data-recording" in a:
                assert group not in self.groups, group
                self.groups[group]=[]
                self.inside[group]=self.sections[-1] if self.sections else None
        if tag=="a" and self.divs and self.divs[-1]:
            self.groups[self.divs[-1]].append(a["href"])

    def handle_endtag(self, tag):
        if tag=="div" and self.divs:
            self.divs.pop()
        if tag=="section" and self.sections:
            self.sections.pop()


page=ROOT/"pages/ray-tracing"
s=Sources()
s.feed((page/"index.html").read_text())
expected={"pipes-rtx":["geometry.xtl","function.xtl"],"glass-rtx":["scene.xtl"],"logo-gpu":["scene.xtl"],"logo-still":["scene.xtl"],"image-pair":["scene.xtl"],"logo":["scene.xtl"]}
assert s.groups.keys()==expected.keys(), s.groups.keys()
assert s.inside["pipes-rtx"]=="pipes" and s.inside["glass-rtx"]=="glass"
for recording, files in expected.items():
    links=s.groups[recording]
    m=json.loads((ROOT/"site-assets/ray"/recording/"manifest.json").read_text())
    for file in files:
        url=f"media/{recording}/{file}"
        assert url in links
        assert (page/url).read_bytes()==(ROOT/"site-assets/ray"/recording/file).read_bytes()
        github=[x for x in links if re.fullmatch(r'https://github.com/softwarewrighter/X_eTaL-gpu/blob/[0-9a-f]{40}/site-assets/ray/'+re.escape(recording+'/'+file),x)]
        assert len(github)==1
        revision = github[0].split('/blob/')[1].split('/')[0]
        path = f"site-assets/ray/{recording}/{file}"
        frozen = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
        assert frozen == (ROOT/path).read_bytes(), "GitHub browse link must match recorded source bytes"
    assert any(f'/blob/{m["libraries"]}/libs/RayTrace/src/RayTrace.xtl' in x for x in links)
    assert 'https://softwarewrighter.github.io/X_eTaL-libraries/doc/libs-RayTrace-src-RayTrace.xtl.html' in links
    for url in links:
        if '/blob/main/examples/' in url:
            assert (ROOT/url.split('/blob/main/')[1]).is_file()
        if '/libs/' in url and '/blob/' in url:
            assert m["libraries"] in url
for name, text in [("home",(ROOT/"pages/index.html").read_text()),("gallery",(page/"index.html").read_text())]:
    dialog=re.search(r'<dialog id="pipe-[^"]+".*?</dialog>',text,re.S)[0]
    assert '/geometry.xtl' in dialog and '/function.xtl' in dialog and '#l.s_hade' in dialog, name
print('ray sources: six recorded source groups, matching public files, immutable GitHub snapshots, correct library pins, current xrefs and both dialogs pass')
