#!/usr/bin/env python3
"""Annotation au clavier de la couverture photo.

Label Studio est taillé pour les boîtes englobantes; pour cocher quatre cases
il impose un aller-retour serveur par image. Cet outil affiche l'image en plein
écran et n'attend que des touches: une photo prend deux à trois secondes, donc
500 images tiennent dans une demi-heure.

    python scripts/annotate_coverage.py --images_dir mes_photos \
        --out dataset/coverage_manual.csv

Raccourcis : 1 avant · 2 arrière · 3 gauche · 4 droite
             Entrée valider · ← revenir · S passer · A tout décocher

Avec --prefill, les cases arrivent pré-cochées par une source existante: on ne
corrige que ce qui est faux, ce qui va trois à cinq fois plus vite.

Le CSV est réécrit après chaque validation: fermer l'onglet ne perd rien, et
relancer la commande reprend là où vous vous étiez arrêté.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from claimsight.domain.taxonomy import COVERAGE_FACES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ClaimSight — annotation</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&family=IBM+Plex+Mono:wght@500&display=swap">
<style>
 :root{--bg:#16181C;--surface:#24272C;--line:#3A3E45;--txt:#F3F4F6;--dim:#9C9EA4;--on:#3979D4;--warn:#C54D47}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);
   font-family:Manrope,system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
 header{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;
   border-bottom:1px solid var(--line)}
 .mono{font-family:"IBM Plex Mono",monospace}
 #stage{flex:1;display:flex;align-items:center;justify-content:center;padding:16px;min-height:0}
 #img{max-width:100%;max-height:100%;object-fit:contain;border-radius:10px}
 footer{display:flex;gap:10px;padding:14px 20px;border-top:1px solid var(--line);justify-content:center}
 .face{flex:0 0 170px;padding:14px 10px;border:2px solid var(--line);border-radius:12px;
   background:var(--surface);text-align:center;cursor:pointer;user-select:none;transition:.12s}
 .face.on{border-color:var(--on);background:#1E3355}
 .face .k{display:block;font-size:11px;color:var(--dim);margin-bottom:4px}
 .face .n{font-size:15px;font-weight:600}
 #bar{height:3px;background:var(--line)} #fill{height:3px;background:var(--on);width:0}
 #done{display:none;text-align:center;padding:60px;font-size:20px}
 kbd{font-family:"IBM Plex Mono",monospace;font-size:11px;background:var(--surface);
   border:1px solid var(--line);border-radius:4px;padding:2px 6px}
</style></head><body>
<header>
  <div><strong>ClaimSight</strong> <span class="mono" style="color:var(--dim)" id="name"></span></div>
  <div class="mono" style="color:var(--dim)"><span id="pos"></span> ·
    <kbd>1-4</kbd> faces <kbd>↵</kbd> valider <kbd>←</kbd> retour <kbd>S</kbd> passer</div>
</header>
<div id="bar"><div id="fill"></div></div>
<div id="stage"><img id="img" alt=""><div id="done">Terminé — toutes les images sont annotées.</div></div>
<footer id="faces"></footer>
<script>
const FACES=__FACES__; let cur=null, sel=new Set();
const el=i=>document.getElementById(i);
function paint(){
  el('faces').innerHTML=FACES.map((f,i)=>
    `<div class="face ${sel.has(f)?'on':''}" data-f="${f}"><span class="k">${i+1}</span><span class="n">${f}</span></div>`).join('');
  [...document.querySelectorAll('.face')].forEach(d=>d.onclick=()=>{toggle(d.dataset.f)});
}
function toggle(f){ sel.has(f)?sel.delete(f):sel.add(f); paint(); }
async function load(dir){
  const r=await fetch('/next'+(dir?('?dir='+dir):'')); const d=await r.json();
  if(d.done){ el('img').style.display='none'; el('done').style.display='block'; el('faces').innerHTML=''; return; }
  cur=d; sel=new Set(d.faces||[]); el('img').src='/image?p='+encodeURIComponent(d.path);
  el('name').textContent=d.name; el('pos').textContent=d.index+' / '+d.total;
  el('fill').style.width=(100*d.index/d.total)+'%'; paint();
}
async function save(skip){
  if(!cur) return;
  await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:cur.path,faces:skip?null:[...sel]})});
  load('next');
}
document.onkeydown=e=>{
  const k=e.key.toLowerCase();
  if(k>='1'&&k<='4'){ const f=FACES[+k-1]; if(f) toggle(f); e.preventDefault(); }
  else if(k==='enter'){ save(false); }
  else if(k==='s'){ save(true); }
  else if(k==='arrowleft'){ load('prev'); }
  else if(k==='a'){ sel.clear(); paint(); }
};
load();
</script></body></html>"""


def _read_labels(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {
            row["image"]: [f for f in COVERAGE_FACES if row.get(f) == "1"]
            for row in csv.DictReader(fh)
        }


class Session:
    def __init__(self, images: list[Path], out: Path, prefill: Path | None = None) -> None:
        self.images = images
        self.out = out
        self.labels = _read_labels(out)
        self.i = 0

        if self.labels:
            print(f"{len(self.labels)} annotation(s) déjà présentes, reprise.")
            # On reprend à la première image non traitée.
            while self.i < len(self.images) and str(self.images[self.i]) in self.labels:
                self.i += 1
        elif prefill is not None:
            # Mode relecture: les cases arrivent pré-cochées par une source
            # existante (labels dérivés, prédictions du modèle). Corriger va
            # bien plus vite qu'étiqueter à blanc, et AUCUNE image n'est
            # sautée — c'est tout l'intérêt de la relecture.
            suggested = _read_labels(prefill)
            by_name = {Path(k).name: v for k, v in suggested.items()}
            for image in self.images:
                proposal = suggested.get(str(image)) or by_name.get(image.name)
                if proposal is not None:
                    self.labels[str(image)] = proposal
            print(f"Relecture: {len(self.labels)}/{len(self.images)} images pré-remplies "
                  f"depuis {prefill}. Corrigez ce qui est faux, validez le reste.")

    def current(self) -> dict:
        if self.i >= len(self.images):
            return {"done": True}
        p = self.images[self.i]
        return {"done": False, "path": str(p), "name": p.name,
                "index": self.i + 1, "total": len(self.images),
                "faces": self.labels.get(str(p), [])}

    def move(self, direction: str | None) -> None:
        if direction == "next":
            self.i = min(self.i + 1, len(self.images))
        elif direction == "prev":
            self.i = max(self.i - 1, 0)

    def record(self, path: str, faces: list[str] | None) -> None:
        if faces is None:
            self.labels.pop(path, None)          # « passer » efface toute valeur
        else:
            self.labels[path] = faces
        self.flush()

    def flush(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["image", *COVERAGE_FACES, "source"])
            w.writeheader()
            for image, faces in self.labels.items():
                w.writerow({"image": image, "source": "manual",
                            **{f: int(f in faces) for f in COVERAGE_FACES}})


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # silence
            pass

        def _send(self, code, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            u = urlparse(self.path)
            if u.path == "/":
                page = PAGE.replace("__FACES__", json.dumps(list(COVERAGE_FACES)))
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/next":
                session.move((parse_qs(u.query).get("dir") or [None])[0])
                self._send(200, json.dumps(session.current()).encode(), "application/json")
            elif u.path == "/image":
                raw = Path(parse_qs(u.query)["p"][0])
                # On ne sert que des images de la liste chargée.
                if raw.resolve() not in {p.resolve() for p in session.images}:
                    self._send(403, b"forbidden", "text/plain")
                    return
                self._send(200, raw.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"", "text/plain")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/save":
                self._send(404, b"", "text/plain")
                return
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            session.record(data["path"], data.get("faces"))
            self._send(200, b'{"ok":true}', "application/json")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out", default="dataset/coverage_manual.csv")
    ap.add_argument("--prefill", default=None,
                    help="CSV de départ (labels dérivés): les cases arrivent pré-cochées, "
                         "vous ne faites que corriger")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no_browser", action="store_true")
    args = ap.parse_args()

    root = Path(args.images_dir)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"Aucune image dans {root}")

    session = Session(images, Path(args.out),
                      Path(args.prefill) if args.prefill else None)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(session))
    url = f"http://127.0.0.1:{args.port}"

    print(f"{len(images)} image(s) — {len(session.labels)} déjà annotée(s)")
    print(f"Ouvrez {url}   (Ctrl+C pour arrêter, le CSV est écrit au fur et à mesure)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        session.flush()
        print(f"\n{len(session.labels)} annotation(s) enregistrée(s) -> {args.out}")


if __name__ == "__main__":
    main()
