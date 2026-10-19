#!/usr/bin/env python3
"""Listing Studio — a local-first marketplace listing writer.

Point it at a product photo; it writes the listing.

    photo ──▶ moondream (open-weight VLM, local)   → structured attributes
              qwen2.5   (open-weight LLM, local)   → title / copy / tags / price
              Python                                → hard formatting limits

Every model call goes to a local Ollama daemon on 127.0.0.1. Nothing —
not the photo, not the copy — leaves the machine. Stdlib only.

Env: PORT, HOST, OLLAMA, VISION_MODEL, WRITER_MODEL, LISTING_DB, CURRENCY.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8004"))
HOST = os.environ.get("HOST", "127.0.0.1")
OLLAMA = os.environ.get("OLLAMA", "http://127.0.0.1:11434")
VISION_MODEL = os.environ.get("VISION_MODEL", "moondream")
WRITER_MODEL = os.environ.get("WRITER_MODEL", "llama3.2:1b")
CURRENCY = os.environ.get("CURRENCY", "GBP")
DB = Path(os.environ.get("LISTING_DB", Path.home() / ".local" / "share" / "listing-studio.db"))

# Hard marketplace limits. These are the ONE thing the model never decides.
PLATFORMS = {
    "etsy": {"label": "Etsy", "title_max": 140, "tags": 13, "tag_max": 20,
             "tone": "warm, handmade, human; Etsy shoppers read the story"},
    "ebay": {"label": "eBay", "title_max": 80, "tags": 0, "tag_max": 0,
             "tone": "factual and keyword-dense; eBay shoppers filter by spec"},
    "depop": {"label": "Depop", "title_max": 65, "tags": 5, "tag_max": 30,
              "tone": "short, casual, style-led; Depop shoppers scroll fast"},
    "vinted": {"label": "Vinted", "title_max": 100, "tags": 5, "tag_max": 30,
               "tone": "plain, condition-first; Vinted shoppers care about wear"},
}
CONDITIONS = ["new", "like new", "good", "fair", "for parts"]

_db_lock = threading.Lock()


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------
def db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT NOT NULL,
            platform TEXT NOT NULL,
            item TEXT, condition TEXT, attributes TEXT,
            title TEXT, description TEXT, tags TEXT,
            price_low REAL, price_high REAL, currency TEXT,
            price_note TEXT, vision_model TEXT, writer_model TEXT,
            seconds REAL, source TEXT)""")


def save_listing(rec: dict) -> int:
    with _db_lock, db() as con:
        cur = con.execute(
            """INSERT INTO listings (created, platform, item, condition, attributes,
               title, description, tags, price_low, price_high, currency, price_note,
               vision_model, writer_model, seconds, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rec["created"], rec["platform"], rec.get("item"), rec.get("condition"),
             json.dumps(rec.get("attributes", {})), rec.get("title"),
             rec.get("description"), json.dumps(rec.get("tags", [])),
             rec.get("price_low"), rec.get("price_high"), rec.get("currency"),
             rec.get("price_note"), rec.get("vision_model"), rec.get("writer_model"),
             rec.get("seconds"), rec.get("source")))
        return cur.lastrowid


def list_listings(limit: int = 50) -> list:
    with db() as con:
        rows = con.execute("SELECT * FROM listings ORDER BY id DESC LIMIT ?",
                           (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["attributes"] = json.loads(d["attributes"] or "{}")
        d["tags"] = json.loads(d["tags"] or "[]")
        out.append(d)
    return out


def delete_listing(lid: int) -> None:
    with _db_lock, db() as con:
        con.execute("DELETE FROM listings WHERE id=?", (lid,))


# --------------------------------------------------------------------------
# ollama (local, keyless)
# --------------------------------------------------------------------------
def _post(path: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        OLLAMA + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ollama_models() -> list:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=8) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def health() -> dict:
    models = ollama_models()
    def has(m):
        return any(x == m or x.startswith(m + ":") or m.startswith(x.split(":")[0])
                   for x in models) if models else False
    return {"ollama": bool(models), "models": models,
            "vision_model": VISION_MODEL, "vision_ready": has(VISION_MODEL),
            "writer_model": WRITER_MODEL, "writer_ready": has(WRITER_MODEL)}


def see(image_b64: str, hint: str = "") -> str:
    """Ask the local VLM what is in the photo. Short questions work best."""
    q = ("Describe the main object in this photograph: what it is, its "
         "category, colour, material, and condition.")
    if hint:
        q += f" Context from the seller: {hint}."
    out = _post("/api/generate", {
        "model": VISION_MODEL, "prompt": q, "images": [image_b64],
        "stream": False, "options": {"temperature": 0.2, "num_predict": 160},
    }).get("response", "").strip()
    if not out:  # small VLMs intermittently return empty; retry once
        out = _post("/api/generate", {
            "model": VISION_MODEL, "prompt": "What is in this image?",
            "images": [image_b64], "stream": False,
            "options": {"temperature": 0.2, "num_predict": 120},
        }).get("response", "").strip()
    return out


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply, tolerating chatter."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("{")
    if start < 0:
        return {}
    depth, end = 0, -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return {}
    chunk = text[start:end]
    for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
        try:
            val = json.loads(candidate)
            return val if isinstance(val, dict) else {}
        except Exception:
            continue
    return {}


def write_listing(attrs: dict, platform: str, currency: str, notes: str = "") -> dict:
    cfg = PLATFORMS[platform]
    tag_rule = (f"- tags: exactly {cfg['tags']} tags, each at most "
                f"{cfg['tag_max']} characters, lowercase, no duplicates"
                if cfg["tags"] else "- tags: empty list")
    sys = ("You are a marketplace copywriter. You write listings that sell, "
           "and you never invent facts that are not in the provided details.")
    user = f"""Write a {cfg['label']} listing from these details.

OBJECT: {attrs.get('item', 'unknown item')}
CATEGORY: {attrs.get('category', 'unknown')}
COLOURS: {attrs.get('colors', '')}
MATERIALS: {attrs.get('materials', '')}
CONDITION: {attrs.get('condition', 'good')}
DETAILS: {attrs.get('details', '')}
SELLER NOTES: {notes or 'none'}

Tone for {cfg['label']}: {cfg['tone']}

Rules:
- title: at most {cfg['title_max']} characters, no emoji, put the strongest keyword first
{tag_rule}
- description: 70-110 words, plain language, no invented measurements or brand names
- price_low / price_high: a realistic second-hand range in {currency} as numbers only
- price_note: one sentence on why that range

Reply with JSON only:
{{"title": "...", "description": "...", "tags": [...], "price_low": 0, "price_high": 0, "price_note": "..."}}"""
    raw = _post("/api/chat", {
        "model": WRITER_MODEL,
        "messages": [{"role": "system", "content": sys},
                     {"role": "user", "content": user}],
        "stream": False, "format": "json",
        "options": {"temperature": 0.6, "num_predict": 480, "num_ctx": 2048},
    }).get("message", {}).get("content", "")
    return extract_json(raw)


# --------------------------------------------------------------------------
# deterministic post-processing — the model proposes, this decides
# --------------------------------------------------------------------------
def clamp_title(title: str, limit: int) -> str:
    title = re.sub(r"\s+", " ", (title or "").replace("✨", "").replace("🔥", "")).strip(" -|,")
    title = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", title).strip()
    if len(title) <= limit:
        return title
    cut = title[:limit]
    if " " in cut:
        cut = cut[:cut.rfind(" ")]
    return cut.strip(" -|,")


def norm_tags(tags, cfg: dict) -> list:
    if not cfg["tags"]:
        return []
    if isinstance(tags, str):  # small models often answer with "a, b, c"
        tags = re.split(r"[,\n;|]", tags)
    elif isinstance(tags, dict):
        tags = list(tags.keys()) + [str(v) for v in tags.values()]
    elif not isinstance(tags, (list, tuple)):
        tags = [tags] if tags else []
    seen, out = set(), []
    for t in tags:
        t = re.sub(r"[^\w\s&+-]", "", str(t)).strip().lower()
        t = re.sub(r"\s+", " ", t)[:cfg["tag_max"]].strip()
        if len(t) == cfg["tag_max"] and " " in t:
            cut = t[:t.rfind(" ")].strip()
            if len(cut) >= 6:
                t = cut
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= cfg["tags"]:
            break
    return out


def num(v):
    try:
        f = float(re.sub(r"[^\d.]", "", str(v)))
        return f if f > 0 else None
    except Exception:
        return None


def compose_desc(attrs: dict, cfg: dict, currency: str) -> str:
    """Deterministic fallback so a listing is never shipped with no copy."""
    item = (attrs.get("item") or "item").strip().rstrip(".")
    bits = []
    if attrs.get("materials"):
        bits.append(f"made of {attrs['materials']}")
    if attrs.get("colors"):
        bits.append(f"in {attrs['colors']}")
    cond = (attrs.get("condition") or "").strip()
    line = f"{item.capitalize()}"
    if bits:
        line += ", " + " and ".join(bits)
    line += "."
    if cond:
        line += f" Condition: {cond}."
    if attrs.get("details"):
        line += " " + attrs["details"].strip().rstrip(".") + "."
    line += f" Sold as described in the photos — happy to answer questions before you buy."
    return line


def finalise(raw: dict, attrs: dict, platform: str, currency: str) -> dict:
    cfg = PLATFORMS[platform]
    low, high = num(raw.get("price_low")), num(raw.get("price_high"))
    if low and high and low > high:
        low, high = high, low
    desc = (raw.get("description") or "").strip()
    if len(desc) < 40:  # small models sometimes return an empty description
        desc = compose_desc(attrs, cfg, currency)
    title = clamp_title(raw.get("title"), cfg["title_max"])
    if not title:
        title = clamp_title((attrs.get("item") or "item").capitalize(), cfg["title_max"])
    return {
        "platform": platform, "label": cfg["label"],
        "title": title,
        "description": desc,
        "tags": norm_tags(raw.get("tags"), cfg),
        "price_low": low, "price_high": high, "currency": currency,
        "price_note": (raw.get("price_note") or "").strip(),
        "limits": {"title_max": cfg["title_max"], "tags": cfg["tags"],
                   "tag_max": cfg["tag_max"]},
    }


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "ListingStudio/1.0"

    def log_message(self, fmt, *a):  # keep the console quiet
        pass

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            html = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html)
            return
        if p == "/api/health":
            return self._json(health())
        if p == "/api/platforms":
            return self._json({"platforms": PLATFORMS, "conditions": CONDITIONS,
                               "currency": CURRENCY})
        if p == "/api/history":
            return self._json({"listings": list_listings()})
        if p == "/api/export.csv":
            return self._csv()
        return self._json({"error": "not found"}, 404)

    def _csv(self):
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["created", "platform", "title", "description", "tags",
                    "price_low", "price_high", "currency", "condition"])
        for r in reversed(list_listings(1000)):
            w.writerow([r["created"], r["platform"], r["title"], r["description"],
                        ", ".join(r["tags"]), r["price_low"] or "", r["price_high"] or "",
                        r["currency"], r["condition"]])
        out = buf.getvalue().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="listings.csv"')
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        p = self.path.split("?")[0]
        body = self._body()

        if p == "/api/generate":
            t0 = time.time()
            platform = body.get("platform") or "etsy"
            if platform not in PLATFORMS:
                return self._json({"ok": False, "error": "unknown platform"}, 400)
            attrs = {k: (body.get("attributes") or {}).get(k, "")
                     for k in ("item", "category", "colors", "materials",
                               "condition", "details")}
            source = "manual"

            img = body.get("image") or ""
            if img:
                if "," in img[:80]:
                    img = img.split(",", 1)[1]
                try:
                    seen = see(img, attrs.get("item") or "")
                except urllib.error.URLError as e:
                    return self._json({"ok": False, "error": f"vision model: {e}"}, 502)
                source = "photo"
                attrs["vision"] = seen
                if not attrs.get("item"):
                    attrs["item"] = seen
                if not attrs.get("details"):
                    attrs["details"] = seen

            if not (attrs.get("item") or "").strip():
                return self._json({"ok": False,
                                   "error": "no photo and no item name given"}, 400)

            try:
                raw = write_listing(attrs, platform, body.get("currency") or CURRENCY,
                                    body.get("notes") or "")
            except urllib.error.URLError as e:
                return self._json({"ok": False, "error": f"writer model: {e}"}, 502)

            listing = finalise(raw, attrs, platform, body.get("currency") or CURRENCY)
            listing["attributes"] = attrs
            listing["seconds"] = round(time.time() - t0, 1)
            listing["source"] = source
            rec = dict(listing)
            rec["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
            rec["attributes"] = attrs
            rec["vision_model"] = VISION_MODEL if source == "photo" else None
            rec["writer_model"] = WRITER_MODEL
            try:
                listing["id"] = save_listing(rec)
            except Exception as e:
                listing["save_error"] = str(e)
            return self._json({"ok": True, "listing": listing})

        if p == "/api/listings/delete":
            try:
                delete_listing(int(body.get("id")))
            except Exception:
                pass
            return self._json({"ok": True})

        return self._json({"error": "not found"}, 404)


def main():
    init_db()
    h = health()
    print(f"Listing Studio → http://{'localhost' if HOST == '127.0.0.1' else HOST}:{PORT}")
    print(f"  ollama: {'up' if h['ollama'] else 'DOWN'}  models: {', '.join(h['models']) or '-'}")
    print(f"  vision: {VISION_MODEL} {'ok' if h['vision_ready'] else 'MISSING'}"
          f"   writer: {WRITER_MODEL} {'ok' if h['writer_ready'] else 'MISSING'}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
