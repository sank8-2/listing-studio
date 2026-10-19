# 🏷️ Listing Studio

**Turn a pile of item notes into marketplace-ready listings — offline, on your own machine.**

Listing Studio is a local-first listing writer for resellers. You describe the
item once; it writes a platform-optimised **title, description, tags and price
guidance** for Etsy, eBay, Depop or Vinted. Every model call goes to a local
[Ollama](https://ollama.com) daemon on `127.0.0.1` — your inventory, your
photos and your pricing never touch a cloud API, and there is no per-listing
cost.

```
item notes ──▶ local open-weight model ──▶ platform-shaped listing ──▶ CSV
                        (Ollama)                  (limits in Python)
```

## Why local

- **Nothing leaves the machine.** A reseller's stock list, supplier names and
  margins are commercially sensitive. Here they stay on the laptop.
- **No per-listing cost.** SaaS listing tools charge per listing or per month.
  Running your own model costs electricity.
- **Deterministic where it matters.** The model writes prose; Python enforces
  each marketplace's hard limits (Etsy: 140-char title, 13 tags × 20 chars;
  eBay: 80-char title, no tags). The model is never trusted with a character
  count.

## Quick start

```bash
# 1. a local model (any small instruct model works)
ollama pull llama3.2:1b

# 2. run it
python3 app.py            # → http://localhost:8004
```

Optional (photo → listing, needs a vision model):

```bash
ollama pull moondream
VISION_MODEL=moondream python3 app.py
```

Requirements: Python 3.10+ (stdlib only) and Ollama running locally.
No pip install, no build step, no account.

## How it works

1. **Item form** — name, category, colours, materials, condition and details.
   Whatever you type in *Details* is the only factual source the model is
   allowed to use; it is instructed never to invent measurements or brands.
2. **Generation** — the model returns strict JSON (`format: json`).
   `extract_json()` tolerates chattier replies, and the fallback composer
   guarantees a description even if the model returns nothing.
3. **Platform shaping** — `finalise()` clamps the title to the marketplace
   limit at a word boundary, normalises tags (lowercase, de-duplicated,
   length-capped, truncated at word boundaries) and orders the price band.
4. **History + export** — every listing is saved to SQLite and can be exported
   as a CSV ready to paste into a bulk uploader.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Ollama status + which models are installed |
| `GET /api/platforms` | platform limits, conditions, currency |
| `POST /api/generate` | `{platform, attributes, notes, image?, currency?}` → listing |
| `GET /api/history` | saved listings |
| `POST /api/listings/delete` | `{id}` |
| `GET /api/export.csv` | all listings as CSV |

### Configuration

| Env | Default | Meaning |
|---|---|---|
| `PORT` | `8004` | HTTP port |
| `HOST` | `127.0.0.1` | bind address |
| `OLLAMA` | `http://127.0.0.1:11434` | Ollama base URL |
| `WRITER_MODEL` | `llama3.2:1b` | the copywriting model |
| `VISION_MODEL` | `moondream` | optional photo model |
| `CURRENCY` | `GBP` | price currency |
| `LISTING_DB` | `~/.local/share/listing-studio.db` | SQLite database |

## Honest notes

- The photo path is wired up but **experimental**. On the machine this was
  built on, the only vision models that fit were unreliable, so the shipped
  default flow is text-first. Set `VISION_MODEL` to a capable VLM and the
  photo field starts contributing.
- A 1B model writes serviceable copy, not poetry. Bigger local models
  (`llama3.1:8b`, `qwen2.5:7b`) give noticeably better results — the code
  does not care which one you point it at.
- Price guidance is a sanity range, not an appraisal.

## License

MIT — see [LICENSE](LICENSE).
