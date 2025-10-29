# Micromonitor.py — llama.cpp + USDA local-first + schema-aligned
# Requires: pip install flask flask-limiter requests requests-oauthlib psycopg2-binary
#           sentence-transformers torch pillow llama-cpp-python
# Env you likely want:
#   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/micromonitor
#   USDA_ALLDATA_DIR=/path/to/FoodData_Central_csv_YYYY-MM-DD
#   USDA_API_KEY=<optional when local-first has hits>
#   FATSECRET_CONSUMER_KEY=...  FATSECRET_CONSUMER_SECRET=...
#   LLAMA_CPP_MODEL_PATH=~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
#   LLAMA_CTX=2048  LLAMA_GPU_LAYERS=0  LLAMA_N_THREADS=8
#   USE_USDA_LOCAL_FIRST=1

import os, io, re, csv, json, time, base64, datetime, concurrent.futures
from typing import Any, Dict, List, Tuple

import traceback
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_limiter import Limiter
from flask import abort
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image

import requests
from requests_oauthlib import OAuth1Session
import psycopg2, psycopg2.extras
import atexit

import torch
from sentence_transformers import SentenceTransformer


print("\n=== Micromonitor Environment Check ===")
print(f"USDA Local Dir: {os.getenv('USDA_ALLDATA_DIR', '(not set)')}")
print(f"USDA_API_KEY: {'✅ Set' if os.getenv('USDA_API_KEY') else '❌ Missing'}")
print(f"USE_USDA_LOCAL_FIRST: {os.getenv('USE_USDA_LOCAL_FIRST', '(default 1)')}")
print(f"FATSECRET_KEY: {'✅ Set' if os.getenv('FATSECRET_KEY') else '❌ Missing'}")
print(f"FATSECRET_SECRET: {'✅ Set' if os.getenv('FATSECRET_SECRET') else '❌ Missing'}")
print("=======================================\n")


# ---------- ENV ----------
DB_URL   = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/micromonitor")
USDA_KEY = os.getenv("USDA_API_KEY")
FS_KEY   = os.getenv("FATSECRET_CONSUMER_KEY")
FS_SECRET= os.getenv("FATSECRET_CONSUMER_SECRET")

# llama.cpp model (GGUF)
LLAMA_MODEL_PATH = os.getenv("LLAMA_CPP_MODEL_PATH", os.path.expanduser("~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))
LLAMA_CTX        = int(os.getenv("LLAMA_CTX", "2048"))
LLAMA_GPU_LAYERS = int(os.getenv("LLAMA_GPU_LAYERS", "0"))  # 0 = CPU only
LLAMA_N_THREADS  = int(os.getenv("LLAMA_N_THREADS", "8"))

# USDA AllData local-first
USDA_DIR = (os.getenv("USDA_ALLDATA_DIR", "").strip()
            or os.getenv("USDA_FOUNDATION_DIR", "").strip())
USE_USDA_LOCAL_FIRST = os.getenv("USE_USDA_LOCAL_FIRST", "1") == "1"

# ---------- APP ----------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
limiter = Limiter(get_remote_address, app=app)

# ---------- DB ----------
def db_conn():
    return psycopg2.connect(DB_URL)

def ensure_schema():
    """We only create app-internal tables: api_cache + documents (pgvector)."""
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
          cache_key   TEXT PRIMARY KEY,
          payload     JSONB NOT NULL,
          normalized  JSONB,
          updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
          ttl_seconds INTEGER NOT NULL DEFAULT 86400
        );""")
        cur.execute("CREATE INDEX IF NOT EXISTS api_cache_updated_idx ON api_cache (updated_at);")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
          id BIGSERIAL PRIMARY KEY,
          source_provider TEXT NOT NULL,
          source_id TEXT NOT NULL,
          title TEXT,
          content TEXT NOT NULL,
          embedding VECTOR(384) NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE (source_provider, source_id)
        );""")
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat
          ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """)
        conn.commit()
ensure_schema()

# ---------- CACHE ----------
def cache_get(cache_key: str) -> Tuple[Dict|None, Dict|None]:
    with db_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT payload, normalized, updated_at, ttl_seconds
              FROM api_cache WHERE cache_key=%s
        """, (cache_key,))
        row = cur.fetchone()
        if not row: return None, None
        age = (datetime.datetime.utcnow() - row["updated_at"].replace(tzinfo=None)).total_seconds()
        if age > row["ttl_seconds"]:
            return None, None
        return row["payload"], row["normalized"]

def cache_put(cache_key: str, payload: Dict, normalized: Dict|None, ttl_seconds: int = 86400):
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO api_cache (cache_key, payload, normalized, ttl_seconds)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (cache_key) DO UPDATE
            SET payload=EXCLUDED.payload,
                normalized=EXCLUDED.normalized,
                ttl_seconds=EXCLUDED.ttl_seconds,
                updated_at=now()
        """, (cache_key, json.dumps(payload), json.dumps(normalized) if normalized else None, ttl_seconds))

# ---------- NORMALIZERS ----------
def _num(v):
    try: return float(v)
    except: return None

# FatSecret
def norm_fs_search(resp: Dict) -> Dict:
    foods = resp.get("foods", {}).get("food", [])
    if isinstance(foods, dict): foods = [foods]
    out = []
    for f in foods:
        servings = f.get("serving_sizes", {}).get("serving", [])
        if isinstance(servings, dict): servings = [servings]
        first = servings[0] if servings else {}
        out.append({
            "food_id": int(f.get("food_id", -1)),
            "name": f.get("food_name"),
            "brand": f.get("brand_name"),
            "type": f.get("food_type"),
            "serving_desc": first.get("serving_description"),
            "provider": "fatsecret",
            "provider_ref": f.get("food_url"),
        })
    return {"results": out, "count": len(out)}

def norm_fs_detail(resp: Dict) -> Dict:
    food = resp.get("food", {})
    servings = food.get("servings", {}).get("serving", [])
    if isinstance(servings, dict): servings = [servings]
    s = servings[0] if servings else {}
    return {
      "food": {
        "food_id": int(food.get("food_id", -1)),
        "name": food.get("food_name"),
        "brand": food.get("brand_name"),
        "type": food.get("food_type"),
        "provider": "fatsecret",
        "provider_ref": food.get("food_url"),
      },
      "nutrition": {
        "serving_desc": s.get("serving_description"),
        "calories": _num(s.get("calories")),
        "protein_g": _num(s.get("protein")),
        "fat_g": _num(s.get("fat")),
        "carbs_g": _num(s.get("carbohydrate")),
        "fiber_g": _num(s.get("fiber")),
        "sugar_g": _num(s.get("sugar")),
        "sodium_mg": _num(s.get("sodium")),
      },
      "servings_count": len(servings)
    }

# USDA API normalizers
def _usda_label(food: Dict):
    return food.get("description") or food.get("brandName") or food.get("additionalDescriptions")

def _usda_nut_map(nutrients: List[Dict]|None) -> Dict:
    m = { n.get("nutrientName"): n for n in (nutrients or []) }
    def v(name): return (m.get(name) or {}).get("value")
    return {
        "calories":  v("Energy"),
        "protein_g": v("Protein"),
        "fat_g":     v("Total lipid (fat)"),
        "carbs_g":   v("Carbohydrate, by difference"),
        "fiber_g":   v("Fiber, total dietary"),
        "sugar_g":   v("Sugars, total including NLEA"),
        "sodium_mg": v("Sodium, Na"),
    }

def norm_usda_search(resp: Dict) -> Dict:
    foods = resp.get("foods", []) or []
    out = []
    for f in foods:
        out.append({
            "food_id": f.get("fdcId"),
            "name": _usda_label(f),
            "brand": f.get("brandName"),
            "type": f.get("dataType"),
            "serving_desc": None,
            "provider": "usda",
            "provider_ref": None,
        })
    return {"results": out, "count": len(out)}

def norm_usda_detail(food: Dict) -> Dict:
    basic = {
        "food_id": food.get("fdcId"),
        "name": _usda_label(food),
        "brand": food.get("brandName"),
        "type": food.get("dataType"),
        "provider": "usda",
        "provider_ref": None,
    }
    serving_desc = food.get("householdServingFullText") or (
        f'{food.get("servingSize")} {food.get("servingSizeUnit")}' if food.get("servingSize") else None
    )
    return {
      "food": basic,
      "nutrition": { **_usda_nut_map(food.get("foodNutrients")), "serving_desc": serving_desc },
      "servings_count": 1
    }

NORMALIZERS = {
    "fatsecret:foods.search": norm_fs_search,
    "fatsecret:food.get.v2": norm_fs_detail,
    "usda:foods.search": norm_usda_search,
    "usda:food.get": norm_usda_detail,
}

# ------ Health check to make sure flask matches html ---
@app.get("/healthz")
def healthz():
    return {"ok": True}, 200

# ---------- PROVIDERS (cache-aware) ----------
FATSECRET_BASE = "https://platform.fatsecret.com/rest/server.api"
USDA_BASE      = "https://api.nal.usda.gov/fdc/v1"

def _fs_client():
    return OAuth1Session(
        client_key=FS_KEY, client_secret=FS_SECRET,
        signature_method="HMAC-SHA1", signature_type="query"
    )

def fetch_fatsecret(method: str, params: Dict[str, Any], ttl=86400) -> Dict:
    key_items = "&".join(f"{k}={params[k]}" for k in sorted(params))
    ck = f"fatsecret:{method}:{key_items}"
    payload, normalized = cache_get(ck)
    if normalized: return normalized
    if payload:    return NORMALIZERS[f"fatsecret:{method}"](payload)

    q = {"method": method, "format": "json", "oauth_timestamp": str(int(time.time()))}
    q.update(params)
    r = _fs_client().get(FATSECRET_BASE, params=q, timeout=20)
    r.raise_for_status()
    data = r.json()
    norm = NORMALIZERS[f"fatsecret:{method}"](data)
    cache_put(ck, data, norm, ttl_seconds=ttl)
    return norm

def fetch_usda(path: str, params: Dict[str, Any], ttl=604800) -> Dict:
    key_items = "&".join(f"{k}={params[k]}" for k in sorted(params))
    method_key = "foods.search" if path == "foods/search" else "food.get"
    ck = f"usda:{method_key}:{key_items}"
    payload, normalized = cache_get(ck)
    if normalized: return normalized
    if payload:    return NORMALIZERS[f"usda:{method_key}"](payload)

    params = {"api_key": USDA_KEY, **params}
    r = requests.get(f"{USDA_BASE}/{path}", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    norm = NORMALIZERS[f"usda:{method_key}"](data)
    cache_put(ck, data, norm, ttl_seconds=ttl)
    return norm

def fs_search(q: str, limit=10, page=0):   return fetch_fatsecret("foods.search", {"search_expression": q, "max_results": limit, "page_number": page})
def fs_detail(food_id: int):               return fetch_fatsecret("food.get.v2", {"food_id": str(food_id)})
def usda_search_api(q: str, size=10, page=1):  return fetch_usda("foods/search", {"query": q, "pageSize": size, "pageNumber": page})
def usda_detail_api(fdc_id: int):              return fetch_usda(f"food/{fdc_id}", {})

# ---------- USDA LOCAL-FIRST (AllData CSVs) ----------
USDA_OPTIONAL: Dict[str, str] = {}
USDA_TRIPLET: Tuple[str, str, str] | None = None
USDA_ROOT: str | None = None
USDA_TABLES: Dict[str, str] = {}
_USDA_READY = False
_USDA_NAME_INDEX = None
_USDA_UPC_INDEX  = None
_USDA_NUTRIENT_MAP = None

def _triplet_and_optional(dirpath: str):
    p = lambda name: os.path.join(dirpath, name)
    food          = p("food.csv")
    nutrient      = p("nutrient.csv")
    food_nutrient = p("food_nutrient.csv")
    opt = {
        "foundation_food": p("foundation_food.csv"),
        "sr_legacy_food":  p("sr_legacy_food.csv"),
        "branded_food":    p("branded_food.csv"),
        "food_portion":    p("food_portion.csv"),
        "measure_unit":    p("measure_unit.csv"),
        "food_category":   p("food_category.csv"),
    }
    if all(os.path.isfile(x) for x in (food, nutrient, food_nutrient)):
        opt = {k: v for k, v in opt.items() if os.path.isfile(v)}
        return (food, nutrient, food_nutrient), opt
    return None, None

def resolve_usda_dir(base_dir: str):
    global USDA_OPTIONAL
    if not base_dir or not os.path.isdir(base_dir):
        return None, None
    trip, opt = _triplet_and_optional(base_dir)
    if trip: USDA_OPTIONAL = opt or {}; return base_dir, trip
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            trip, opt = _triplet_and_optional(entry.path)
            if trip: USDA_OPTIONAL = opt or {}; return entry.path, trip
    for root, dirs, files in os.walk(base_dir):
        trip, opt = _triplet_and_optional(root)
        if trip: USDA_OPTIONAL = opt or {}; return root, trip
    return None, None

def ensure_usda_loaded():
    """Connects to USDA CSVs; prints a friendly connected summary."""
    global USDA_TRIPLET, USDA_ROOT, USDA_TABLES, _USDA_READY
    if _USDA_READY: return
    if USDA_DIR:
        USDA_ROOT, USDA_TRIPLET = resolve_usda_dir(USDA_DIR)
    if not USDA_TRIPLET:
        print("❌ USDA local CSVs not found. Set USDA_ALLDATA_DIR to the folder that contains food.csv, nutrient.csv, food_nutrient.csv.")
        _USDA_READY = True
        return
    F_FOOD, F_NUTR, F_F2N = USDA_TRIPLET
    USDA_TABLES.clear()
    USDA_TABLES.update({
        "food": F_FOOD,
        "nutrient": F_NUTR,
        "food_nutrient": F_F2N,
        **USDA_OPTIONAL
    })
    # visual summary
    def _count(path):
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                return max(0, sum(1 for _ in f) - 1)
        except: return -1
    print("\n USDA Connection Summary")
    print(f"📂 Directory: {USDA_ROOT}")
    for k in ["food","nutrient","food_nutrient"]:
        pth = USDA_TABLES.get(k)
        if pth: print(f"   ✅ {k:<15} → {os.path.basename(pth)} ({_count(pth):,} rows)")
    others = [k for k in USDA_TABLES if k not in ("food","nutrient","food_nutrient")]
    if others:
        print("\n Additional files connected:")
        for k in sorted(others):
            print(f"   ✅ {k:<18} ({_count(USDA_TABLES[k]):,} rows)")
    print(f"\n✅ Successfully connected to {len(USDA_TABLES)} USDA CSV files.\n")
    _USDA_READY = True

def get_usda_csv_paths() -> dict:
    ensure_usda_loaded()
    return USDA_TABLES.copy()

def _ensure_local_indexes():
    """Build lightweight name/UPC and nutrient maps from CSVs (lazily)."""
    global _USDA_NAME_INDEX, _USDA_UPC_INDEX, _USDA_NUTRIENT_MAP
    ensure_usda_loaded()
    if not USDA_TRIPLET or _USDA_NAME_INDEX is not None:
        return
    _USDA_NAME_INDEX = []
    _USDA_UPC_INDEX  = {}
    def _rows(path):
        if not path or not os.path.isfile(path): return []
        with open(path, newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)
    # names from food.csv
    for r in _rows(USDA_TABLES.get("food")):
        nm = (r.get("description") or r.get("food_description") or "").strip()
        fid = r.get("fdc_id") or r.get("fdcId") or r.get("FDC_ID")
        if nm and fid:
            try: _USDA_NAME_INDEX.append((nm.lower(), int(fid), "USDA-food", r))
            except: pass
    # foundation/sr/branded
    for key, label in (("foundation_food","USDA-foundation"),
                       ("sr_legacy_food","USDA-sr_legacy"),
                       ("branded_food","USDA-branded")):
        p = USDA_TABLES.get(key)
        if not p: continue
        for r in _rows(p):
            nm = (r.get("description") or r.get("long_description") or r.get("brand_name") or "").strip()
            fid = r.get("fdc_id") or r.get("fdcId") or r.get("FDC_ID")
            if nm and fid:
                try: _USDA_NAME_INDEX.append((nm.lower(), int(fid), label, r))
                except: pass
            upc = (r.get("gtin_upc") or r.get("gtin") or r.get("upc") or "").strip()
            if upc and fid:
                key_upc = re.sub(r"\D","",upc)
                try: _USDA_UPC_INDEX[key_upc] = int(fid)
                except: pass
    # nutrient dictionary
    _USDA_NUTRIENT_MAP = {}
    for r in _rows(USDA_TABLES.get("nutrient")):
        nid = r.get("id") or r.get("nutrient_id")
        if nid:
            _USDA_NUTRIENT_MAP[str(nid)] = {
                "nutrient_id": str(nid),
                "name": r.get("name") or r.get("nutrient_name"),
                "unit_name": r.get("unit_name") or r.get("unitName"),
            }

def usda_local_search(q: str, limit=10):
    ensure_usda_loaded()
    if not USDA_TRIPLET: return {"results": [], "count": 0}
    _ensure_local_indexes()
    ql = q.strip().lower()
    hits = []
    for nm, fid, source, row in _USDA_NAME_INDEX:
        if ql in nm:
            hits.append({
                "food_id": fid,
                "name": row.get("description") or row.get("long_description") or row.get("brand_name") or "",
                "brand": row.get("brand_owner") or row.get("brand_name") or "",
                "type": source.replace("USDA-",""),
                "serving_desc": None,
                "provider": "usda",
                "provider_ref": None,
                "_watermark":"USDA_LOCAL"
            })
            if len(hits) >= limit: break
    return {"results": hits, "count": len(hits)}

def usda_local_detail(fdc_id: int):
    ensure_usda_loaded()
    if not USDA_TRIPLET: return None
    _ensure_local_indexes()
    fn = USDA_TABLES.get("food_nutrient")
    if not fn: return None
    out = []
    with open(fn, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                if int(r.get("fdc_id", -1)) != int(fdc_id): continue
            except: continue
            nid = str(r.get("nutrient_id"))
            nutr = _USDA_NUTRIENT_MAP.get(nid, {"name": None, "unit_name": None})
            out.append({
                "fdc_id": int(fdc_id),
                "nutrient_id": int(nid) if nid.isdigit() else nid,
                "name": nutr.get("name"),
                "unit_name": nutr.get("unit_name"),
                "amount": float(r.get("amount") or 0.0),
            })
    return {
      "food": {"food_id": int(fdc_id), "name": f"FDC {fdc_id}", "brand": None, "type": "USDA_LOCAL", "provider":"usda"},
      "nutrition": {"serving_desc": None},
      "servings_count": 1,
      "_watermark": "USDA_LOCAL",
      "foodNutrients": out
    }

#Enrich for due to html changes?
# Utilities we rely on here:
# - USDA_TRIPLET: tuple(path_food_csv, path_nutrient_csv, path_food_nutrient_csv)
# - USDA_OPTIONAL: dict with optional paths (e.g., 'measure_unit', etc.)
# - db_conn(): your existing DB connection helper (or adapt _get_db_conn from earlier patch)

def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row

def _usda_local_search_by_name(name: str, limit: int = 5):
    """Very simple case-insensitive substring search on food.csv."""
    if not USDA_TRIPLET:
        return []
    food_csv, _, _ = USDA_TRIPLET
    q = name.casefold()
    hits = []
    for row in _read_csv_rows(food_csv):
        desc = (row.get("description") or "").casefold()
        if q in desc:
            try:
                hits.append({
                    "food_id": int(row.get("fdc_id")),
                    "name": row.get("description"),
                    "data_type": row.get("data_type")
                })
            except Exception:
                continue
            if len(hits) >= limit:
                break
    return hits

def _usda_local_detail(fdc_id: int):
    """Build micronutrients from nutrient.csv + food_nutrient.csv for a given fdc_id."""
    if not USDA_TRIPLET:
        return {"micronutrients": []}
    food_csv, nutrient_csv, food_nutrient_csv = USDA_TRIPLET

    # Map nutrient_id -> (name, unit)
    nutr_meta = {}
    for row in _read_csv_rows(nutrient_csv):
        try:
            nid = int(row.get("id"))
        except Exception:
            continue
        nutr_meta[nid] = (row.get("name"), row.get("unit_name"))

    micros = []
    for row in _read_csv_rows(food_nutrient_csv):
        try:
            if int(row.get("fdc_id")) != fdc_id:
                continue
            nid = int(row.get("nutrient_id"))
            amount = row.get("amount")
            if amount is None:
                continue
            try:
                amount = float(amount)
            except Exception:
                # keep as string if needed
                pass
            name, unit = nutr_meta.get(nid, (None, None))
            if not name:
                continue
            micros.append({"name": name, "amount": amount, "unit": unit})
        except Exception:
            continue
    return {"micronutrients": micros}

def _usda_api_search_and_detail(name: str, api_key: str):
    """Fallback to USDA API search -> first result -> detail."""
    if not api_key:
        return {"micronutrients": []}
    try:
        # Search
        s = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"api_key": api_key, "query": name, "pageSize": 1},
            timeout=10
        )
        if s.status_code != 200:
            return {"error": f"search http {s.status_code}", "micronutrients": []}
        data = s.json() or {}
        foods = data.get("foods") or []
        if not foods:
            return {"micronutrients": []}
        fdc_id = foods[0].get("fdcId")
        if not fdc_id:
            return {"micronutrients": []}

        # Detail
        d = requests.get(
            f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}",
            params={"api_key": api_key},
            timeout=10
        )
        if d.status_code != 200:
            return {"error": f"detail http {d.status_code}", "micronutrients": []}
        dj = d.json() or {}
        fn = dj.get("foodNutrients") or []
        micros = []
        for n in fn:
            nut = n.get("nutrient") or {}
            name = nut.get("name")
            unit = nut.get("unitName")
            amount = n.get("amount")
            if name is None:
                continue
            micros.append({"name": name, "amount": amount, "unit": unit})
        return {"micronutrients": micros, "fdc_id": fdc_id}
    except Exception as e:
        return {"error": f"api failure: {e}", "micronutrients": []}

@app.get("/api/usda_enrich")
def api_usda_enrich():
    """
    /api/usda_enrich?name=Banana
    /api/usda_enrich?name=Banana&food_id=123  -> will persist micros for that saved food
    """
    name = (request.args.get("name") or "").strip()
    food_id = request.args.get("food_id")
    if not name:
        return jsonify({"error": "name required"}), 400

    try:
        # 1) LOCAL CSV FIRST
        hits = _usda_local_search_by_name(name, limit=1)
        micros = []
        used = "USDA_LOCAL"
        fdc_id = None

        if hits:
            fdc_id = hits[0]["food_id"]
            detail = _usda_local_detail(fdc_id)
            micros = detail.get("micronutrients") or []

        # 2) API FALLBACK (if nothing found locally)
        if not micros:
            used = "USDA_API"
            api_key = os.getenv("USDA_API_KEY", "")
            res = _usda_api_search_and_detail(name, api_key)
            micros = res.get("micronutrients") or []
            fdc_id = res.get("fdc_id")

        # 3) optionally persist to DB for this food_id
        saved = False
        if food_id and micros:
            try:
                with db_conn() as conn, conn.cursor() as cur:
                    cur.execute("DELETE FROM public.micronutrients WHERE food_id=%s", (food_id,))
                    for m in micros:
                        if not m.get("name"):
                            continue
                        cur.execute("""
                            INSERT INTO public.micronutrients (food_id, nutrient_name, amount, unit)
                            VALUES (%s,%s,%s,%s)
                        """, (food_id, m.get("name"), m.get("amount"), m.get("unit")))
                    conn.commit()
                    saved = True
            except Exception as e:
                # non-fatal; include info
                return jsonify({
                    "ok": True, "source": used, "fdc_id": fdc_id,
                    "nutrients": len(micros),
                    "save_warning": str(e)
                }), 200

        return jsonify({
            "ok": True, "source": used, "fdc_id": fdc_id,
            "nutrients": len(micros), "saved": saved
        }), 200

    except Exception as e:
        # show traceback in server logs; return a helpful error JSON to client
        traceback.print_exc()
        return jsonify({"error": f"enrichment failed: {e}"}), 500


# ---------- EMBEDDINGS / RAG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EMB_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEVICE)

def to_pgvector(vec: List[float]) -> str:
    return "[" + ", ".join(f"{float(x):.7f}" for x in vec) + "]"

def embed_texts(texts: List[str]) -> List[List[float]]:
    arr = _EMB_MODEL.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    return arr.tolist()

def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]

MACRO_KEYS = ("calories","protein_g","fat_g","carbs_g","fiber_g","sugar_g","sodium_mg")
def _mk_summary(detail_norm: Dict) -> str:
    food = detail_norm["food"]; nut = detail_norm.get("nutrition", {})
    parts = [f"{food.get('name','')}"]
    if food.get("brand"): parts.append(f"brand: {food['brand']}")
    if food.get("type"):  parts.append(f"type: {food['type']}")
    if nut.get("serving_desc"): parts.append(f"serving: {nut['serving_desc']}")
    macro = ", ".join([f"{k.replace('_',' ')}: {nut[k]}" for k in MACRO_KEYS if nut.get(k) is not None])
    if macro: parts.append(macro)
    return " | ".join(parts)

def upsert_document(provider: str, source_id: str, title: str, content: str, emb: List[float]):
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("""
          INSERT INTO documents (source_provider, source_id, title, content, embedding)
          VALUES (%s,%s,%s,%s,%s::vector)
          ON CONFLICT (source_provider, source_id) DO UPDATE
          SET title=EXCLUDED.title, content=EXCLUDED.content, embedding=EXCLUDED.embedding
        """, (provider, source_id, title, content, to_pgvector(emb)))

def ingest_detail(provider: str, detail_norm: Dict):
    food = detail_norm["food"]
    content = _mk_summary(detail_norm)
    emb = embed_text(content)
    upsert_document(provider, str(food["food_id"]), food["name"], content, emb)

def search_docs(query: str, k: int = 5) -> List[Dict]:
    qvec = embed_text(query)
    with db_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
          SELECT source_provider, source_id, title, content,
                 1 - (embedding <=> %s::vector) AS score
            FROM documents
        ORDER BY embedding <=> %s::vector
           LIMIT %s
        """, (to_pgvector(qvec), to_pgvector(qvec), k))
        return cur.fetchall()

# ---------- llama.cpp (local LLM) ----------
try:
    from llama_cpp import Llama
    _LLM = Llama(model_path=LLAMA_MODEL_PATH,
                 n_ctx=LLAMA_CTX, n_threads=LLAMA_N_THREADS,
                 n_gpu_layers=LLAMA_GPU_LAYERS, chat_format="llama-3")
except Exception as e:
    _LLM = None
    print(f"[warn] llama.cpp model init failed: {e}")

def ask_llamacpp(question: str, contexts: List[str], image_bytes: bytes|None = None, mime="image/png") -> str:
    if not _LLM:
        return "LLM not available."
    ctx = "\n".join(f"- {c}" for c in contexts) if contexts else "None"
    system = "You are a helpful nutrition assistant. Use the provided context when available and be concise."
    user = f"Use the context to answer factually.\n\nContext:\n{ctx}\n\nQ: {question}\nA:"
    msgs = [{"role":"system","content":system},{"role":"user","content":user}]
    out = _LLM.create_chat_completion(messages=msgs, temperature=0.2, max_tokens=700)
    return out["choices"][0]["message"]["content"]

# ---------- Map normalized detail ➜ your schema (foods + micronutrients) ----------
MACRO_KEYS_UNITS = {
    "calories":    ("Energy (kcal)", "kcal"),
    "protein_g":   ("Protein", "g"),
    "fat_g":       ("Total lipid (fat)", "g"),
    "carbs_g":     ("Carbohydrate, by difference", "g"),
    "fiber_g":     ("Fiber, total dietary", "g"),
    "sugar_g":     ("Sugars, total including NLEA", "g"),
    "sodium_mg":   ("Sodium, Na", "mg"),
}

def _extract_calories(detail_norm: dict) -> int | None:
    v = (detail_norm.get("nutrition") or {}).get("calories")
    try: return int(round(float(v)))
    except: return None

def save_food_and_micros(detail_norm: dict, user_id: int | None = None, quantity: float = 1.0) -> int:
    """
    Insert into public.foods + public.micronutrients (matches your schema).
    Returns the created foods.id
    """
    food = detail_norm.get("food", {})
    name = food.get("name") or "Unnamed food"
    cals = _extract_calories(detail_norm)

    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO public.foods (user_id, food_name, calories, quantity)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (user_id, name, cals, float(quantity)))
        food_id = cur.fetchone()[0]

        # macros
        nut = detail_norm.get("nutrition") or {}
        for key, (friendly, unit) in MACRO_KEYS_UNITS.items():
            val = nut.get(key)
            if val is None: continue
            try:
                cur.execute("""
                    INSERT INTO public.micronutrients (food_id, nutrient_name, amount, unit)
                    VALUES (%s, %s, %s, %s)
                """, (food_id, friendly, float(val), unit))
            except Exception:
                pass

        # detailed list if present (USDA API/local)
        extra = detail_norm.get("foodNutrients") or detail_norm.get("micronutrients") or []
        for item in extra:
            name2 = item.get("name") or item.get("nutrientName") or item.get("nutrient_name")
            unit2 = item.get("unit_name") or item.get("unitName") or item.get("unit") or ""
            amt2  = item.get("amount")
            try:
                if name2 and amt2 is not None:
                    cur.execute("""
                        INSERT INTO public.micronutrients (food_id, nutrient_name, amount, unit)
                        VALUES (%s, %s, %s, %s)
                    """, (food_id, name2, float(amt2), unit2))
            except Exception:
                pass

        conn.commit()
        return food_id

# ---------- Convenience provider queries (USDA local-first) ----------
def query_fatsecret(query: str):
    try:
        data = fs_search(query, limit=10, page=0)
        return {"data": data, "source": "FatSecret"}
    except Exception as e:
        return {"error": f"FatSecret error: {e}"}

def query_usda(query: str):
    # try local first
    if USE_USDA_LOCAL_FIRST:
        try:
            local = usda_local_search(query, limit=10)
            if local and local.get("count",0) > 0:
                return {"data": local, "source": "USDA_LOCAL"}
        except Exception as e:
            print("[usda local] warn:", e)
    # API fallback
    try:
        data = usda_search_api(query, size=10, page=1)
        return {"data": data, "source": "USDA_API"}
    except Exception as e:
        return {"error": f"USDA error: {e}"}

# ---------- Auth ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    username = request.form.get('username')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    sex = request.form.get('sex')
    age = request.form.get('age')
    weight = request.form.get('weight')
    height = request.form.get('height')
    activity_level = request.form.get('activity_level')
    if not username or not password or not confirm_password:
        return "All fields are required.", 400
    if len(password) < 6:
        return "Password must be at least 6 characters.", 400
    if password != confirm_password:
        return "Passwords do not match.", 400
    hashed_password = generate_password_hash(password)
    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.users (username, password_hash, sex, age, weight, height, activity_level)
                VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id
            """, (
                username, hashed_password,
                sex or None,
                int(age) if age else None,
                float(weight) if weight else None,
                float(height) if height else None,
                activity_level or None
            ))
            user_id = cur.fetchone()[0]
            conn.commit()
        session['user_id'] = user_id
        session['username'] = username
        return redirect(url_for('home'))
    except Exception as e:
        return f"Error creating account: {str(e)}", 400

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    username = request.form.get('username')
    password = request.form.get('password')
    if not username or not password:
        return "Username and password are required.", 400
    try:
        with db_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM public.users WHERE username = %s", (username,))
            user = cur.fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('home'))
        else:
            return "Invalid username or password.", 401
    except Exception as e:
        return f"Error logging in: {str(e)}", 400

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/current_user', methods=['GET'])
def current_user():
    if 'user_id' in session:
        return jsonify({"user_id": session['user_id'], "username": session['username']})
    else:
        return jsonify({"error": "Not logged in"}), 401

# ---------- Pages ----------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/my_foods_page')
def my_foods_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('my_foods.html')

# ---------- Food search/aggregate ----------
@app.get("/search_all/<query>")
def search_all(query: str):
    query = (query or "").strip()
    resp = {"fatsecret": None, "usda": None}

    # ---------- FatSecret ----------
    try:
        # fs_search() should already be available in your file.
        # It should return something like:
        # { "results": [ { "food_id":..., "name":..., "brand":...}, ... ],
        #   "count": N }
        fs_norm = fs_search(query, limit=10, page=0)
        fs_results_raw = fs_norm.get("results", []) if fs_norm else []

        # Force each result to match the front-end expectation
        fatsecret_results = []
        for item in fs_results_raw:
            fatsecret_results.append({
                "food_id": item.get("food_id") or item.get("id"),
                "name": item.get("name") or item.get("food_name") or "",
                "brand": item.get("brand") or item.get("brand_name") or item.get("brandOwner") or ""
            })

        resp["fatsecret"] = {
            "data": { "results": fatsecret_results },
            "error": None
        }
    except Exception as e:
        resp["fatsecret"] = {
            "data": { "results": [] },
            "error": f"FatSecret error: {e}"
        }

    # ---------- USDA (local first, API fallback) ----------
    try:
        usda_results_all = []
        usda_err = None
        usda_source = None

        # 1. Try local CSV search first if enabled
        if USE_USDA_LOCAL_FIRST:
            try:
                # usda_local_search() should return:
                # { "results": [ { ... }, ... ], "count": N }
                local_norm = usda_local_search(query, limit=10)
                local_raw = local_norm.get("results", []) if local_norm else []

                if local_raw:
                    usda_source = "USDA_LOCAL"
                    for item in local_raw:
                        usda_results_all.append({
                            "food_id": (
                                item.get("food_id")
                                or item.get("fdc_id")
                                or item.get("fdcId")
                                or item.get("id")
                            ),
                            "name": (
                                item.get("name")
                                or item.get("food_name")
                                or item.get("description")
                                or ""
                            ),
                            "brand": (
                                item.get("brand")
                                or item.get("brand_name")
                                or item.get("brandOwner")
                                or ""
                            )
                        })
            except Exception as ee:
                usda_err = f"local USDA error: {ee}"

        # 2. If we didn't get anything local, try API
        if not usda_results_all:
            try:
                api_norm = usda_search_api(query, size=10, page=1)  # you already have this helper
                api_raw = api_norm.get("results", []) if api_norm else []

                if api_raw:
                    usda_source = "USDA_API"
                    for item in api_raw:
                        usda_results_all.append({
                            "food_id": (
                                item.get("food_id")
                                or item.get("fdc_id")
                                or item.get("fdcId")
                                or item.get("id")
                            ),
                            "name": (
                                item.get("name")
                                or item.get("food_name")
                                or item.get("description")
                                or ""
                            ),
                            "brand": (
                                item.get("brand")
                                or item.get("brand_name")
                                or item.get("brandOwner")
                                or ""
                            )
                        })
            except Exception as ee2:
                if usda_err:
                    usda_err += f" | api USDA error: {ee2}"
                else:
                    usda_err = f"api USDA error: {ee2}"

        # Fall back label if we never set it
        if not usda_source:
            usda_source = "USDA_LOCAL"

        resp["usda"] = {
            "source": usda_source,
            "data": { "results": usda_results_all },
            "error": usda_err
        }

    except Exception as e:
        resp["usda"] = {
            "source": "USDA_LOCAL",
            "data": { "results": [] },
            "error": str(e)
        }

    return jsonify(resp)


@app.get("/api/search_aggregate")
def api_search_aggregate():
    q = request.args.get("q","").strip()
    n = int(request.args.get("n", 5))
    which = request.args.get("provider","both")  # "usda" | "fatsecret" | "both"
    use_usda = which in ("both","usda")
    use_fs   = which in ("both","fatsecret")
    results = []
    if use_fs:   results += fs_search(q, limit=n, page=0)["results"]
    if use_usda:
        local = usda_local_search(q, limit=n) if USE_USDA_LOCAL_FIRST else {"results":[]}
        if local.get("results"): results += local["results"]
        else: results += usda_search_api(q, size=n, page=1)["results"]
    return jsonify({"query": q, "results": results})

# ----- Normalizer for USDA details ------
def usda_api_detail(fdc_id: int, api_key: str) -> dict:
    """Fetch a USDA food and normalize name + calories + micronutrients."""
    if not api_key:
        raise RuntimeError("USDA_API_KEY not set")

    r = requests.get(
        f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}",
        params={"api_key": api_key},
        timeout=12,
    )
    r.raise_for_status()
    j = r.json() or {}

    # --- Name / brand / ingredients ---
    desc  = (j.get("description") or "").strip()
    brand = (j.get("brandOwner") or "").strip()
    ingred = (j.get("ingredients") or "").strip()
    pretty_name = f"{desc} ({brand})" if (desc and brand) else (desc or brand or f"FDC {fdc_id}")

    # --- Calories (kcal) ---
    kcal = None

    # 1) labelNutrients.calories.value (preferred for branded foods)
    ln = j.get("labelNutrients") or {}
    if isinstance(ln, dict):
        cal_block = ln.get("calories") or ln.get("energy")
        if isinstance(cal_block, dict) and cal_block.get("value") is not None:
            kcal = cal_block.get("value")

    # 2) scan foodNutrients for Energy (kcal / kJ)
    if kcal is None:
        for n in j.get("foodNutrients") or []:
            nut  = n.get("nutrient") or {}
            name = (nut.get("name") or "").lower()
            unit = (nut.get("unitName") or "").lower()
            amt  = n.get("amount")

            if name == "energy" and unit == "kcal" and amt is not None:
                kcal = amt
                break
            if name == "calories" and unit in ("kcal", "calories") and amt is not None:
                kcal = amt
                break
            if name == "energy" and unit in ("kj", "kilojoules") and amt is not None:
                try:
                    kcal = float(amt) / 4.184
                except Exception:
                    pass
                break

    # Nicely round kcal if float
    if isinstance(kcal, float):
        try:
            if kcal == 0:
                kcal = 0
            else:
                sig = 4
                p = int(math.floor(math.log10(abs(kcal))))
                kcal = round(kcal, sig - p - 1)
        except Exception:
            pass

    # --- Micronutrients list ---
    micros = []
    for n in j.get("foodNutrients") or []:
        nut  = n.get("nutrient") or {}
        nm   = nut.get("name")
        unit = nut.get("unitName")
        amt  = n.get("amount")
        if nm is None:
            continue
        micros.append({"name": nm, "amount": amt, "unit": unit})

    return {
        "food": {"name": pretty_name, "ingredients": ingred},
        "nutrition": {"calories": kcal},
        "foodNutrients": micros,       # your UI also accepts 'micronutrients'
        "provider": "USDA_API",
        "fdc_id": fdc_id,
    }


# ---------- Food detail (USDA local-first) + save to your schema ----------
@app.get("/api/food_detail")
def api_food_detail():
    """
    Preview or save a food from a provider.

    Preview (no DB write):
      GET /api/food_detail?provider=usda&id=<fdcId>&preview=1
      GET /api/food_detail?provider=fatsecret&id=<id>&preview=1

    Save (DB write):
      GET /api/food_detail?provider=usda&id=<fdcId>
      GET /api/food_detail?provider=fatsecret&id=<id>
    """
    prov    = (request.args.get("provider") or "").lower()
    _id     = (request.args.get("id") or "").strip()
    preview = (request.args.get("preview") or "0") in ("1", "true", "yes")
    
    if not prov or not _id:
        return jsonify({"error": "provider and id are required"}), 400

    # If saving, require login; preview can be anonymous
    if not preview:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Authentication required"}), 401
    else:
        user_id = session.get("user_id")  # may be None

    try:
        if prov == "usda":
            fdc_id = int(_id)
            detail = usda_api_detail(fdc_id, os.getenv("USDA_API_KEY",""))

            if preview:
                return jsonify({"detail": detail, "provider": "usda", "id": fdc_id, "preview": True})

            name = (detail.get("food") or {}).get("name") or "Food"
            ingred = (detail.get("food") or {}).get("ingredients") or ""
            calories = (detail.get("nutrition") or {}).get("calories")
            micros = detail.get("foodNutrients") or detail.get("micronutrients") or []

            new_id = save_food_with_micros(
                user_id=user_id,
                name=name,
                calories=calories,
                quantity=1.0,
                ingredients=ingred,
                micros=micros
            )
            return jsonify({"saved_food_id": new_id, "detail": detail})

        elif prov == "fatsecret":
            # --- your existing fatsecret detail fetch here ---
            # Must return a dict 'detail' with same keys you display in the UI
            # For preview:
            #   if preview: return jsonify({"detail": detail, "provider":"fatsecret", "id": _id, "preview": True})
            # For save:
            #   new_id = save_food_with_micros(user_id, name, calories, 1.0, ingredients, micros)
            #   return jsonify({"saved_food_id": new_id, "detail": detail})
            return jsonify({"error": "fatsecret detail not yet implemented in this snippet"}), 501

        else:
            return jsonify({"error": f"Unknown provider '{prov}'"}), 400

    except requests.HTTPError as e:
        return jsonify({"error": f"HTTP {e.response.status_code} from provider"}), 502
    except Exception as e:
        return jsonify({"error": f"detail failed: {e}"}), 500

# --- Save a food + micronutrients ------------------------------------------------
def save_food_with_micros(user_id: int, name: str, calories, quantity: float,
                          ingredients: str, micros: list) -> int:
    """
    Inserts a row into public.foods and related rows into public.micronutrients.
    Returns the new foods.id.
    """
    # calories: int or None
    cal_val = int(calories) if (calories is not None and str(calories).strip() != "") else None

    with db_conn() as conn, conn.cursor() as cur:
        # Insert into foods (schema you provided)
        cur.execute(
            """
            INSERT INTO public.foods (user_id, food_name, calories, quantity)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (user_id, name, cal_val, float(quantity or 1.0)),
        )
        food_id = cur.fetchone()[0]

        # Optionally store ingredients somewhere (only if you added a column for it).
        # If you created a 'ingredients' column in foods, uncomment the next block:
        # cur.execute("UPDATE public.foods SET ingredients=%s WHERE id=%s", (ingredients, food_id))

        # Insert micronutrients (name, amount, unit)
        if micros:
            for m in micros:
                nm   = (m.get("name") or "").strip()
                amt  = m.get("amount")
                unit = m.get("unit")
                if not nm:
                    continue
                cur.execute(
                    """
                    INSERT INTO public.micronutrients (food_id, nutrient_name, amount, unit)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (food_id, nm, amt, unit),
                )
        conn.commit()
    return food_id


# ---- Nutrition calculator --------- --
@app.get("/calculate_nutrition")
def calculate_nutrition():
    """Simple Mifflin–St Jeor calorie calculator."""
    try:
        age = float(request.args.get("age", 0))
        sex = request.args.get("sex", "male").lower()
        weight = float(request.args.get("weight", 0))  # kg
        height = float(request.args.get("height", 0))  # cm
        activity = request.args.get("activity_level", "moderate")

        # BMR (Mifflin–St Jeor)
        if sex.startswith("m"):
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        # Activity multiplier
        factors = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
        }
        tdee = bmr * factors.get(activity, 1.55)
        return jsonify({
            "bmr": round(bmr, 1),
            "tdee": round(tdee, 1),
            "activity_level": activity
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------- Manual input + my foods ----------
def login_required(f):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/manual_input_food', methods=['POST'])
@limiter.limit("20 per minute")
@login_required
def manual_input_food():
    data = request.json or {}
    user_id = session.get("user_id")
    food_name = data.get("food_name")
    calories = data.get("calories")
    quantity = data.get("quantity", 1)
    micronutrients = data.get("micronutrients", [])
    created_at = data.get("created_at")
    if not food_name:
        return jsonify({"error": "Food name is required."}), 400
    try:
        with db_conn() as conn, conn.cursor() as cur:
            if created_at:
                cur.execute(
                    "INSERT INTO public.foods (user_id, food_name, calories, quantity, created_at) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                    (user_id, food_name, calories, quantity, created_at)
                )
            else:
                cur.execute(
                    "INSERT INTO public.foods (user_id, food_name, calories, quantity) VALUES (%s,%s,%s,%s) RETURNING id",
                    (user_id, food_name, calories, quantity)
                )
            food_id = cur.fetchone()[0]
            for nutrient in micronutrients:
                cur.execute(
                    "INSERT INTO public.micronutrients (food_id, nutrient_name, amount, unit) VALUES (%s,%s,%s,%s)",
                    (food_id, nutrient.get('name'), nutrient.get('amount'), nutrient.get('unit'))
                )
            conn.commit()
        return jsonify({"message": "Food entry saved successfully!", "food_id": food_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/my_foods', methods=['GET'])
@limiter.limit("10 per minute")
@login_required
def my_foods():
    user_id = session.get("user_id")
    try:
        with db_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT f.id AS food_id, f.food_name, f.calories, f.quantity, f.created_at,
                       m.nutrient_name, m.amount, m.unit
                  FROM public.foods f
             LEFT JOIN public.micronutrients m ON f.id = m.food_id
                 WHERE f.user_id = %s
              ORDER BY f.created_at ASC
            """, (user_id,))
            rows = cur.fetchall()
        foods_map = {}
        for row in rows:
            fid = row['food_id']
            if fid not in foods_map:
                foods_map[fid] = {
                    'id': row['food_id'],
                    'food_name': row['food_name'],
                    'calories': row['calories'],
                    'quantity': row['quantity'],
                    'created_at': row['created_at'],
                    'micronutrients': []
                }
            if row['nutrient_name']:
                foods_map[fid]['micronutrients'].append({
                    'name': row['nutrient_name'],
                    'amount': row['amount'],
                    'unit': row['unit']
                })
        return jsonify({"foods": list(foods_map.values())})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Recipes (matches your columns) ----------
@app.post("/api/recipes")
@login_required
def create_recipe():
    data = request.get_json(force=True)
    user_id = session.get("user_id")
    title = (data.get("title") or "").strip()
    web_link = (data.get("web_link") or "").strip()
    generative = bool(data.get("generative", True))
    # We’ll craft content with llama.cpp from provided inputs:
    ingredients = data.get("ingredients", [])
    micronutrients = data.get("micronutrients", [])
    prefs = data.get("preferences","")

    if not title:
        return jsonify({"error":"title required"}), 400

    prompt = (
        f"Create a recipe titled '{title}'.\n"
        f"Ingredients: {ingredients}\n"
        f"Target micronutrients: {micronutrients}\n"
        f"Preferences: {prefs}\n"
        f"Provide: title, ingredients (with amounts), instructions, and a brief nutritional summary."
    )
    content = ask_llamacpp(prompt, contexts=[])

    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.recipes (user_id, title, content, web_link, generative)
                VALUES (%s,%s,%s,%s,%s) RETURNING id
            """, (user_id, title, content, web_link, generative))
            rid = cur.fetchone()[0]
            conn.commit()
        return jsonify({"recipe_id": rid, "content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/recipes/mine")
@login_required
def my_recipes():
    user_id = session.get("user_id")
    try:
        with db_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, content, web_link, generative, created_at
                  FROM public.recipes
                 WHERE user_id=%s
              ORDER BY created_at DESC
            """, (user_id,))
            rows = cur.fetchall()
        return jsonify({"recipes": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/recipes/search")
@login_required
def search_my_recipes():
    """
    Search the current user's recipes.
    Query params:
      term=        (matches title or content)
      nutrient=    (matches content string, e.g. 'iron', 'vitamin c')
    If both are provided, both must match.
    """
    user_id = session.get("user_id")

    term      = (request.args.get("term") or "").strip().lower()
    nutrient  = (request.args.get("nutrient") or "").strip().lower()

    try:
        with db_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, content, web_link, generative, created_at
                  FROM public.recipes
                 WHERE user_id = %s
              ORDER BY created_at DESC
            """, (user_id,))
            rows = cur.fetchall()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    def matches(r):
        text_title = (r.get("title")   or "").lower()
        text_body  = (r.get("content") or "").lower()

        # name / general text match
        if term:
            if term not in text_title and term not in text_body:
                return False

        # micronutrient match (also just substring in body for now)
        if nutrient:
            if nutrient not in text_body:
                return False

        return True

    filtered = [r for r in rows if matches(r)]

    return jsonify({"recipes": filtered})

# Optional: comments
@app.post("/api/recipes/<int:recipe_id>/comments")
@login_required
def add_comment(recipe_id):
    content = (request.get_json(force=True).get("content") or "").strip()
    if not content:
        return jsonify({"error":"content required"}), 400
    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.recipe_comments (recipe_id, user_id, content)
                VALUES (%s,%s,%s) RETURNING id
            """, (recipe_id, session["user_id"], content))
            cid = cur.fetchone()[0]
            conn.commit()
        return jsonify({"comment_id": cid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------- Recipe Helpers for renaming -------

def _require_json_field(obj, key, max_len=200):
    if not obj or key not in obj:
        return None, f"Missing field: {key}"
    val = (obj.get(key) or "").strip()
    if not val:
        return None, f"{key} cannot be empty"
    if len(val) > max_len:
        return None, f"{key} too long (>{max_len})"
    return val, None

def _rename_recipe_for_user(db_conn, recipe_id: int, user_id: int, new_title: str):
    with db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            UPDATE public.recipes
               SET title = %s
             WHERE id = %s AND user_id = %s
         RETURNING id, user_id, title, content, created_at, web_link, generative
            """,
            (new_title, recipe_id, user_id),
        )
        row = cur.fetchone()
        if not row:
            return None
    db_conn.commit()
    return row

def _get_db_conn():
    # reuse your existing connection helper if you already have one
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL env var is not set")
    return psycopg2.connect(url)

def _current_user_id_or_401():
    uid = session.get("user_id")
    if not uid:
        abort(401, description="Authentication required")
    return uid

@app.route("/api/recipes/<int:recipe_id>", methods=["PATCH", "OPTIONS"])
def patch_recipe(recipe_id: int):
    # CORS preflight support (no-op if you don't use CORS)
    if request.method == "OPTIONS":
        return ("", 204)

    user_id = _current_user_id_or_401()
    payload = request.get_json(silent=True) or {}
    new_title, err = _require_json_field(payload, "title", max_len=200)
    if err:
        return jsonify({"error": err}), 400

    try:
        conn = _get_db_conn()
        row = _rename_recipe_for_user(conn, recipe_id, user_id, new_title)
    except Exception as e:
        return jsonify({"error": f"DB error: {e}"}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not row:
        # either not found or not owned by this user
        return jsonify({"error": "Recipe not found or not owned by this user"}), 404
    return jsonify({"ok": True, "recipe": row})

@app.route("/api/recipes/rename", methods=["POST"])
def post_recipe_rename():
    user_id = _current_user_id_or_401()
    payload = request.get_json(silent=True) or {}

    # Accept id/title from fallback route
    rid = payload.get("id")
    try:
        recipe_id = int(rid)
    except Exception:
        return jsonify({"error": "Invalid or missing recipe id"}), 400

    new_title, err = _require_json_field(payload, "title", max_len=200)
    if err:
        return jsonify({"error": err}), 400

    try:
        conn = _get_db_conn()
        row = _rename_recipe_for_user(conn, recipe_id, user_id, new_title)
    except Exception as e:
        return jsonify({"error": f"DB error: {e}"}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not row:
        return jsonify({"error": "Recipe not found or not owned by this user"}), 404
    return jsonify({"ok": True, "recipe": row})


@app.route("/api/recipes/<int:recipe_id>", methods=["DELETE"])
def delete_recipe(recipe_id: int):
    """
    Delete a recipe owned by the current user.
    Returns {ok: True} if deleted, 404 if not found / not owned.
    """
    user_id = _current_user_id_or_401()

    try:
        conn = _get_db_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                DELETE FROM public.recipes
                 WHERE id = %s AND user_id = %s
             RETURNING id
            """, (recipe_id, user_id))
            row = cur.fetchone()
        conn.commit()
    except Exception as e:
        return jsonify({"error": f"DB error: {e}"}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not row:
        return jsonify({"error": "Recipe not found or not owned by this user"}), 404
    return jsonify({"ok": True})

# ---------- Text parsing with llama.cpp ----------
@app.route('/parse_food_text', methods=['POST'])
def parse_food_text():
    data = request.json or {}
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    prompt = (
        "Convert the following food description into a valid JSON object with keys: "
        "'food_name' (string), 'calories' (number), 'micronutrients' (array of objects "
        "with keys 'name', 'amount', 'unit'), and optionally 'ingredients' (string). "
        "Return only the JSON.\n\n" + text
    )
    model_out = ask_llamacpp(prompt, contexts=[])
    try:
        food_json = json.loads(model_out)
        return jsonify({"parsed_food": food_json})
    except Exception:
        return jsonify({"error": "Failed to parse JSON from model output", "llm_output": model_out}), 500

# ---------- Image scan (plain; vision not used by llama.cpp) ----------
ALLOWED_EXTENSIONS = {"png","jpg","jpeg","webp"}
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/scan_food_image', methods=['POST'])
@limiter.limit("10 per minute")
def scan_food_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    img_bytes = file.read()
    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            width, height = img.size
            quality = "High" if width * height > 500_000 else "Standard"

        # Placeholder detector; integrate real model if desired
        detected_foods = [{"food_name": "Example Food", "confidence": 0.90, "bounding_box": [50, 50, 200, 200]}]

        prompt = (
            "Detected food items: " +
            ", ".join(f"{f['food_name']} (confidence: {f['confidence']:.2f})" for f in detected_foods) +
            ". Provide a detailed nutritional summary and health advice for these items."
        )

        response_text = ask_llamacpp(prompt, contexts=[])

        return jsonify({
            "detected_foods": detected_foods,
            "llm_response": response_text,
            "quality": quality
        })
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

# ---------- Ask with RAG ----------
@app.post("/ask")
def api_ask():
    q = (request.form.get("question") or (request.json or {}).get("question") or "").strip()
    if not q: return jsonify({"error":"question required"}), 400
    hits = search_docs(q, k=5)
    contexts = [h["content"] for h in hits]
    answer = ask_llamacpp(q, contexts)
    return jsonify({"answer": answer, "retrieved": hits})

# ----- Foods: update & delete (to support my_foods page) -----
@app.patch("/api/foods/<int:food_id>")
@login_required
def update_food(food_id):
    data = request.get_json(force=True)
    name = data.get("food_name")
    calories = data.get("calories")
    quantity = data.get("quantity")
    micronutrients = data.get("micronutrients")  # list[{name,amount,unit}]

    try:
        with db_conn() as conn, conn.cursor() as cur:
            if name is not None or calories is not None or quantity is not None:
                cur.execute("""
                    UPDATE public.foods
                       SET food_name = COALESCE(%s, food_name),
                           calories  = COALESCE(%s, calories),
                           quantity  = COALESCE(%s, quantity)
                     WHERE id=%s AND user_id=%s
                """, (name, calories, quantity, food_id, session["user_id"]))
            if isinstance(micronutrients, list):
                # Replace micronutrients for this food_id
                cur.execute("DELETE FROM public.micronutrients WHERE food_id=%s", (food_id,))
                for n in micronutrients:
                    cur.execute("""
                        INSERT INTO public.micronutrients (food_id, nutrient_name, amount, unit)
                        VALUES (%s,%s,%s,%s)
                    """, (food_id, n.get("name"), n.get("amount"), n.get("unit")))
            conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.delete("/api/foods/<int:food_id>")
@login_required
def delete_food(food_id):
    try:
        with db_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM public.foods WHERE id=%s AND user_id=%s",
                        (food_id, session["user_id"]))
            cur.execute("DELETE FROM public.micronutrients WHERE food_id=%s", (food_id,))
            conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ensure llama object is dropped cleanly on exit
def _cleanup_llm():
    global _LLM
    _LLM = None
atexit.register(_cleanup_llm)

# ---------- Main ----------
if __name__ == "__main__":
    ensure_usda_loaded()  # optional: prints local USDA summary
    ensure_schema() #Needed for schema validation
# Turn OFF the reloader to avoid double-initialization of llama.cpp
    app.run(host="0.0.0.0",
            port=int(os.getenv("PORT", "5000")),
            debug=False,
            use_reloader=False)
