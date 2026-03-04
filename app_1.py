# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
from treeinterpreter import treeinterpreter as ti
from sklearn.pipeline import Pipeline
from analysis import FEATURE_COLS

import requests
import time
import re
import hashlib
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
import jieba
from typing import Dict, Any, List, Optional, Tuple

import os
import random

def _load_dotenv_if_present(path: str = ".env") -> None:
    """
    Minimal .env loader (no external deps).
    - Only sets keys that are not already in os.environ.
    - Supports lines like: KEY=VALUE, export KEY=VALUE, and quoted values.
    """
    try:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if not k:
                    continue
                if k in os.environ:
                    continue
                if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ("'", '"')):
                    v = v[1:-1]
                os.environ[k] = v
    except Exception as e:
        print(f"[INIT] .env load skipped ({e})")

_load_dotenv_if_present(".env")

def _compute_feature_zscores(
    data: pd.DataFrame,
    cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    x = data[cols].astype(float).values
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return mu, sigma

def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm = np.where(a_norm == 0, 1.0, a_norm)
    b_norm = np.where(b_norm == 0, 1.0, b_norm)
    return (a / a_norm) @ (b / b_norm).T

def find_similar_ups(uid: int, k: int = 3) -> List[Dict[str, Any]]:
    """
    从当前数据集中找最相似的 k 个 UP（基于 FEATURE_COLS 的 z-score + cosine similarity）。
    返回：[{uid, up_name, followers, similarity}]
    """
    if df is None or df.empty:
        return []
    if uid not in set(df["uid"].astype(int).tolist()):
        return []

    k = max(1, min(10, int(k)))

    data = df.copy()
    data["uid"] = data["uid"].astype(int)

    # 只使用 feature 全部可用的行
    feat_df = data.dropna(subset=FEATURE_COLS)
    if feat_df.empty:
        return []

    row_df = feat_df[feat_df["uid"] == int(uid)]
    if row_df.empty:
        return []

    mu, sigma = _compute_feature_zscores(feat_df, FEATURE_COLS)
    x_all = feat_df[FEATURE_COLS].astype(float).values
    x_all_z = (x_all - mu) / sigma
    x_uid = row_df[FEATURE_COLS].astype(float).values
    x_uid_z = (x_uid - mu) / sigma

    sims = _cosine_sim_matrix(x_all_z, x_uid_z).reshape(-1)
    feat_df = feat_df.assign(_sim=sims)
    peers = feat_df[feat_df["uid"] != int(uid)].sort_values("_sim", ascending=False).head(k)

    out: List[Dict[str, Any]] = []
    for _, r in peers.iterrows():
        out.append({
            "uid": int(r["uid"]),
            "up_name": str(r.get("up_name", "") or ""),
            "followers": int(r.get("followers", -1)) if "followers" in r else -1,
            "similarity": float(r.get("_sim", 0.0)),
        })
    return out


# ================== B 站 API 配置 ==================
BILI_COOKIE = {
    "SESSDATA": "48208f15%2C1780795775%2Ca09b7%2Ac2CjCFv2lRhhwSKTbEionR_IgxMwQNZsq_uwOtRwKYage6h7IJc8Kwn9ZDo3_D9r2E68YSVkM2ODgycVFJQS1mOTU1YTI0Q2tqV2xVVHdjSG54QjVZYk5SRnEyaXYwdTNHOVNGNlh2Ykt3aHZFTXZYUjZxWGZvZFdfS2JIMEVIN2pKbENFTHdWQVlRIIEC",
    "bili_jct": "91ef32ebe104a8dccc4cf362434663ff",
    "buvid3": "B0113614-A503-CE63-EDFC-3DB07AEFB9B642722infoc",
}

BILI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15.7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
}

VIDEOS_PER_UP = 20
MAX_COMMENTS_PER_VIDEO = 100

# ================== 特征显示（前端友好） ==================
# better: "high" 表示越高越好；"low" 表示越低越好；None 表示不做单调假设
FEATURE_META: Dict[str, Dict[str, Any]] = {
    "avg_comment_scraped": {
        "zh_name": "平均评论数（抓取）",
        "zh_desc": "每条视频抓取到的评论条数的平均值（最多抓取一定数量）。",
        "en_name": "Avg comments (scraped)",
        "en_desc": "Average number of comments scraped per video (capped by crawler).",
        "better": "high",
    },
    "avg_danmaku": {
        "zh_name": "平均弹幕数",
        "zh_desc": "每条视频弹幕条数的平均值。",
        "en_name": "Avg danmaku",
        "en_desc": "Average danmaku count per video.",
        "better": "high",
    },
    "avg_length": {
        "zh_name": "平均视频时长（秒）",
        "zh_desc": "近若干条视频的平均时长（秒）。",
        "en_name": "Avg length (sec)",
        "en_desc": "Average video length in seconds.",
        "better": None,
    },
    "avg_play": {
        "zh_name": "平均播放量",
        "zh_desc": "近若干条视频播放量的平均值。",
        "en_name": "Avg plays",
        "en_desc": "Average play count across recent videos.",
        "better": "high",
    },
    "comment_repetition": {
        "zh_name": "评论重复度",
        "zh_desc": "评论分词后 Top 词占比（越高表示重复/灌水越多）。",
        "en_name": "Comment repetition",
        "en_desc": "Top-token share after tokenization (higher means more repetitive/spammy).",
        "better": "low",
    },
    "danmaku_missing_rate": {
        "zh_name": "弹幕缺失率",
        "zh_desc": "弹幕为 0 的视频占比（越高代表互动缺失越多）。",
        "en_name": "Danmaku missing rate",
        "en_desc": "Share of videos with zero danmaku (higher means more missing engagement).",
        "better": "low",
    },
    "med_danmaku": {
        "zh_name": "弹幕中位数",
        "zh_desc": "近若干条视频弹幕条数的中位数。",
        "en_name": "Median danmaku",
        "en_desc": "Median danmaku count across recent videos.",
        "better": "high",
    },
    "med_play": {
        "zh_name": "播放量中位数",
        "zh_desc": "近若干条视频播放量的中位数。",
        "en_name": "Median plays",
        "en_desc": "Median play count across recent videos.",
        "better": "high",
    },
    "std_length": {
        "zh_name": "时长波动（标准差）",
        "zh_desc": "视频时长的标准差（越高代表内容结构/节奏不稳定）。",
        "en_name": "Length volatility (std)",
        "en_desc": "Standard deviation of video length (higher means less consistent).",
        "better": "low",
    },
    "upload_freq": {
        "zh_name": "更新频率（视频/天）",
        "zh_desc": "近若干条视频覆盖时间窗内的日均更新量。",
        "en_name": "Upload frequency (videos/day)",
        "en_desc": "Average uploads per day over the observed window.",
        "better": "high",
    },
}

def get_feature_meta_for_lang(lang: str) -> Dict[str, Dict[str, Any]]:
    lang = (lang or "zh").strip().lower()
    is_en = (lang == "en")
    meta: Dict[str, Dict[str, Any]] = {}
    for k in FEATURE_COLS:
        m = FEATURE_META.get(k) or {}
        meta[k] = {
            "name": (m.get("en_name") if is_en else m.get("zh_name")) or k,
            "desc": (m.get("en_desc") if is_en else m.get("zh_desc")) or "",
            "better": m.get("better"),
        }
    return meta

def _fmt_num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float, np.floating, np.integer)):
            return float(v)
        return float(str(v))
    except Exception:
        return None

def build_rule_based_summary(
    items: List[Dict[str, Any]],
    feature_meta: Dict[str, Dict[str, Any]],
    lang: str,
) -> str:
    if not items:
        return ""

    lang = (lang or "zh").strip().lower()
    is_en = (lang == "en")

    # 找到最强正/负驱动（并尽量给出“对标优质UP”的解释）
    sorted_by_abs = sorted(items, key=lambda x: abs(float(x.get("contribution") or 0.0)), reverse=True)
    top_neg = [x for x in sorted_by_abs if float(x.get("contribution") or 0.0) < 0][:3]
    top_pos = [x for x in sorted_by_abs if float(x.get("contribution") or 0.0) > 0][:3]

    def _name(k: str) -> str:
        return (feature_meta.get(k) or {}).get("name") or k

    def _fmt(v: Any) -> str:
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            if isinstance(v, (float, np.floating)):
                return f"{float(v):.3g}"
            return str(v)
        except Exception:
            return "—"

    def _compare_line(x: Dict[str, Any], *, is_weakness: bool) -> str:
        feat = str(x.get("feature") or "")
        nm = _name(feat)
        contrib = float(x.get("contribution") or 0.0)
        value = x.get("value")
        med = x.get("good_median")
        better = x.get("better")
        sign = "+" if contrib >= 0 else ""
        base = f"{nm}（贡献 {sign}{contrib:.3f}）"
        if value is None or med is None:
            return base
        if better == "high":
            direction = "偏低" if is_weakness else "偏高"
        elif better == "low":
            direction = "偏高" if is_weakness else "偏低"
        else:
            direction = "偏离"
        return f"{base}：当前 {_fmt(value)}，优质UP中位数 {_fmt(med)}（{direction}）"

    if is_en:
        bad = [_compare_line(x, is_weakness=True) for x in top_neg]
        good = [_compare_line(x, is_weakness=False) for x in top_pos]
        recs = []
        for x in top_neg:
            k = str(x.get("feature") or "")
            better = (feature_meta.get(k) or {}).get("better")
            if better == "high":
                recs.append(f"Increase {_name(k)} (it is currently below strong creators).")
            elif better == "low":
                recs.append(f"Reduce {_name(k)} (lower is better for this metric).")
            else:
                recs.append(f"Optimize {_name(k)} based on your content strategy.")
        recs = recs[:3]
        lines: List[str] = []
        lines.append("Verdict")
        lines.append("- Your business value is currently driven mainly by a few key factors.")
        lines.append("")
        lines.append("Strengths (top)")
        if good:
            for g in good:
                lines.append(f"- {g}")
        else:
            lines.append("- No clear strengths detected yet.")
        lines.append("")
        lines.append("Weaknesses (top)")
        if bad:
            for b in bad:
                lines.append(f"- {b}")
        else:
            lines.append("- No clear weaknesses detected yet.")
        lines.append("")
        lines.append("Next actions (priority)")
        if recs:
            for r in recs:
                lines.append(f"- {r}")
        else:
            lines.append("- Focus on the top negative drivers first.")
        return "\n".join(lines).strip()

    bad = [_compare_line(x, is_weakness=True) for x in top_neg]
    good = [_compare_line(x, is_weakness=False) for x in top_pos]
    recs = []
    for x in top_neg:
        k = str(x.get("feature") or "")
        better = (feature_meta.get(k) or {}).get("better")
        if better == "high":
            recs.append(f"优先提升「{_name(k)}」：当前偏弱，建议对标优质 UP 的中位数水平。")
        elif better == "low":
            recs.append(f"优先降低「{_name(k)}」：该指标越低越好，建议减少异常/灌水/缺失情况。")
        else:
            recs.append(f"重点优化「{_name(k)}」：结合账号定位做结构化改进。")
    recs = recs[:3]
    lines: List[str] = []
    lines.append("结论")
    lines.append("- 当前商业价值主要被少数关键因素拉低/拉升。")
    lines.append("")
    lines.append("优势（Top）")
    if good:
        for g in good:
            lines.append(f"- {g}")
    else:
        lines.append("- 暂无明显优势项。")
    lines.append("")
    lines.append("短板（Top）")
    if bad:
        for b in bad:
            lines.append(f"- {b}")
    else:
        lines.append("- 暂无明显短板项。")
    lines.append("")
    lines.append("下一步建议（按优先级）")
    if recs:
        for r in recs:
            lines.append(f"- {r}")
    else:
        lines.append("- 先从贡献为负的特征入手逐项优化。")
    return "\n".join(lines).strip()

# ---- WBI ----
_WBI_KEYS: Dict[str, Optional[str]] = {"img_key": None, "sub_key": None}
_WBI_KEYS_TS: float = 0.0
_MIXIN_KEY_ENC_TAB = [
    46,47,18,2,53,8,23,32,15,50,10,31,58,3,45,35,27,43,5,49,33,9,42,19,29,28,
    14,39,12,38,41,13,37,48,7,16,24,55,40,61,26,17,0,1,60,51,30,4,22,25,54,21,
    56,59,6,36,34,11,52,20,57,44
]


def _mixin_key(img_key: str, sub_key: str) -> str:
    s = img_key + sub_key
    return ''.join([s[i] for i in _MIXIN_KEY_ENC_TAB])[:32]


def _refresh_wbi_keys_sync(force: bool = False) -> None:
    global _WBI_KEYS, _WBI_KEYS_TS
    now = time.time()
    if (not force) and _WBI_KEYS["img_key"] and (now - _WBI_KEYS_TS) < 600:
        return

    resp = requests.get(
        "https://api.bilibili.com/x/web-interface/nav",
        params={"fnval": 976},
        headers=BILI_HEADERS,
        cookies=BILI_COOKIE,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    d = (data or {}).get("data") or {}

    img_url = ((d.get("wbi_img") or {}).get("img_url")) or ""
    sub_url = ((d.get("wbi_img") or {}).get("sub_url")) or ""

    def _key_from(url: str) -> str:
        m = re.search(r'/([^/]+)\.(?:png|jpg)$', url)
        return m.group(1) if m else ""

    _WBI_KEYS["img_key"] = _key_from(img_url)
    _WBI_KEYS["sub_key"] = _key_from(sub_url)
    _WBI_KEYS_TS = now

    print(f"[WBI] img_key={_WBI_KEYS['img_key']}, sub_key={_WBI_KEYS['sub_key']}")

    if not (_WBI_KEYS["img_key"] and _WBI_KEYS["sub_key"]):
        raise RuntimeError("Failed to obtain WBI keys from nav API")


def _wbi_sign(params: Dict[str, Any]) -> Dict[str, Any]:
    filt: Dict[str, Any] = {}
    for k, v in params.items():
        v = re.sub(r"[!'\(\)*]", "", str(v))
        filt[k] = v

    filt["wts"] = str(int(time.time()))
    items = sorted(filt.items())
    query = '&'.join([f"{k}={v}" for k, v in items])

    mixin = _mixin_key(_WBI_KEYS["img_key"], _WBI_KEYS["sub_key"])
    filt["w_rid"] = hashlib.md5((query + mixin).encode("utf-8")).hexdigest()
    return filt


def _bili_get_json(url: str, params: Dict[str, Any], log_ctx: str = "") -> Dict[str, Any]:
    resp = requests.get(
        url,
        params=params,
        headers=BILI_HEADERS,
        cookies=BILI_COOKIE,
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    code = data.get("code", 0)
    if code != 0:
        print(f"[warn] bili api code={code}, msg={data.get('message')}, ctx={log_ctx}")
    return data


app = Flask(__name__)

# ================== 路径 ==================
CSV_PATH = "database/upfile_data_labeled_10.csv"
MODEL_PATH = "classifier/up_classifier_10dim.pkl"

# ================== 全局变量 ==================
df: pd.DataFrame | None = None
clf = None
SHAP_MIN: float = 0.0
SHAP_MAX: float = 1.0

# ================== 工具函数 ==================
def bucket_from_percentile(p: float) -> str:
    if p >= 80:
        return "Top 20%"
    elif p <= 20:
        return "Bottom 20%"
    return "Middle 60%"


def get_model_and_X_for_ti(clf, X_raw: np.ndarray):
    if isinstance(clf, Pipeline):
        if len(clf.steps) > 1:
            preproc = clf[:-1]
            model = clf.steps[-1][1]
            X_for_ti = preproc.transform(X_raw)
        else:
            model = clf.steps[-1][1]
            X_for_ti = X_raw
    else:
        model = clf
        X_for_ti = X_raw
    return model, X_for_ti


def recompute_scores():
    global df, SHAP_MIN, SHAP_MAX, clf
    if df is None or clf is None:
        raise RuntimeError("df 或 clf 尚未初始化")

    X_all = df[FEATURE_COLS].values
    proba_all = clf.predict_proba(X_all)
    pred_labels_all = clf.predict(X_all)

    df["model_prob_high"] = proba_all[:, 1]
    df["model_pred_label"] = pred_labels_all

    df["confidence"] = np.where(
        df["model_pred_label"] == 1,
        df["model_prob_high"],
        1.0 - df["model_prob_high"]
    )

    model_for_ti, X_for_ti = get_model_and_X_for_ti(clf, X_all)
    prediction_all, bias_all, contrib_all = ti.predict(model_for_ti, X_for_ti)

    if contrib_all.ndim == 3:
        class_idx = 1 if contrib_all.shape[2] > 1 else 0
        contrib_class = contrib_all[:, :, class_idx]
    elif contrib_all.ndim == 2:
        contrib_class = contrib_all
    else:
        raise ValueError(f"Unexpected contrib_all ndim: {contrib_all.ndim}")

    shap_sums = contrib_class.sum(axis=1)
    SHAP_MIN = float(shap_sums.min())
    SHAP_MAX = float(shap_sums.max())
    df["shap_sum"] = shap_sums

    if SHAP_MAX > SHAP_MIN:
        df["shap_norm"] = (df["shap_sum"] - SHAP_MIN) / (SHAP_MAX - SHAP_MIN)
    else:
        df["shap_norm"] = 0.5

    df["shap_norm"] = df["shap_norm"].clip(0.0, 1.0)

    df["value_score"] = 100.0 * (0.5 * df["confidence"] + 0.5 * df["shap_norm"])
    df["score_percentile"] = df["value_score"].rank(pct=True) * 100.0
    df["score_bucket"] = df["score_percentile"].apply(bucket_from_percentile)


# ================== B 站爬虫相关 ==================
def _fetch_up_profile(uid: int) -> Dict[str, Any]:
    _refresh_wbi_keys_sync()
    signed = _wbi_sign({"mid": uid})
    data = _bili_get_json(
        "https://api.bilibili.com/x/space/wbi/acc/info",
        params=signed,
        log_ctx=f"[up info uid={uid}]"
    )
    if data.get("code") != 0:
        return {}
    return (data or {}).get("data") or {}


def _fetch_relation_stat(uid: int) -> Dict[str, Any]:
    data = _bili_get_json(
        "https://api.bilibili.com/x/relation/stat",
        params={"vmid": uid},
        log_ctx=f"[relation stat uid={uid}]"
    )
    if data.get("code") != 0:
        return {}
    return (data or {}).get("data") or {}


def _fetch_user_videos(uid: int, limit: int = 20) -> List[Dict[str, Any]]:
    print(f"[CRAWL] fetching videos for uid={uid}, limit={limit} ...")
    _refresh_wbi_keys_sync()
    results: List[Dict[str, Any]] = []
    pn = 1
    ps = min(20, limit)

    while len(results) < limit:
        signed = _wbi_sign({"mid": uid, "pn": pn, "ps": ps, "order": "pubdate"})
        data = _bili_get_json(
            "https://api.bilibili.com/x/space/wbi/arc/search",
            params=signed,
            log_ctx=f"[wbi arc uid={uid} pn={pn}]"
        )
        if data.get("code") != 0:
            break

        vlist = (((data or {}).get("data") or {}).get("list") or {}).get("vlist") or []
        if not vlist:
            break

        for v in vlist:
            results.append({
                "uid": uid,
                "aid": v.get("aid"),
                "bvid": v.get("bvid"),
                "title": v.get("title"),
                "description": v.get("description", ""),
                "length": v.get("length"),
                "created": v.get("created"),
                "play": v.get("play"),
                "comment": v.get("comment"),
                "favorites": v.get("favorites"),
                "author": v.get("author"),
                "mid": v.get("mid"),
                "review": v.get("review"),
                "subtitle": v.get("subtitle"),
                "video_review": v.get("video_review"),
                "publication_date": v.get("created"),
                "raw": v,
            })
            if len(results) >= limit:
                break

        pn += 1
        time.sleep(0.8)

    print(f"[CRAWL] videos done uid={uid}, got={len(results)}")
    return results


def _parse_length_to_seconds(length_str: Optional[str]) -> Optional[float]:
    if not length_str:
        return None
    try:
        parts = [int(x) for x in length_str.split(":")]
    except Exception:
        return None
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    elif len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    return None


def _fetch_comments_for_video(aid: int, max_comments: int = MAX_COMMENTS_PER_VIDEO) -> List[str]:
    texts: List[str] = []
    if not aid:
        print(f"[CRAWL] skip comments: no aid")
        return texts

    print(f"[CRAWL] fetching comments for aid={aid} ...")
    ps = 20
    pn = 1
    loaded = 0

    while loaded < max_comments:
        data = _bili_get_json(
            "https://api.bilibili.com/x/v2/reply",
            params={"type": 1, "oid": aid, "sort": 2, "pn": pn, "ps": ps},
            log_ctx=f"[reply top oid={aid} pn={pn}]"
        )
        if data.get("code") != 0:
            print(f"[CRAWL] comments api non-zero code={data.get('code')} aid={aid}")
            break

        top = (((data or {}).get("data") or {}).get("replies")) or []
        if not top:
            print(f"[CRAWL] comments no more replies, aid={aid}, pn={pn}")
            break

        for r in top:
            content = (r.get("content") or {}).get("message") or ""
            if isinstance(content, str):
                content = content.strip()
            else:
                content = ""
            if not content:
                continue
            texts.append(content)
            loaded += 1
            if loaded >= max_comments:
                break
        pn += 1
        time.sleep(0.5)

    print(f"[CRAWL] comments done aid={aid}, got={len(texts)}")
    return texts


def _fetch_danmaku_first_page(bvid: str) -> List[str]:
    """
    拉首 P 的【全部】弹幕（不设上限），如果 412 / 4xx / 5xx 或 XML 错误会直接抛异常。
    """
    if not bvid:
        print("[CRAWL] skip danmaku: empty bvid")
        return []

    print(f"[CRAWL] fetching danmaku for bvid={bvid} ...")

    data = _bili_get_json(
        "https://api.bilibili.com/x/web-interface/view",
        params={"bvid": bvid},
        log_ctx=f"[view bvid={bvid}]"
    )
    if data.get("code") != 0:
        raise RuntimeError(
            f"view api non-zero code={data.get('code')} msg={data.get('message')} bvid={bvid}"
        )

    info = (data or {}).get("data") or {}
    pages = info.get("pages") or []
    if not pages:
        raise RuntimeError(f"no pages for bvid={bvid}")

    cid = pages[0].get("cid")
    if not cid:
        raise RuntimeError(f"no cid for bvid={bvid}")
    
    headers = dict(BILI_HEADERS)
    headers["Referer"] = f"https://www.bilibili.com/video/{bvid}"

    resp = requests.get(
        "https://api.bilibili.com/x/v1/dm/list.so",
        params={"oid": cid},
        headers=BILI_HEADERS,
        cookies=BILI_COOKIE,
        timeout=60,
    )
    resp.raise_for_status()
    xml_text = resp.text

    root = ET.fromstring(xml_text)
    texts: List[str] = []
    for d in root.findall(".//d"):
        t = (d.text or "").strip()
        if t:
            texts.append(t)

    print(f"[CRAWL] danmaku done bvid={bvid}, got={len(texts)}")
    return texts


_ = jieba.lcut("初始化一下分词器")


def compute_repetition_ratio(texts: List[str], top_k: int = 30, max_tokens: int = 50000) -> float:
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return 0.0

    tokens: List[str] = []
    for t in texts:
        toks = jieba.lcut(t)
        tokens.extend(toks)
        if len(tokens) >= max_tokens:
            break

    if not tokens:
        return 0.0

    total = len(tokens)
    counter = Counter(tokens)
    most_common = counter.most_common(top_k)
    top_sum = sum(c for _, c in most_common)
    return float(top_sum) / float(total) if total > 0 else 0.0


FEATURE_KEYS_FOR_CRAWL = [
    "uid",
    "up_name",
    "followers",
    "label_binary",
    "avg_comment_scraped",
    "avg_danmaku",
    "avg_length",
    "avg_play",
    "comment_repetition",
    "danmaku_missing_rate",
    "med_danmaku",
    "med_play",
    "std_length",
    "upload_freq",
]


def fetch_features_for_uid(uid: int) -> dict:
    print(f"[CRAWL] === start fetch_features_for_uid uid={uid} ===")

    up_info = _fetch_up_profile(uid)
    rel = _fetch_relation_stat(uid)
    print(f"[CRAWL] up_info name={up_info.get('name')} followers={rel.get('follower') if rel else None}")

    up_name = up_info.get("name") or ""
    followers = None
    if rel and isinstance(rel, dict):
        followers = rel.get("follower")
    if followers is None:
        followers = up_info.get("follower")
    if followers is None:
        followers = 0

    videos = _fetch_user_videos(uid, limit=VIDEOS_PER_UP)
    print(f"[CRAWL] fetched {len(videos)} videos for uid={uid}")
    if not videos:
        raise RuntimeError(f"该 UID ({uid}) 未获取到任何视频，无法计算特征")

    length_secs: List[float] = []
    play_counts: List[float] = []
    comment_counts: List[float] = []
    danmaku_counts: List[float] = []
    created_ts: List[int] = []
    all_comment_texts: List[str] = []

    for v in videos:
        aid = v.get("aid")
        bvid = v.get("bvid")

        ls = _parse_length_to_seconds(v.get("length"))
        if ls is not None:
            length_secs.append(ls)

        play = v.get("play")
        try:
            play = float(play) if play is not None else 0.0
        except Exception:
            play = 0.0
        play_counts.append(play)

        ts = v.get("created")
        if isinstance(ts, (int, float)):
            created_ts.append(int(ts))

        c_texts = _fetch_comments_for_video(aid, max_comments=MAX_COMMENTS_PER_VIDEO)
        all_comment_texts.extend(c_texts)
        comment_counts.append(float(len(c_texts)))

        dm_texts = _fetch_danmaku_first_page(bvid)
        time.sleep(1)
        danmaku_counts.append(float(len(dm_texts)))

    def _safe_mean(lst: List[float]) -> float:
        return float(statistics.mean(lst)) if lst else 0.0

    def _safe_median(lst: List[float]) -> float:
        return float(statistics.median(lst)) if lst else 0.0

    def _safe_stdev(lst: List[float]) -> float:
        return float(statistics.pstdev(lst)) if len(lst) > 1 else 0.0

    avg_length = _safe_mean(length_secs)
    std_length = _safe_stdev(length_secs)

    avg_play = _safe_mean(play_counts)
    med_play = _safe_median(play_counts)

    avg_comment_scraped = _safe_mean(comment_counts)

    avg_danmaku = _safe_mean(danmaku_counts)
    med_danmaku = _safe_median(danmaku_counts)

    if danmaku_counts:
        miss_dm = sum(1 for x in danmaku_counts if x == 0.0)
        danmaku_missing_rate = miss_dm / float(len(danmaku_counts))
    else:
        danmaku_missing_rate = 0.0

    if created_ts:
        t_min = min(created_ts)
        t_max = max(created_ts)
        days = max(1.0, (t_max - t_min) / 86400.0)
        upload_freq = len(created_ts) / days
    else:
        upload_freq = 0.0

    comment_repetition = compute_repetition_ratio(all_comment_texts, top_k=30, max_tokens=50000)

    label_binary = -1

    print(f"[CRAWL] lengths={len(length_secs)}, plays={len(play_counts)}, cmt={len(comment_counts)}, dm={len(danmaku_counts)}")
    print(f"[CRAWL] comment_repetition={comment_repetition}, upload_freq={upload_freq:.4f}")
    print(f"[CRAWL] === end fetch_features_for_uid uid={uid} ===")

    return {
        "uid": int(uid),
        "up_name": up_name,
        "followers": int(followers),
        "label_binary": int(label_binary),
        "avg_comment_scraped": float(avg_comment_scraped),
        "avg_danmaku": float(avg_danmaku),
        "avg_length": float(avg_length),
        "avg_play": float(avg_play),
        "comment_repetition": float(comment_repetition) if comment_repetition is not None else 0.0,
        "danmaku_missing_rate": float(danmaku_missing_rate),
        "med_danmaku": float(med_danmaku),
        "med_play": float(med_play),
        "std_length": float(std_length),
        "upload_freq": float(upload_freq),
    }


def upsert_row_and_recompute(new_row: dict) -> Tuple[str, pd.Series]:
    """
    - 写入/更新全局 df
    - 重算预测 / 分数 / SHAP
    - 新插入的 uid：label_binary = model_pred_label
    - 已存在的 uid：不改 label_binary
    - 写回 CSV
    """
    global df

    uid_int = int(new_row["uid"])
    new_row_df = pd.DataFrame([new_row])
    new_row_df["uid"] = uid_int

    mask_existing = (df["uid"] == uid_int)

    if mask_existing.any():
        # 已存在：更新除 label_binary 以外的爬虫特征
        for col in FEATURE_KEYS_FOR_CRAWL:
            if col == "label_binary":
                continue
            if col in new_row_df.columns:
                if col not in df.columns:
                    df[col] = None
                df.loc[mask_existing, col] = new_row_df[col].iloc[0]
        action = "updated"
    else:
        # 不存在：直接追加
        if "label_binary" not in new_row_df.columns:
            new_row_df["label_binary"] = -1
        df = pd.concat([df, new_row_df], ignore_index=True)
        action = "inserted"

    # 重算全表分数（包含这行）
    recompute_scores()

    # 只对新插入的行同步 label_binary
    if "label_binary" not in df.columns:
        df["label_binary"] = -1

    mask_uid = (df["uid"] == uid_int)
    if action == "inserted":
        df.loc[mask_uid, "label_binary"] = df.loc[mask_uid, "model_pred_label"].astype(int)

    # 写回 CSV
    df.to_csv(CSV_PATH, index=False)

    # 返回最新这一行
    row_df = df[df["uid"] == uid_int]
    row = row_df.iloc[0]
    print(f"[UPSERT] uid={uid_int}, action={action}, label_binary={row.get('label_binary')}")
    return action, row

def call_deepseek_summary(
    items: List[Dict[str, Any]],
    feature_meta: Dict[str, Dict[str, Any]],
    lang: str = "zh",
) -> str:
    """
    调用 DeepSeek 生成总结。key 从环境变量读取：DEEPSEEK_API_KEY
    返回纯文本；失败返回空串。
    """
    global _LLM_KEY_WARNED
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        if not _LLM_KEY_WARNED:
            print("[LLM] DEEPSEEK_API_KEY not set, skip llm_summary (use fallback summary).")
            _LLM_KEY_WARNED = True
        return ""

    # 控制输入长度：按绝对值取前 10（本项目正好 10 维），并附带可读名称/描述/对标信息
    sorted_items = sorted(items, key=lambda x: abs(float(x.get("contribution") or 0.0)), reverse=True)
    top_items = sorted_items[:10]

    if lang == "en":
        system = (
            "You are a product data analyst. "
            "Use the model interpretation outputs to write a user-friendly diagnostic summary. "
            "Be concrete, avoid jargon, output plain text."
        )
        user = (
            "Context:\n"
            "- Each item contains: feature key, human name/desc, current value, good creators' median/min, and contribution.\n"
            "- A negative contribution means the feature is currently hurting the predicted business value.\n\n"
            f"Top items:\n{top_items}\n\n"
            f"Feature dictionary:\n{feature_meta}\n\n"
            "Output format requirements:\n"
            "- Use line breaks and bullet points.\n"
            "- Keep it scannable (no long paragraphs).\n"
            "- Do NOT output JSON or code fences.\n\n"
            "Please output exactly these section titles:\n"
            "Verdict\n"
            "Strengths (Top 3)\n"
            "Weaknesses (Top 3)\n"
            "Next actions (Top 3, prioritized)\n"
        )
    else:
        system = (
            "你是产品数据分析师。"
            "请基于模型解释结果，写一段用户能看懂的诊断总结：指出亮点与短板，并给出下一步可执行建议。"
            "避免术语堆砌，输出纯文本。"
        )
        user = (
            "说明：每个 item 包含特征 key、可读名称/解释、当前值、优质UP中位数/最小值、以及贡献值。\n"
            "贡献值为负表示该特征在当前状态下拉低商业价值判断。\n\n"
            f"Top items:\n{top_items}\n\n"
            f"特征字典:\n{feature_meta}\n\n"
            "输出格式要求：\n"
            "- 必须分行，使用项目符号（-）\n"
            "- 每点尽量一句话，避免长段落\n"
            "- 不要输出 JSON/代码块\n\n"
            "请严格按以下小标题输出：\n"
            "结论\n"
            "优势（Top 3）\n"
            "短板（Top 3）\n"
            "下一步建议（Top 3，按优先级）\n"
        )

    # ⚠️ base_url / model 以你实际 DeepSeek 文档为准
    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()
    except Exception as e:
        print("[LLM] deepseek summary error:", e)
        return ""

_LLM_KEY_WARNED = False

# ================== 初始化 ==================
print("[INIT] Loading CSV and model...")
df = pd.read_csv(CSV_PATH)
clf = joblib.load(MODEL_PATH)

df["uid"] = df["uid"].astype(int)

print("[INIT] Computing scores & SHAP...")
recompute_scores()
print("[INIT] Ready.")



# ================== 页面路由 ==================
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    uid = request.args.get("uid", "").strip()
    return render_template("dashboard.html", uid=uid)

@app.route("/api/search")
def api_search():
    """
    本地数据集搜索：
    - query 为数字：按 uid 精确匹配
    - query 为文本：按 up_name 模糊匹配（大小写不敏感）
    """
    query = (request.args.get("query", "") or "").strip()
    lang = request.args.get("lang", "zh").strip().lower()
    limit = request.args.get("limit", "8").strip()
    try:
        limit_int = max(1, min(20, int(limit)))
    except Exception:
        limit_int = 8

    if not query:
        msg = "请输入 UID 或 UP 主名称。" if lang != "en" else "Please enter a UID or creator name."
        return jsonify({"success": False, "message": msg, "results": []}), 400

    results = []
    if query.isdigit():
        uid_int = int(query)
        row_df = df[df["uid"] == uid_int]
        if not row_df.empty:
            r = row_df.iloc[0]
            results.append({
                "uid": int(r["uid"]),
                "up_name": str(r.get("up_name", "") or ""),
                "followers": int(r.get("followers", -1)) if "followers" in r else -1,
            })
    else:
        if "up_name" in df.columns:
            q = query.lower()
            mask = df["up_name"].fillna("").astype(str).str.lower().str.contains(q, na=False)
            hits = df[mask].head(limit_int)
            for _, r in hits.iterrows():
                results.append({
                    "uid": int(r["uid"]),
                    "up_name": str(r.get("up_name", "") or ""),
                    "followers": int(r.get("followers", -1)) if "followers" in r else -1,
                })

    msg = ""
    if not results:
        msg = "未在当前数据集中找到匹配项。" if lang != "en" else "No matches found in the current dataset."

    return jsonify({"success": True, "query": query, "results": results, "message": msg})

@app.route("/api/recommendations")
def api_recommendations():
    """随机推荐一些数据集中的 UP（闭环模式：只用现有数据）。"""
    lang = request.args.get("lang", "zh").strip().lower()
    limit = request.args.get("limit", "8").strip()
    try:
        limit_int = max(1, min(20, int(limit)))
    except Exception:
        limit_int = 8

    if df is None or df.empty:
        msg = "数据集为空。" if lang != "en" else "Dataset is empty."
        return jsonify({"success": False, "message": msg, "results": []}), 500

    n = min(limit_int, len(df))
    idxs = random.sample(range(len(df)), k=n)
    rows = df.iloc[idxs]
    results = []
    for _, r in rows.iterrows():
        results.append({
            "uid": int(r["uid"]),
            "up_name": str(r.get("up_name", "") or ""),
            "followers": int(r.get("followers", -1)) if "followers" in r else -1,
        })

    return jsonify({"success": True, "results": results})

@app.route("/api/peers/<uid>")
def api_peers(uid):
    uid = (uid or "").strip()
    if not uid.isdigit():
        return jsonify({"success": False, "message": "Invalid UID"}), 400

    uid_int = int(uid)
    lang = request.args.get("lang", "zh").strip().lower()
    k = request.args.get("k", "3").strip()
    try:
        k_int = int(k)
    except Exception:
        k_int = 3

    row_df = df[df["uid"] == uid_int]
    if row_df.empty:
        msg = (
            f"UID={uid_int} 不在当前数据集中。"
            if lang != "en"
            else f"UID={uid_int} is not in the current dataset."
        )
        return jsonify({"success": False, "message": msg, "results": []}), 404

    peers = find_similar_ups(uid_int, k=k_int)
    if not peers:
        msg = "未找到可用的相似 UP（可能存在缺失特征）。" if lang != "en" else "No peers found (missing features)."
        return jsonify({"success": True, "uid": uid_int, "results": [], "message": msg})

    return jsonify({"success": True, "uid": uid_int, "results": peers})


# ================== API: 单个 UP 信息（预测 + 综合评分） ==================
@app.route("/api/predict/<uid>")
def api_predict(uid):
    uid = uid.strip()
    if not uid.isdigit():
        return jsonify({"success": False, "message": "Invalid UID"}), 400

    uid_int = int(uid)
    print(f"[API] /api/predict called uid={uid_int}")
    lang = request.args.get("lang", "zh").strip().lower()

    # 仅使用本地 CSV 现有数据，不做实时爬取/写回
    row_df = df[df["uid"] == uid_int]
    if row_df.empty:
        msg = (
            f"UID={uid_int} 不在当前数据集中，请换一个 UID。"
            if lang != "en"
            else f"UID={uid_int} is not in the current dataset. Please try another UID."
        )
        return jsonify({"success": False, "message": msg}), 404
    row = row_df.iloc[0]

    pred_label = int(row["model_pred_label"])
    prob_high = float(row["model_prob_high"])
    confidence = float(row["confidence"])
    value_score = float(row["value_score"])
    score_percentile = float(row["score_percentile"])
    score_bucket = row["score_bucket"]
    shap_sum = float(row["shap_sum"])
    shap_norm = float(row["shap_norm"])

    label_name = "高商业价值" if pred_label == 1 else "低商业价值"

    return jsonify({
        "success": True,
        "uid": uid_int,
        "up_name": row.get("up_name", ""),
        "followers": int(row.get("followers", -1)),
        "feature_meta": get_feature_meta_for_lang(lang),
        "prediction": {
            "label_binary": pred_label,
            "label_name": label_name,
            "prob_high": prob_high,
            "confidence": confidence,
            "value_score": value_score,
            "score_percentile": score_percentile,
            "score_bucket": score_bucket,
            "shap_sum": shap_sum,
            "shap_norm": shap_norm,
        },
        "features": {c: float(row[c]) for c in FEATURE_COLS}
    })


# ================== API: 优质 UP 统计 ==================
@app.route("/api/stats/good")
def good_stats():
    good_df = df[df.get("label_binary", 0) == 1]

    if good_df.empty:
        return jsonify({
            "success": False,
            "message": "No good UPs found (label_binary == 1)."
        }), 404

    median_vals = good_df[FEATURE_COLS].median().to_dict()
    min_vals = good_df[FEATURE_COLS].min().to_dict()

    return jsonify({
        "success": True,
        "median": median_vals,
        "min": min_vals
    })


# ================== API: 商业价值处方解释 ==================
@app.route("/api/prescription/<uid>")
def api_prescription(uid):
    uid = uid.strip()
    if not uid.isdigit():
        return jsonify({"success": False, "message": "Invalid UID"}), 400

    uid_int = int(uid)
    lang = request.args.get("lang", "zh").strip().lower()
    feature_meta = get_feature_meta_for_lang(lang)

    # 这里不再爬数据，只用“当前内存里的最新 df”
    row_df = df[df["uid"] == uid_int]
    if row_df.empty:
        return jsonify({
            "success": False,
            "message": "UID not found in server cache. 请先调用 /api/predict/<uid> 触发爬取。"
        }), 404

    row = row_df.iloc[0]

    x_raw = np.array([[row[c] for c in FEATURE_COLS]])
    model_for_ti, x_for_ti = get_model_and_X_for_ti(clf, x_raw)

    try:
        prediction, bias, contributions = ti.predict(model_for_ti, x_for_ti)
        contrib_arr = contributions[0]
        if contrib_arr.ndim == 2:
            class_idx = 1 if contrib_arr.shape[1] > 1 else 0
            contrib_arr = contrib_arr[:, class_idx]
    except Exception as e:
        return jsonify({
            "success": False,
            "uid": uid_int,
            "message": f"Failed to compute contributions: {e}"
        }), 500

    contrib_dict = {
        FEATURE_COLS[i]: float(contrib_arr[i])
        for i in range(len(FEATURE_COLS))
    }

    shap_sum = float(contrib_arr.sum())
    if SHAP_MAX > SHAP_MIN:
        shap_norm = (shap_sum - SHAP_MIN) / (SHAP_MAX - SHAP_MIN)
    else:
        shap_norm = 0.5
    shap_norm = float(np.clip(shap_norm, 0.0, 1.0))

    # 对标：优质 UP 的中位数/最小值（用于更“可执行”的总结）
    good_df = df[df.get("label_binary", 0) == 1]
    good_median = (good_df[FEATURE_COLS].median().to_dict() if not good_df.empty else {})
    good_min = (good_df[FEATURE_COLS].min().to_dict() if not good_df.empty else {})

    items: List[Dict[str, Any]] = []
    for feat in FEATURE_COLS:
        items.append({
            "feature": feat,
            "name": (feature_meta.get(feat) or {}).get("name") or feat,
            "desc": (feature_meta.get(feat) or {}).get("desc") or "",
            "value": _fmt_num(row.get(feat)),
            "good_median": _fmt_num(good_median.get(feat)),
            "good_min": _fmt_num(good_min.get(feat)),
            "contribution": float(contrib_dict.get(feat, 0.0)),
            "better": (feature_meta.get(feat) or {}).get("better"),
        })

    # 结构化建议：前端可按语言渲染
    sorted_items = sorted(items, key=lambda x: float(x.get("contribution") or 0.0))
    neg_items = [x for x in sorted_items if float(x.get("contribution") or 0.0) < 0]
    pos_items = [x for x in sorted(items, key=lambda x: float(x.get("contribution") or 0.0), reverse=True) if float(x.get("contribution") or 0.0) > 0]

    suggestions_struct: List[Dict[str, Any]] = []
    for x in neg_items[:5]:
        suggestions_struct.append({
            "feature": x["feature"],
            "impact": "negative",
            "contribution": float(x["contribution"]),
            "better": x.get("better"),
        })
    for x in pos_items[:3]:
        suggestions_struct.append({
            "feature": x["feature"],
            "impact": "positive",
            "contribution": float(x["contribution"]),
            "better": x.get("better"),
        })

    # 兼容旧的 suggestions 文案（默认中文，前端可继续显示或忽略）
    suggestions: List[str] = []
    for feat, contrib in contrib_dict.items():
        nm = (feature_meta.get(feat) or {}).get("name") or feat
        if contrib < -0.02:
            suggestions.append(f"【{nm}】对商业价值造成负向影响（{contrib:.3f}）。建议重点优化。")
        elif contrib > 0.02:
            suggestions.append(f"【{nm}】当前表现较好（贡献 {contrib:.3f}）。建议保持。")
        else:
            suggestions.append(f"【{nm}】影响较弱（{contrib:.3f}），可根据业务策略灵活调整。")

    llm_summary = call_deepseek_summary(items, feature_meta, lang=lang)
    if not llm_summary:
        llm_summary = build_rule_based_summary(items, feature_meta, lang=lang)
    return jsonify({
        "success": True,
        "uid": uid_int,
        "shap_sum": shap_sum,
        "shap_norm": shap_norm,
        "feature_meta": feature_meta,
        "items": items,
        "contributions": contrib_dict,
        "suggestions": suggestions,
        "suggestions_struct": suggestions_struct,
        "llm_summary": llm_summary
    })


# ================== 启动 ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
