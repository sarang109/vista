"""Login-signal priors from the public RBA login dataset (Wiefling et al., ACM TOPS 2022, CC BY 4.0).

The dataset (Zenodo record 6782156, no account or key needed) holds about 33 million login attempts by
3.3 million users of a Norwegian single-sign-on service (Feb 2020 - Feb 2021). Its authors synthesised the
released rows from the real login log, keeping the global and per-user categorical feature probabilities
identical to the original; geolocation, IPs and user-agent strings were regenerated and timestamps carry
added randomness. It contains logins, not payments, and no authentication-strength fields.

Four VISTA claim chains are derived per login from the user's own earlier logins (levels 0..3):
  device_continuity   OS family + browser family + device type: new, seen (1-2 earlier), habitual (>= 3),
                      dominant (habitual and >= 50% of the user's earlier logins)
  account_age         days since the user's first login in the data: <= 7, 8-30, 31-90, > 90
  auth_freshness      time since the user's last successful login: none or > 30 d, 7-30 d, 1-7 d, < 1 d
  network_continuity  country + ASN, with the same levels as device_continuity
States are recorded for successful, non-attack logins after a warm-up period, for a stable hash sample of
users. The empirical joint of the four chains is the real-signal prior used by E23, E24 and E26.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
import zipfile

import numpy as np

from ..claims import Attribute, Schema

URL = "https://zenodo.org/records/6782156/files/rba-dataset.zip?download=1"
MD5 = "cc1b1078b3929650e6c08678caffcc57"
CITATION = ("S. Wiefling, P. R. Jorgensen, S. Thunem, L. Lo Iacono. Pump Up Password Security! Evaluating and "
            "Enhancing Risk-Based Authentication on a Real-World Large-Scale Online Service. ACM TOPS, 2022. "
            "doi:10.1145/3546069. Data: doi:10.5281/zenodo.6782156 (CC BY 4.0)")
CACHE_VERSION = 1
DAY = 86_400
CHAINS = ("device_continuity", "account_age", "auth_freshness", "network_continuity")
RBA_SCHEMA = Schema("vista-rba-2026.1", [
    Attribute("device_continuity", ("new", "seen", "habitual", "dominant"), "device"),
    Attribute("account_age", ("le7d", "8_30d", "31_90d", "gt90d"), "account"),
    Attribute("auth_freshness", ("gt30d", "7_30d", "1_7d", "lt1d"), "authorization"),
    Attribute("network_continuity", ("new", "seen", "habitual", "dominant"), "network"),
])
_COLS = {"ts": "login timestamp", "user": "user id", "country": "country", "asn": "asn",
         "browser": "browser name and version", "os": "os name and version", "device": "device type",
         "success": "login successful", "attack_ip": "is attack ip", "ato": "is account takeover"}


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _family(s: str) -> str:
    """'Chrome Mobile 81.0.4044' -> 'chrome mobile' (version numbers change with every update)."""
    out = []
    for tok in s.strip().lower().split():
        if tok[:1].isdigit():
            break
        out.append(tok)
    return " ".join(out) or "unknown"


def _parse_ts(s: str) -> int:
    s = s.strip()
    if s.lstrip("-").isdigit():
        v = int(s)
        return v // 1000 if v > 10 ** 11 else v
    from datetime import datetime, timezone
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp())


def _user_key(uid: str) -> int:
    try:
        v = int(uid) & 0xFFFFFFFFFFFFFFFF
    except ValueError:
        v = int.from_bytes(hashlib.blake2b(uid.encode(), digest_size=8).digest(), "big")
    v = (v * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    return v ^ (v >> 29)


def _bool(s: str) -> bool:
    return s.strip().lower() in ("true", "1", "yes")


def _continuity(count_before: int, top_before: bool, total_before: int) -> int:
    if count_before == 0:
        return 0
    if count_before < 3:
        return 1
    return 3 if top_before and 2 * count_before >= total_before else 2


def _age(days: float) -> int:
    return 0 if days <= 7 else 1 if days <= 30 else 2 if days <= 90 else 3


def _fresh(gap_s: float | None) -> int:
    if gap_s is None or gap_s > 30 * DAY:
        return 0
    return 1 if gap_s > 7 * DAY else 2 if gap_s > DAY else 3


def derive_states(rows: np.ndarray, warmup_s: int) -> dict:
    """rows: structured array (user, ts, dev, net, success, attack) sorted by (user, ts). Returns per-login
    chain states for successful non-attack logins after the warm-up, plus attack-login states."""
    t_start = int(rows["ts"].min())
    st_legit, st_attack = [], []
    cur, first, last_ok, dev_c, net_c, total = None, 0, None, {}, {}, 0
    for u, ts, dv, nt, ok, att in zip(rows["user"].tolist(), rows["ts"].tolist(), rows["dev"].tolist(),
                                      rows["net"].tolist(), rows["success"].tolist(), rows["attack"].tolist()):
        if u != cur:
            cur, first, last_ok, dev_c, net_c, total = u, ts, None, {}, {}, 0
        cd, cn = dev_c.get(dv, 0), net_c.get(nt, 0)
        top_d = cd > 0 and cd >= max(dev_c.values())
        top_n = cn > 0 and cn >= max(net_c.values())
        state = (_continuity(cd, top_d, total), _age((ts - first) / DAY),
                 _fresh(None if last_ok is None else ts - last_ok), _continuity(cn, top_n, total))
        if ts - t_start >= warmup_s:
            if att:
                st_attack.append(state)
            elif ok:
                st_legit.append(state)
        if not att:  # an attacker's login does not become part of the user's history
            dev_c[dv] = cd + 1
            net_c[nt] = cn + 1
            total += 1
            if ok:
                last_ok = ts
    return {"legit": np.array(st_legit, dtype=np.int8).reshape(-1, 4),
            "attack": np.array(st_attack, dtype=np.int8).reshape(-1, 4)}


def build_cache(zip_or_csv: str, cache_dir: str, user_fraction: float = 0.06, warmup_days: int = 60,
                max_rows: int | None = None, report=None) -> dict:
    """One streaming pass over the dataset (zip or csv); keeps a stable hash sample of users."""
    t0 = time.time()
    if zip_or_csv.endswith(".zip"):
        zf = zipfile.ZipFile(zip_or_csv)
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.open(member)
        size = zf.getinfo(member).file_size
    else:
        raw, size, member = open(zip_or_csv, "rb"), os.path.getsize(zip_or_csv), os.path.basename(zip_or_csv)
    fh = io.TextIOWrapper(raw, encoding="utf-8", newline="")
    rd = csv.reader(fh)
    header = [h.strip().lower() for h in next(rd)]
    ix = {k: header.index(v) for k, v in _COLS.items()}
    thr = int(user_fraction * 2 ** 64)
    users, dev_ids, net_ids = {}, {}, {}
    cols = {k: [] for k in ("user", "ts", "dev", "net", "success", "attack")}
    n_rows = 0
    for row in rd:
        n_rows += 1
        if max_rows and n_rows > max_rows:
            break
        if report and n_rows % 1_000_000 == 0:
            report(n_rows, 0, "rows-read")
        uid = row[ix["user"]]
        uk = _user_key(uid)
        if uk >= thr:
            continue
        u = users.setdefault(uid, len(users))
        dkey = (_family(row[ix["os"]]), _family(row[ix["browser"]]), row[ix["device"]].strip().lower())
        nkey = (row[ix["country"]].strip(), row[ix["asn"]].strip())
        cols["user"].append(u)
        cols["ts"].append(_parse_ts(row[ix["ts"]]))
        cols["dev"].append(dev_ids.setdefault(dkey, len(dev_ids)))
        cols["net"].append(net_ids.setdefault(nkey, len(net_ids)))
        cols["success"].append(_bool(row[ix["success"]]))
        cols["attack"].append(_bool(row[ix["attack_ip"]]) or _bool(row[ix["ato"]]))
    fh.close()
    rows = np.zeros(len(cols["user"]), dtype=[("user", np.int64), ("ts", np.int64), ("dev", np.int64),
                                              ("net", np.int64), ("success", bool), ("attack", bool)])
    for k, v in cols.items():
        rows[k] = v
    rows = rows[np.lexsort((rows["ts"], rows["user"]))]
    st = derive_states(rows, warmup_days * DAY)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "states_legit.npy"), st["legit"])
    np.save(os.path.join(cache_dir, "states_attack.npy"), st["attack"])
    meta = {"cache_version": CACHE_VERSION, "source_member": member, "source_bytes": size, "rows_read": n_rows,
            "user_fraction": user_fraction, "users_sampled": len(users), "rows_sampled": int(len(rows)),
            "warmup_days": warmup_days, "legit_states": int(len(st["legit"])),
            "attack_states": int(len(st["attack"])), "device_combos": len(dev_ids), "network_combos": len(net_ids),
            "first_ts": int(rows["ts"].min()) if len(rows) else None,
            "last_ts": int(rows["ts"].max()) if len(rows) else None, "chains": list(CHAINS),
            "levels": [list(a.levels) for a in RBA_SCHEMA.attributes], "citation": CITATION,
            "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def ensure_cache(zip_path: str, cache_dir: str, user_fraction: float, warmup_days: int, allow_download: bool = True,
                 report=None) -> dict:
    mp = os.path.join(cache_dir, "meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
        if meta.get("cache_version") == CACHE_VERSION and meta.get("user_fraction") == user_fraction \
                and meta.get("warmup_days") == warmup_days:
            return meta
    if not os.path.exists(zip_path):
        if not allow_download:
            raise FileNotFoundError(zip_path)
        from .source import download
        download(zip_path, URL, report)
    got = md5_file(zip_path)
    if got != MD5:
        raise ValueError(f"RBA dataset checksum mismatch: {got} != {MD5} (published on Zenodo)")
    meta = build_cache(zip_path, cache_dir, user_fraction, warmup_days, report=report)
    meta["md5"] = got
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_states(cache_dir: str, kind: str = "legit") -> np.ndarray:
    return np.load(os.path.join(cache_dir, f"states_{kind}.npy"))


def empirical_prior(states: np.ndarray, pseudo: float = 0.0):
    """Empirical joint over the four chains (4^4 states). pseudo > 0 adds Laplace smoothing."""
    from ..prior import Prior
    pr = Prior(RBA_SCHEMA.dims)
    counts = np.bincount(states.astype(np.int64) @ pr._radix, minlength=len(pr.states)).astype(float) + pseudo
    pr.p = counts / counts.sum()
    pr.counts = counts - pseudo
    return pr


# ---------------------------------------------------------------------------- positive association (FKG)
def pairwise_association(states: np.ndarray, z: float = 3.0) -> list[dict]:
    """For every chain pair: TP2 (adjacent 2x2 log odds ratios >= 0, the pairwise FKG lattice condition) and
    positive quadrant dependence P(Xi>=a, Xj>=b) >= P(Xi>=a) P(Xj>=b). A violation is significant when the
    estimate is below zero by more than z standard errors (Woolf SE for log odds ratios, binomial SE else)."""
    n = len(states)
    out = []
    d = RBA_SCHEMA.dims
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            T = np.zeros((d[i], d[j]))
            np.add.at(T, (states[:, i], states[:, j]), 1)
            minors = sig = 0
            worst = math.inf
            for a in range(d[i] - 1):
                for b in range(d[j] - 1):
                    c = T[a:a + 2, b:b + 2]
                    if (c == 0).any():
                        continue
                    lor = math.log(c[0, 0] * c[1, 1] / (c[0, 1] * c[1, 0]))
                    se = math.sqrt((1 / c).sum())
                    minors += 1
                    worst = min(worst, lor)
                    sig += lor + z * se < 0
            P = T / n
            pq_sig, pq_worst = 0, math.inf
            for a in range(1, d[i]):
                for b in range(1, d[j]):
                    joint = P[a:, b:].sum()
                    diff = joint - P[a:, :].sum() * P[:, b:].sum()
                    se = math.sqrt(max(joint * (1 - joint), 1e-12) / n)
                    pq_worst = min(pq_worst, diff)
                    pq_sig += diff + z * se < 0
            out.append({"chain_i": CHAINS[i], "chain_j": CHAINS[j], "n": n, "tp2_minors": minors,
                        "tp2_significant_violations": sig, "min_log_odds_ratio": worst,
                        "pqd_significant_violations": pq_sig, "min_pqd_diff": pq_worst,
                        "spearman": _spearman(states[:, i], states[:, j])})
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        vals, inv = np.unique(x, return_inverse=True)
        sums = np.bincount(inv, weights=r)
        return (sums / np.bincount(inv))[inv]
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def lattice_condition(prior, z: float = 3.0) -> dict:
    """Full-joint FKG check pi(x v y) pi(x ^ y) >= pi(x) pi(y) over all state pairs with counts, separating
    structural zeros (x v y or x ^ y never observed while x, y are) from significant numeric violations."""
    c = prior.counts
    S = prior.states
    I, J = np.triu_indices(len(S), 1)
    up = np.maximum(S[I], S[J]) @ prior._radix
    lo = np.minimum(S[I], S[J]) @ prior._radix
    comparable = (up == I) | (up == J)
    I, J, up, lo = I[~comparable], J[~comparable], up[~comparable], lo[~comparable]
    pos = (c[I] > 0) & (c[J] > 0)
    zero = pos & ((c[up] == 0) | (c[lo] == 0))
    full = pos & ~zero
    lr = np.log(c[up[full]]) + np.log(c[lo[full]]) - np.log(c[I[full]]) - np.log(c[J[full]])
    se = np.sqrt(1 / c[up[full]] + 1 / c[lo[full]] + 1 / c[I[full]] + 1 / c[J[full]])
    return {"pairs_tested": int(pos.sum()), "structural_zero_violations": int(zero.sum()),
            "point_violations": int((lr < 0).sum()), "significant_violations": int((lr + z * se < 0).sum()),
            "share_significant": float((lr + z * se < 0).sum() / max(full.sum(), 1))}
