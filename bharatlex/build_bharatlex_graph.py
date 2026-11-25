"""Build graph.json for the BharatLex 3D explorer from the cleaned section files.
Nodes : 3 acts (hubs), every section (Hindi+English text), 12 legal-theme tags
Edges : section->act, section->theme (keyword rules), section->section cross-references
        ("section 103 ..."), section<->section similarity (TF-IDF top-3, English text)
Usage : python build_bharatlex_graph.py DATA_DIR graph.json
"""
import json, re, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

D, OUT = sys.argv[1], sys.argv[2]
FILES = {"BNS": "bns_aligned_fixed.json", "BNSS": "BNSS_sections_fixed.json",
         "BSA": "IEA_sections_fixed.json"}
ACTNAME = {"BNS": "Bharatiya Nyaya Sanhita", "BNSS": "Bharatiya Nagarik Suraksha Sanhita",
           "BSA": "Bharatiya Sakshya Adhiniyam"}
THEMES = {"death penalty": r"\bdeath\b", "life imprisonment": r"imprisonment for life",
  "fine": r"\bfine\b", "women": r"\bwom[ae]n\b|\bfemale\b", "children": r"\bchild(ren)?\b|\bminor\b",
  "public servant": r"public servant", "property": r"\bproperty\b|\btheft\b|\brobbery\b",
  "hurt & violence": r"\bhurt\b|\bassault\b|\bgrievous\b", "murder & homicide": r"\bmurder\b|\bhomicide\b",
  "bail & arrest": r"\bbail\b|\barrest", "evidence & witness": r"\bevidence\b|\bwitness",
  "court & trial": r"\btrial\b|\bcourt\b|\bmagistrate\b"}
XACT = [("BNSS", r"nagarik suraksha"), ("BNS", r"nyaya sanhita"), ("BSA", r"sakshya")]

nodes, links, text = [], [], {}
for a, f in FILES.items():
    nodes.append({"id": a, "type": "act", "label": a + " – " + ACTNAME[a], "val": 40})
    for k, v in json.load(open(f"{D}/{f}", encoding="utf-8")).items():
        nid = f"{a}:{k}"
        en = v.get("english") or ""
        nodes.append({"id": nid, "type": "section", "act": a, "label": f"{a} §{k}",
                      "en": en, "hi": v.get("hindi") or "", "val": 2})
        text[nid] = en
        links.append({"source": nid, "target": a, "kind": "in_act"})

ids = set(text)
for nid, t in text.items():
    a = nid.split(":")[0]
    for name, rx in THEMES.items():
        if re.search(rx, t, re.I):
            links.append({"source": nid, "target": "T:" + name, "kind": "theme"})
    for m in re.finditer(r"[Ss]ections?\s+(\d+)([^.;]{0,90})", t):
        tgt_act = a
        for ac, rx in XACT:
            if re.search(rx, m.group(2), re.I): tgt_act = ac; break
        tid = f"{tgt_act}:{m.group(1)}"
        if tid in ids and tid != nid:
            links.append({"source": nid, "target": tid, "kind": "refers_to"})
for name in THEMES:
    nodes.append({"id": "T:" + name, "type": "theme", "label": name, "val": 12})

order = list(text); X = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True
                                          ).fit_transform([text[i] for i in order])
for i, nid in enumerate(order):
    sims = linear_kernel(X[i], X).ravel(); sims[i] = 0
    for j in sims.argsort()[::-1][:3]:
        if sims[j] > 0.35:
            links.append({"source": nid, "target": order[j], "kind": "similar"})

seen, uniq = set(), []                       # drop duplicate / reversed edges
for l in links:
    key = (frozenset((l["source"], l["target"])), l["kind"])
    if key not in seen: seen.add(key); uniq.append(l)
deg = {}
for l in uniq:
    for e in (l["source"], l["target"]): deg[e] = deg.get(e, 0) + 1
for n in nodes:
    if n["type"] == "section": n["val"] = 1 + deg.get(n["id"], 0) ** 0.5
json.dump({"nodes": nodes, "links": uniq}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
from collections import Counter
print(len(nodes), "nodes,", len(uniq), "links", dict(Counter(l["kind"] for l in uniq)))
