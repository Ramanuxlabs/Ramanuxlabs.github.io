import pickle, json, re, sys, collections
import numpy as np, pandas as pd, faiss
R, OUT = sys.argv[1], sys.argv[2]
ws=lambda s: re.sub(r"\s+"," ",str(s)).strip()
ch=pickle.load(open(R+"/index/chunks.pkl","rb"))
idx=faiss.read_index(R+"/index/faiss_index.bin"); assert idx.ntotal==len(ch)
V=idx.reconstruct_n(0,idx.ntotal).astype("float32"); V/=np.linalg.norm(V,axis=1,keepdims=True)
NAME={"Digital Personal Data Protection Rules 2025":"DPDP Rules 2025","Privacy, Security and Ownership of Data in the Telecom Sector Takshashila Policy Advisory":"TRAI Telecom Advisory","CSCRF":"SEBI CSCRF","ISO 27001:2022":"ISO/IEC 27001:2022","State Data Centre security guidelines":"State DC Security Guidelines","Cyber Security in Power Sector 2021":"Cyber Security Power Sector","NCIIPC Guidelines for Government Entities":"NCIIPC Guidelines (Govt. Entities)"}
GROUP={"DPDP Act 2023":"in-law","DPDP Rules 2025":"in-law","IT Act 2000":"in-law","Electricity Act 2003":"in-law","NCIIPC Rules":"in-law",
 "GDPR":"eu","NIST SP 800-66r2":"intl","NIST Cybersecurity Framework":"intl","ISO/IEC 27001:2022":"intl","OECD Digital Security Guidelines":"intl","PCI Security Standards":"intl"}
dn=lambda d: NAME.get(d,d)
docs=sorted({dn(c["doc_name"]) for c in ch})
nodes=[]; links=[]
for d in docs: nodes.append(dict(id="D:"+d,t="act",g=GROUP.get(d,"in-guide"),name=d))
norm=[ws(c["text"]) for c in ch]
for i,c in enumerate(ch):
    d=dn(c["doc_name"])
    nodes.append(dict(id=f"P:{i}",t="psg",g=GROUP.get(d,"in-guide"),doc=d,ci=c["chunk_idx"],name=f"{d} · passage {c['chunk_idx']}",text=norm[i]))
    links.append(dict(source="D:"+d,target=f"P:{i}",k="part"))
# window adjacency (consecutive chunks of one document, as the pipeline's ±1 window)
for i in range(len(ch)-1):
    if ch[i]["doc_name"]==ch[i+1]["doc_name"]: links.append(dict(source=f"P:{i}",target=f"P:{i+1}",k="win"))
# similarity: top-3 BGE neighbours outside the ±1 window, cosine >= 0.80
S=V@V.T; np.fill_diagonal(S,-1); seen=set(); nsim=0; cross=collections.Counter()
for i in range(len(ch)):
    order=np.argsort(-S[i])
    got=0
    for j in order[:12]:
        if S[i,j]<0.80 or got>=3: break
        if ch[i]["doc_name"]==ch[j]["doc_name"] and abs(i-j)<=1: continue
        key=(min(i,j),max(i,j))
        got+=1
        if key in seen: continue
        seen.add(key); x=ch[i]["doc_name"]!=ch[j]["doc_name"]
        links.append(dict(source=f"P:{key[0]}",target=f"P:{key[1]}",k="sim",w=round(float(S[i,j]),3),x=int(x))); nsim+=1
        if x: cross[tuple(sorted((dn(ch[i]['doc_name']),dn(ch[j]['doc_name']))))]+=1
# questions
SOURCE_CANON=[("e-Governance Policy Initiatives","e-Governance Policy India"),("General Data Protection Regulation","GDPR"),("GDPR","GDPR"),("OECD/LEGAL/0479","OECD Digital Security Guidelines"),("NIST SP 800-66r2","NIST SP 800-66r2"),("DPDP Act 2023","DPDP Act 2023"),("NCIIPC Guidelines","NCIIPC Guidelines"),("ISO 27001:2022","ISO/IEC 27001:2022"),("ISO/IEC 27001:2022","ISO/IEC 27001:2022"),("Presentation - CSCRF","SEBI CSCRF"),("State Data Centre security guidelines","State DC Security Guidelines"),("Digital Personal Data Protection Rules, 2025","DPDP Rules 2025"),("PCI Security Standards","PCI Security Standards"),("Takshashila","TRAI Telecom Advisory")]
PART=[("CELEX_32016R0679","GDPR"),("DPDP_Rules","DPDP Rules 2025"),("DPDP ACT","DPDP Act 2023"),("NCIIPC_Guidelines","NCIIPC Guidelines"),("NCIIPC-Rules","NCIIPC Rules"),("CERT-In_Directions","CERT-In Directions 2022")]
def canon(l):
    l=str(l)
    if " + " in l or l.strip()=="Cross-Document": return "CROSS"
    for s,n in SOURCE_CANON:
        if s in l: return n
def part(p):
    for s,n in PART:
        if s in p: return n
    return p.strip()
def f1(p,g):
    pt=p.lower().split(); gt=g.lower().split(); c=set(pt)&set(gt)
    if not c: return 0.0
    pr=len(c)/len(pt); rc=len(c)/len(gt); return 2*pr*rc/(pr+rc)
bydoc=collections.defaultdict(list)
for i,c in enumerate(ch): bydoc[dn(c["doc_name"])].append(i)
def locate(doc,span):
    span=ws(span); ids=bydoc.get(doc,[])
    for n in (120,60,30):
        for i in ids:
            if span[:n] in norm[i]: return i
    for i in ids:
        if span[-40:] in norm[i]: return i
    return None
ds=pd.read_csv(R+"/dataset/regulationqa_v1.csv"); log=json.load(open(R+"/evaluation/extraction_log.json"))
res=pd.read_csv(R+"/evaluation/extraction_results.csv")
miss=0; qn=0; docset={'D:'+d for d in docs}
for row,e,r in zip(ds.itertuples(),log,res.itertuples()):
    src=canon(row.document_source); lab=str(row.document_source)
    parts=[part(p) for p in lab.split(" + ")] if " + " in lab else ([] if src=="CROSS" else [src])
    cands=[]
    for c in e["retrieved"]:
        d=dn(c["doc_name"]); pid=None
        for s in c["extracted_spans"]:
            pid=locate(d,s)
            if pid is not None: break
        cands.append(dict(p=pid,d=d,ce=round(c["top_ce_score"],2),rrf=round(c["rrf_score"],4)))
    t1=locate(dn(r.top1_doc),r.top1_span)
    if t1 is None: miss+=1
    tf=round(f1(str(r.top1_span),row.ground_truth_answer),3)
    nodes.append(dict(id=f"Q:{row.id}",t="q",name=row.question,src=src,parts=parts,gold=ws(row.ground_truth_answer),span=ws(r.top1_span),sd=dn(r.top1_doc),f=tf,c=cands))
    if t1 is not None: links.append(dict(source=f"Q:{row.id}",target=f"P:{t1}",k="ans",ok=int(tf>=0.5)))
    for d in dict.fromkeys(parts):
        if "D:"+d in docset: links.append(dict(source=f"Q:{row.id}",target="D:"+d,k="src"))
json.dump(dict(nodes=nodes,links=links),open(OUT,"w"),separators=(",",":"),ensure_ascii=False,allow_nan=False)
cnt=collections.Counter(l["k"] for l in links)
print("nodes",len(nodes),"links",dict(cnt),"top1 unlocated",miss)
print("cross-instrument similar pairs:",cross.most_common(8))
# passage embeddings for the live query (float16, index order), written next to graph.json
import os
V.astype(np.float16).tofile(os.path.join(os.path.dirname(os.path.abspath(OUT)), "vectors.f16"))
