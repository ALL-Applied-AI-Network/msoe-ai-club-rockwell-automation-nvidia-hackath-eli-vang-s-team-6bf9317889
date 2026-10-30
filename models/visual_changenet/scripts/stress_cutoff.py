import csv, sys
def load(p): return [(r["label"], float(r["siamese_score"]), r["object_name"]) for r in csv.DictReader(open(p))]
for run in ("iter1","control"):
    b=f"{sys.argv[1]}/r1/{run}"
    st=load(f"{b}/inference_val_stress/inference.csv"); te=load(f"{b}/inference_test/inference.csv"); va=load(f"{b}/inference_val/inference.csv")
    thr=min(s for l,s,_ in st if l!="PASS")
    def ev(rows,t):
        d=[(l,s) for l,s,_ in rows if l!="PASS"]; g=[s for l,s,_ in rows if l=="PASS"]
        miss={}
        for l,s in d:
            if s<t: miss[l]=miss.get(l,0)+1
        fa=sum(s>=t for s in g); return len(d),miss,fa,len(g)
    print(run,"stress catch-all thr",thr)
    for n,r in (("val",va),("stress",st),("test",te)):
        nd,m,fa,ng=ev(r,thr); print(f"  {n}: defects {nd} missed {m} FA {fa}/{ng} = {100*fa/ng:.2f}%")
    # weakest stressed defect
    print("  weakest stressed defects", sorted([(s,l,o) for l,s,o in st if l!="PASS"])[:3])
    print("  top stressed passes", sorted([(s,o) for l,s,o in st if l=="PASS"],reverse=True)[:3])
    print("  test max pass", max(s for l,s,_ in te if l=="PASS"), "min defect", min(s for l,s,_ in te if l!="PASS"))
