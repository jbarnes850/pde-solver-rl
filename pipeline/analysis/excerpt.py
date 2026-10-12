import sys
import sys,json
sys.path.insert(0, __import__("os").path.dirname(__file__))
from getrow import get
# usage: excerpt.py set ck case idx ranges(a-b,c-d) [keepcomments]
s,ck,case,idx,ranges=sys.argv[1:6]; keep=len(sys.argv)>6
r=get(s,ck,case,int(idx)); L=r["code"].splitlines()
print(f"# total lines {len(L)}")
for rg in ranges.split(","):
    a,b=map(int,rg.split("-"))
    for i in range(a,min(b+1,len(L))):
        l=L[i]
        if not keep and l.strip().startswith("#"): continue
        if not l.strip(): continue
        print(f"{i:4d}| {l}")
    print("   ...")
