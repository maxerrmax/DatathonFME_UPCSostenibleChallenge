"""python train_any.py <A|B> <sortida.pkl> [train_script]: entrena models 'fit' amb la partició indicada."""
import sys, os, shutil, runpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
R = "/home/user/DatathonFME_UPCSostenibleChallenge"
sys.path.insert(0, R + "/estandard"); sys.path.insert(0, R + "/dificil")
import data
from splits import split_B, split_C
if sys.argv[1] in ("B", "C"):
    data.split_val = split_B if sys.argv[1] == "B" else split_C
script = sys.argv[3] if len(sys.argv) > 3 else R + "/dificil/train_v7.py"
out = R + "/dificil/agent_hard/models_fit7.pkl"
bak = out + ".bak"; shutil.copy(out, bak)
sys.argv = [script, "fit"]
try:
    runpy.run_path(script, run_name="__main__")
    shutil.move(out, sys.argv_out if False else os.environ["OUT"])
finally:
    shutil.move(bak, out)
