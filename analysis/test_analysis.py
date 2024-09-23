from dcmnet.analysis import *

path = Path("/pchem-data/meuwly/boittier/home/jaxeq/misc")
paths = list(path.glob("*"))
for path in paths:
    multipoles(path)
