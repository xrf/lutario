import subprocess, sys
sys.path.insert(1, "../_urandom")
from utils import *

fn = "src/imsrg.hpp"
s = load_file(fn)

s = subprocess.run(["clang-format"], input=s, check=True,
                   stdout=subprocess.PIPE, universal_newlines=True).stdout

# remove the 'rfl' stuff
s = re.sub(r"brfl\((B\d\[l\w+\])\((u\w+), (u\w+)\)\)",
           r"\1(\3, \2)", s)
s = re.sub(r"arfl\((A\d\[l\w+\])\((u\w+), (u\w+)\)\)",
           r"\1(\3, \2)", s)

# remove the signed view indices
s = re.sub(r"\b(u\w+)_\b",
           r"\1", s)

# convert getters to new format
s = re.sub(r"([A-Z])\d(\d?\[)", r"\1\2", s)
s = re.sub(r"(\w+)\[l\w\]\(u(\w), u(\w)\)",
           r"GET_1(\1, \2, \3)", s)
s = re.sub(r"(\w+)\[l\w\w\]\(u(\w)(\w), u(\w)(\w)\)",
           r"GET_2(\1, \2, \3, \4, \5)", s)

s = re.sub(r"for \(auto &&l(\w) : basis_1\.channels\(\)\)",
           r"for (CHANNEL_1(\1))", s)
s = re.sub(r"for \(auto &&l(\w)(\w) : basis_2\.channels\(\)\)",
           r"for (CHANNEL_2(\1, \2))", s)

s = re.sub(r"for \(auto &&u(\w) : basis_1.subindices\(l(\w)\)\)",
           r"for (STATE_1(\1, {0, 2})) /* l\1 <- l\2 */", s)
s = re.sub(r"for \(auto &&u(\w) : basis_1.subindices\(l(\w), 0\)\)",
           r"for (STATE_1(\1, {0, 1})) /* l\1 <- l\2 */", s)
s = re.sub(r"for \(auto &&u(\w) : basis_1.subindices\(l(\w), 1\)\)",
           r"for (STATE_1(\1, {1, 2})) /* l\1 <- l\2 */", s)
s = re.sub(r"for \(auto &&u(\w)(\w) : basis_2.subindices\(l(\w)(\w)\)\)",
           r"for (STATE_2(\1, \2, {0, 4})) /* l\1\2 <- l\3\4 */", s)
s = re.sub(r"for \(auto &&u(\w)(\w) : basis_2.subindices\(l(\w)(\w), 0\)\)",
           r"for (STATE_2(\1, \2, {0, 1})) /* l\1\2 <- l\3\4 */", s)
s = re.sub(r"for \(auto &&u(\w)(\w) : basis_2.subindices\(l(\w)(\w), 2\)\)",
           r"for (STATE_2(\1, \2, {3, 4})) /* l\1\2 <- l\3\4 */", s)

s = subprocess.run(["clang-format"], input=s, check=True,
                   stdout=subprocess.PIPE, universal_newlines=True).stdout

save_file(fn, s)
