
from json_tricks import dump, dumps, load, loads

File = "C:\\Users\\YUSTE\\Desktop\\test.json"



config = {
    "preprocessing": {
        "shutter artifact length": 1000,
        "grouped-z project bin size": 3,
        "median filter tensor size": (7, 3, 3)
    },
    "postprocesssing": {
        "color": "blue",
        "name": "test"
    }
}

_s = dumps(config, indent=0, maintain_tuples=True)

with open(File, "w") as f:
    f.write(_s)
f.close()

with open(File, "r") as j:
    content = loads(j.read())
j.close()

print(content)
