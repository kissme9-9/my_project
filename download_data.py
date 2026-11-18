"""批量下载 Montgomery 数据集（CXR_png + ManualMask/leftMask + rightMask）"""
import os
import urllib.request
import concurrent.futures
import time

BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet"
DATA_ROOT = r"D:\vibe_practice\lung_seg\data\MontgomerySet"
IMG_DIR = os.path.join(DATA_ROOT, "CXR_png")
LEFT_DIR = os.path.join(DATA_ROOT, "ManualMask", "leftMask")
RIGHT_DIR = os.path.join(DATA_ROOT, "ManualMask", "rightMask")

for d in [IMG_DIR, LEFT_DIR, RIGHT_DIR]:
    os.makedirs(d, exist_ok=True)

# 138 个文件名（从官方 index.html 提取）
FILENAMES = [
    "MCUCXR_0001_0.png", "MCUCXR_0002_0.png", "MCUCXR_0003_0.png", "MCUCXR_0004_0.png",
    "MCUCXR_0005_0.png", "MCUCXR_0006_0.png", "MCUCXR_0008_0.png", "MCUCXR_0011_0.png",
    "MCUCXR_0013_0.png", "MCUCXR_0015_0.png", "MCUCXR_0016_0.png", "MCUCXR_0017_0.png",
    "MCUCXR_0019_0.png", "MCUCXR_0020_0.png", "MCUCXR_0021_0.png", "MCUCXR_0022_0.png",
    "MCUCXR_0023_0.png", "MCUCXR_0024_0.png", "MCUCXR_0026_0.png", "MCUCXR_0027_0.png",
    "MCUCXR_0028_0.png", "MCUCXR_0029_0.png", "MCUCXR_0030_0.png", "MCUCXR_0031_0.png",
    "MCUCXR_0035_0.png", "MCUCXR_0038_0.png", "MCUCXR_0040_0.png", "MCUCXR_0041_0.png",
    "MCUCXR_0042_0.png", "MCUCXR_0043_0.png", "MCUCXR_0044_0.png", "MCUCXR_0045_0.png",
    "MCUCXR_0046_0.png", "MCUCXR_0047_0.png", "MCUCXR_0048_0.png", "MCUCXR_0049_0.png",
    "MCUCXR_0051_0.png", "MCUCXR_0052_0.png", "MCUCXR_0053_0.png", "MCUCXR_0054_0.png",
    "MCUCXR_0055_0.png", "MCUCXR_0056_0.png", "MCUCXR_0057_0.png", "MCUCXR_0058_0.png",
    "MCUCXR_0059_0.png", "MCUCXR_0060_0.png", "MCUCXR_0061_0.png", "MCUCXR_0062_0.png",
    "MCUCXR_0063_0.png", "MCUCXR_0064_0.png", "MCUCXR_0068_0.png", "MCUCXR_0069_0.png",
    "MCUCXR_0070_0.png", "MCUCXR_0071_0.png", "MCUCXR_0072_0.png", "MCUCXR_0074_0.png",
    "MCUCXR_0075_0.png", "MCUCXR_0077_0.png", "MCUCXR_0079_0.png", "MCUCXR_0080_0.png",
    "MCUCXR_0081_0.png", "MCUCXR_0082_0.png", "MCUCXR_0083_0.png", "MCUCXR_0084_0.png",
    "MCUCXR_0085_0.png", "MCUCXR_0086_0.png", "MCUCXR_0087_0.png", "MCUCXR_0089_0.png",
    "MCUCXR_0090_0.png", "MCUCXR_0091_0.png", "MCUCXR_0092_0.png", "MCUCXR_0094_0.png",
    "MCUCXR_0095_0.png", "MCUCXR_0096_0.png", "MCUCXR_0097_0.png", "MCUCXR_0099_0.png",
    "MCUCXR_0100_0.png", "MCUCXR_0101_0.png", "MCUCXR_0102_0.png", "MCUCXR_0103_0.png",
    "MCUCXR_0104_1.png", "MCUCXR_0108_1.png", "MCUCXR_0113_1.png", "MCUCXR_0117_1.png",
    "MCUCXR_0126_1.png", "MCUCXR_0140_1.png", "MCUCXR_0141_1.png", "MCUCXR_0142_1.png",
    "MCUCXR_0144_1.png", "MCUCXR_0150_1.png", "MCUCXR_0162_1.png", "MCUCXR_0166_1.png",
    "MCUCXR_0170_1.png", "MCUCXR_0173_1.png", "MCUCXR_0182_1.png", "MCUCXR_0188_1.png",
    "MCUCXR_0194_1.png", "MCUCXR_0195_1.png", "MCUCXR_0196_1.png", "MCUCXR_0203_1.png",
    "MCUCXR_0213_1.png", "MCUCXR_0218_1.png", "MCUCXR_0223_1.png", "MCUCXR_0228_1.png",
    "MCUCXR_0243_1.png", "MCUCXR_0251_1.png", "MCUCXR_0253_1.png", "MCUCXR_0254_1.png",
    "MCUCXR_0255_1.png", "MCUCXR_0258_1.png", "MCUCXR_0264_1.png", "MCUCXR_0266_1.png",
    "MCUCXR_0275_1.png", "MCUCXR_0282_1.png", "MCUCXR_0289_1.png", "MCUCXR_0294_1.png",
    "MCUCXR_0301_1.png", "MCUCXR_0309_1.png", "MCUCXR_0311_1.png", "MCUCXR_0313_1.png",
    "MCUCXR_0316_1.png", "MCUCXR_0331_1.png", "MCUCXR_0334_1.png", "MCUCXR_0338_1.png",
    "MCUCXR_0348_1.png", "MCUCXR_0350_1.png", "MCUCXR_0352_1.png", "MCUCXR_0354_1.png",
    "MCUCXR_0362_1.png", "MCUCXR_0367_1.png", "MCUCXR_0369_1.png", "MCUCXR_0372_1.png",
    "MCUCXR_0375_1.png", "MCUCXR_0383_1.png", "MCUCXR_0387_1.png", "MCUCXR_0390_1.png",
    "MCUCXR_0393_1.png", "MCUCXR_0399_1.png",
]

def download_one(args):
    name, url, dest = args
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return name, "skip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        return name, "ok"
    except Exception as e:
        return name, f"fail:{str(e)[:60]}"

# 构建下载任务列表
tasks = []
for name in FILENAMES:
    tasks.append((name, f"{BASE}/CXR_png/{name}", os.path.join(IMG_DIR, name)))
    tasks.append((name, f"{BASE}/ManualMask/leftMask/{name}", os.path.join(LEFT_DIR, name)))
    tasks.append((name, f"{BASE}/ManualMask/rightMask/{name}", os.path.join(RIGHT_DIR, name)))

print(f"Total tasks: {len(tasks)}")
t0 = time.time()
ok = skip = fail = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for i, (name, status) in enumerate(ex.map(download_one, tasks), 1):
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            fail += 1
            print(f"  FAIL {name}: {status}")
        if i % 30 == 0:
            print(f"  Progress: {i}/{len(tasks)} (ok={ok} skip={skip} fail={fail}) {time.time()-t0:.0f}s")

print(f"\nDone in {time.time()-t0:.0f}s. ok={ok} skip={skip} fail={fail}")
