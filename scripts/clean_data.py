import re
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "data.csv"
VEC_PATH = PROJECT_DIR / "data" / "processed" / "data_vectorized.csv"

def read_csv_with_fallback(path):
    for enc in ["utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"无法读取文件: {path}")

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    return "" if x.lower() == "nan" else x

def sanitize_name(x):
    x = re.sub(r"\s+", "", str(x))
    x = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", x)
    x = x.strip("_")
    return x if x else "unknown"

def convert_date(x):
    if pd.isna(x):
        return pd.NaT

    s = str(x).strip().replace("日", "")

    # Excel 日期序列号
    if re.fullmatch(r"\d+(\.0+)?", s):
        n = int(float(s))
        return pd.to_datetime("1899-12-30") + pd.to_timedelta(n, unit="D")

    # 3.28 / 03.28 -> 2023-03-28
    md = re.match(r"^\s*(\d{1,2})\.(\d{1,2})\s*$", s)
    if md:
        m = int(md.group(1))
        d = int(md.group(2))
        return pd.to_datetime(f"2023-{m:02d}-{d:02d}", errors="coerce")

    return pd.to_datetime(s, errors="coerce")

def parse_gender(x):
    text = clean_text(x).upper()
    if text in {"男", "M", "MALE", "男性"} or text.startswith("男"):
        return 1
    if text in {"女", "F", "FEMALE", "女性"} or text.startswith("女"):
        return 0
    return pd.NA

def parse_asa(x):
    text = clean_text(x).upper()
    if text == "":
        return pd.NA

    text = (
        text.replace("ASA", "")
        .replace("级", "")
        .replace("：", "")
        .replace(":", "")
        .replace("Ⅰ", "I")
        .replace("Ⅱ", "II")
        .replace("Ⅲ", "III")
        .replace("Ⅳ", "IV")
        .replace("Ⅴ", "V")
        .replace("Ⅵ", "VI")
        .strip()
    )
    roman_map = {"VI": 6, "V": 5, "IV": 4, "III": 3, "II": 2, "I": 1}
    for roman, value in roman_map.items():
        if text.startswith(roman):
            return value

    m = re.search(r"[1-6]", text)
    return int(m.group()) if m else pd.NA

def classify_surgery_site(x):
    text = clean_text(x)
    if text == "":
        return "其他"

    obstetrics_keywords = [
        "剖宫产", "产科", "分娩", "胎儿", "妊娠", "子宫下段剖宫产", "子宫下段横切口"
    ]
    gyne_keywords = [
        "子宫", "卵巢", "输卵管", "宫颈", "妇科", "宫腔", "子宫肌瘤"
    ]

    ortho_spine_keywords = ["脊柱", "胸椎", "腰椎", "颈椎", "椎体", "椎弓根", "椎间"]
    ortho_upper_keywords = [
        "上肢", "肩", "锁骨", "肱骨", "尺骨", "桡骨", "腕", "手", "掌", "肘", "臂", "指"
    ]
    ortho_lower_keywords = [
        "下肢", "髋", "膝", "股骨", "胫", "腓", "踝", "足", "髌", "跟骨", "髋关节", "膝关节"
    ]
    ortho_general_keywords = [
        "骨", "关节", "骨折", "置换", "内固定", "截骨", "骨科"
    ]

    upper_abd_keywords = [
        "胃", "肝", "胆", "胰", "脾", "食管", "十二指肠", "胆囊"
    ]
    lower_abd_keywords = [
        "阑尾", "结肠", "直肠", "小肠", "乙状结肠", "回盲", "盲肠"
    ]
    abdomen_general_keywords = [
        "腹", "腹腔镜", "肠", "消化", "疝"
    ]

    if any(k in text for k in obstetrics_keywords):
        return "产科"
    if any(k in text for k in gyne_keywords):
        return "妇科"

    if any(k in text for k in ortho_spine_keywords):
        return "骨科-脊柱"
    if any(k in text for k in ortho_upper_keywords):
        return "骨科-上肢"
    if any(k in text for k in ortho_lower_keywords):
        return "骨科-下肢"
    if any(k in text for k in ortho_general_keywords):
        return "骨科-其他"

    if any(k in text for k in upper_abd_keywords):
        return "腹外科-上腹"
    if any(k in text for k in lower_abd_keywords):
        return "腹外科-下腹"
    if any(k in text for k in abdomen_general_keywords):
        return "腹外科-其他"

    return "其他"

def classify_block_site_categories(x):
    text = clean_text(x)
    if text == "":
        return {"未记录/未实施"}

    if any(k in text for k in ["无", "未做", "未行", "未实施", "未阻滞"]):
        return {"未记录/未实施"}

    neuraxial_keywords = ["硬膜外", "腰麻", "蛛网膜下腔", "蛛网膜下", "脊麻", "椎管内", "腰硬联合"]
    paravertebral_keywords = ["椎旁", "胸椎旁", "腰椎旁", "tpvb"]
    local_infiltration_keywords = ["切口浸润", "浸润", "局麻", "局部浸润"]

    lower_limb_peripheral_keywords = [
        "收肌管",
        "髂筋膜",
        "股神经",
        "坐骨神经",
        "坐骨",
        "腘窝",
        "隐神经",
        "胫神经",
        "股外侧皮",
    ]
    upper_limb_neck_keywords = ["臂丛", "肌间沟", "颈丛", "颈浅丛", "锁骨上", "锁骨下", "腋路", "颈神经通路"]
    abdominal_wall_keywords = ["腹横肌平面", "腹横平面", "腹横肌", "腹横", "tap", "腹直肌鞘", "腹直肌后鞘", "腹直肌"]
    chest_wall_keywords = ["胸肌", "pecs", "前锯肌", "肋间", "胸神经"]

    labels = split_multi_labels(text)
    if not labels:
        labels = [text]

    categories = set()
    for label in labels:
        label_l = label.lower()
        # "Ⅰ/Ⅱ型胸肌阻滞" 等字段拆分后会出现无意义片段，避免误入“其他”
        if re.fullmatch(r"(i{1,3}|iv|v|vi|[ⅠⅡⅢⅣⅤⅥ]+|[12])", label_l):
            continue
        if any(k.lower() in label_l for k in neuraxial_keywords):
            categories.add("椎管内镇痛")
        elif any(k.lower() in label_l for k in paravertebral_keywords):
            categories.add("椎旁阻滞")
        elif any(k.lower() in label_l for k in local_infiltration_keywords):
            categories.add("局部浸润")
        elif any(k.lower() in label_l for k in lower_limb_peripheral_keywords):
            categories.add("下肢外周神经阻滞")
        elif any(k.lower() in label_l for k in upper_limb_neck_keywords):
            categories.add("上肢/颈丛神经阻滞")
        elif any(k.lower() in label_l for k in abdominal_wall_keywords):
            categories.add("腹壁平面阻滞")
        elif any(k.lower() in label_l for k in chest_wall_keywords):
            categories.add("胸壁平面阻滞")
        elif ("神经阻滞" in label_l) or ("阻滞" in label_l and ("神经" in label_l or "平面" in label_l)):
            categories.add("其他外周神经阻滞")
        else:
            categories.add("其他")

    if not categories:
        categories.add("其他")

    return categories


def split_multi_labels(x):
    text = clean_text(x)
    if text == "":
        return []

    text = (
        text.replace("＋", "+")
        .replace("，", ",")
        .replace("、", ",")
        .replace("；", ";")
        .replace("／", "/")
        .replace("｜", "|")
    )
    text = (
        text.replace("Ⅰ/Ⅱ型胸肌", "ⅠⅡ型胸肌")
        .replace("I/II型胸肌", "II型胸肌")
        .replace("1/2型胸肌", "2型胸肌")
    )
    parts = re.split(r"[+,/;|]", text)
    labels = []
    skip_tokens = {"Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "I", "II", "III", "IV", "V", "VI", "1", "2"}
    for p in parts:
        p = p.strip()
        if p and p not in {"无", "未见", "none", "None"} and p not in skip_tokens:
            labels.append(p)
    return labels


def add_multilabel_onehot(input_df, col, min_count=10):
    created_cols = []
    if col not in input_df.columns:
        return created_cols

    label_series = input_df[col].apply(split_multi_labels)
    all_labels = [label for labels in label_series for label in labels]
    if not all_labels:
        return created_cols

    counts = pd.Series(all_labels).value_counts()
    keep_labels = counts[counts >= min_count].index.tolist()
    if not keep_labels:
        keep_labels = counts.index.tolist()
    keep_set = set(keep_labels)

    for label in keep_labels:
        new_col = f"{col}_oh_{sanitize_name(label)}"
        input_df[new_col] = label_series.apply(lambda labels, t=label: int(t in labels))
        created_cols.append(new_col)

    other_col = f"{col}_oh_其他"
    input_df[other_col] = label_series.apply(
        lambda labels: int(len(labels) > 0 and any(label not in keep_set for label in labels))
    )
    created_cols.append(other_col)
    return created_cols


def extract_drug_dose_mg(series, drug_name):
    def parse_one(text):
        text = clean_text(text)
        if text == "":
            return pd.NA

        # 1) 优先读取明确剂量：药名 + 数值 + 单位（ug/mg/g）
        m = re.search(
            rf"{re.escape(drug_name)}\s*([0-9]+(?:\.[0-9]+)?)\s*(ug|mg|g)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            dose = float(m.group(1))
            unit = m.group(2).lower()
            if unit == "ug":
                return dose / 1000.0
            if unit == "g":
                return dose * 1000.0
            return dose

        # 2) 没有明确mg时，读取“浓度% + 体积ml”
        # 例如：0.3%罗哌卡因25ml -> 0.3% = 3mg/ml, 总剂量=3*25=75mg
        m = re.search(
            rf"([0-9]+(?:\.[0-9]+)?)\s*%\s*{re.escape(drug_name)}.*?([0-9]+(?:\.[0-9]+)?)\s*ml",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            m = re.search(
                rf"{re.escape(drug_name)}\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*.*?([0-9]+(?:\.[0-9]+)?)\s*ml",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                pct = float(m.group(1))
                vol_ml = float(m.group(2))
                return pct * 10.0 * vol_ml
            return pd.NA

        pct = float(m.group(1))
        vol_ml = float(m.group(2))
        return pct * 10.0 * vol_ml

    return series.apply(parse_one)


def normalize_block_text(text):
    text = clean_text(text)
    if text == "":
        return ""
    text = (
        text.replace("％", "%")
        .replace("&", "%")
        .replace("，", ",")
        .replace("；", ";")
        .replace("：", ":")
        .replace("（", "(")
        .replace("）", ")")
        .replace("＋", "+")
        .replace("盐酸", "")
    )
    return text

def extract_volume_ml(text, prefer_final=True):
    """
    提取总体积(ml)：
    - 识别“稀释至/配至/共/至/:(冒号)”等最终体积
    - 识别“各20ml”“20ml/侧”并按双侧*2
    - 忽略速率“ml/h”
    """
    text = normalize_block_text(text)
    if text == "":
        return pd.NA

    if prefer_final:
        final_patterns = [
            r"(?:稀释至|配至|共|至)\s*([0-9]+(?:\.[0-9]+)?)\s*ml",
            r"[:]\s*([0-9]+(?:\.[0-9]+)?)\s*ml",
            r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*ml\s*\)",
        ]
        final_hits = []
        for pat in final_patterns:
            final_hits.extend(re.findall(pat, text, flags=re.IGNORECASE))
        if final_hits:
            return float(final_hits[-1])

    total = 0.0
    found = False
    for m in re.finditer(
        r"(各\s*)?([0-9]+(?:\.[0-9]+)?)\s*ml(?!\s*/\s*h)(\s*/\s*侧)?",
        text,
        flags=re.IGNORECASE,
    ):
        found = True
        val = float(m.group(2))
        mult = 2 if (m.group(1) is not None or m.group(3) is not None) else 1
        total += val * mult

    return total if found else pd.NA


def mg_from_value_unit(value, unit):
    v = float(value)
    u = unit.lower()
    if u == "ug":
        return v / 1000.0
    if u == "g":
        # 阻滞镇痛用药里极大概率存在 "133g" 误填（应为133mg），做纠偏
        if v >= 10:
            return v
        return v * 1000.0
    return v


def extract_block_drug_dose(series, drug_pattern):
    """
    统一输出“剂量”数值：
    1) 优先质量单位(ug/mg/g) -> mg
    2) 其次浓度% * 体积ml -> mg
    3) 再次仅体积可得时 -> ml(作为剂量近似)
    """
    def parse_one(text):
        text = normalize_block_text(text)
        if text == "":
            return pd.NA

        total_mg = 0.0
        found_mass = False

        # 药名后接剂量
        for dose, unit in re.findall(
            rf"{drug_pattern}\s*([0-9]+(?:\.[0-9]+)?)\s*(ug|mg|g)",
            text,
            flags=re.IGNORECASE,
        ):
            found_mass = True
            total_mg += mg_from_value_unit(dose, unit)

        # 剂量在前，药名在后
        for dose, unit in re.findall(
            rf"([0-9]+(?:\.[0-9]+)?)\s*(ug|mg|g)\s*{drug_pattern}",
            text,
            flags=re.IGNORECASE,
        ):
            found_mass = True
            total_mg += mg_from_value_unit(dose, unit)

        if found_mass:
            return total_mg

        # 浓度% * 体积ml
        total_from_pct = 0.0
        found_pct = False
        for m in re.finditer(
            rf"([0-9]+(?:\.[0-9]+)?)\s*%\s*{drug_pattern}[^+;,，；]*",
            text,
            flags=re.IGNORECASE,
        ):
            pct = float(m.group(1))
            vol = extract_volume_ml(m.group(0), prefer_final=False)
            if pd.notna(vol):
                found_pct = True
                total_from_pct += pct * 10.0 * float(vol)

        for m in re.finditer(
            rf"{drug_pattern}\s*([0-9]+(?:\.[0-9]+)?)\s*%[^+;,，；]*",
            text,
            flags=re.IGNORECASE,
        ):
            pct = float(m.group(1))
            vol = extract_volume_ml(m.group(0), prefer_final=False)
            if pd.notna(vol):
                found_pct = True
                total_from_pct += pct * 10.0 * float(vol)

        if found_pct:
            return total_from_pct

        # 只剩体积可解析时，用体积作为剂量近似
        total_vol = 0.0
        found_vol = False
        for seg in re.split(r"[+;,，；]", text):
            if re.search(drug_pattern, seg, flags=re.IGNORECASE):
                vol = extract_volume_ml(seg, prefer_final=False)
                if pd.notna(vol):
                    found_vol = True
                    total_vol += float(vol)

        if found_vol:
            return total_vol
        return pd.NA

    return series.apply(parse_one)


df = read_csv_with_fallback(DATA_PATH)
print("数据读取成功！")

# 1. 删除“补充_”列
df = df.loc[:, ~df.columns.str.contains("补充_")]

# 2. 删除无意义列
df = df.drop(columns=["病区", "床号", "麻醉医生"], errors="ignore")

# 3. 删除缺失值过高的列（保留你指定的术前/术中特征）
required_cols = [
    "年份",
    "手术日期",
    "性别",
    "年龄",
    "体重",
    "ASA分级",
    "手术名称",
    "麻醉方法",
    "镇痛方式",
    "镇痛泵配方",
    "总量",
    "背景量",
    "阻滞镇痛部位",
    "阻滞镇痛用药",
    "阻滞镇痛配方",
]
threshold = len(df) * 0.7
keep_mask = (df.notna().sum(axis=0) >= threshold) | df.columns.isin(required_cols)
df = df.loc[:, keep_mask]
# 明确删除缺失过多字段
df = df.drop(columns=["单次按压量", "锁定时间"], errors="ignore")

# 4. 日期标准化 + 衍生日期特征
date_series = df["手术日期"].apply(convert_date)
df["手术月份"] = date_series.dt.month
df["手术星期"] = date_series.dt.dayofweek
df["手术日期"] = date_series.dt.strftime("%Y-%m-%d")

# 5. "0/无" 列转数值
for col in ["手术当天_皮肤瘙痒", "术后第二天_皮肤瘙痒"]:
    if col in df.columns:
        s = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"无": "0", "未见": "0", "未见皮肤瘙痒": "0"})
        )
        df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

# 6. 明显数值列统一转为数值
numeric_cols = [
    "年份",
    "年龄",
    "体重",
    "总量",
    "背景量",
    "手术当天_静息痛",
    "手术当天_活动痛",
    "手术当天_镇静评分",
    "手术当天_活动状态",
    "手术当天_恶心呕吐",
    "术后第一天_静息痛",
    "术后第一天_活动痛",
    "术后第一天_镇静评分",
    "术后第一天_活动状态",
    "术后第一天_恶心呕吐",
    "术后第二天_静息痛",
    "术后第二天_活动痛",
    "术后第二天_镇静评分",
    "术后第二天_活动状态",
    "术后第二天_恶心呕吐",
    "术后第三天_静息痛",
    "术后第三天_活动痛",
    "术后第三天_镇静评分",
    "术后第三天_活动状态",
    "术后第三天_恶心呕吐",
]
for col in numeric_cols:
    if col in df.columns:
        s = df[col].astype(str).str.strip()
        s = s.str.extract(r"^(-?\d+(\.\d+)?)")[0]
        df[col] = pd.to_numeric(s, errors="coerce")

# 缺失值处理：日期/年份/月/周按上一行前向填充
if "手术日期" in df.columns:
    date_text = df["手术日期"].astype("string").str.strip()
    date_text = date_text.replace({"": pd.NA, "nan": pd.NA, "NaT": pd.NA})
    df["手术日期"] = date_text.ffill()
for col in ["年份", "手术月份", "手术星期"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").ffill()

# 缺失值处理：体重按全体中位数填补
if "体重" in df.columns:
    weight_median = pd.to_numeric(df["体重"], errors="coerce").median()
    df["体重"] = pd.to_numeric(df["体重"], errors="coerce").fillna(weight_median)
    # 特殊样本修正：年龄=77 且体重=8 -> 体重改为77，年龄改为59
    if "年龄" in df.columns:
        age_num = pd.to_numeric(df["年龄"], errors="coerce")
        weight_num = pd.to_numeric(df["体重"], errors="coerce")
        special_mask = (age_num == 77) & (weight_num == 8)
        df.loc[special_mask, "体重"] = 77
        df.loc[special_mask, "年龄"] = 59

# 结局变量列名（保留原始缺失，不做填补）
outcome_days = ["手术当天", "术后第一天", "术后第二天", "术后第三天"]
outcome_metrics = ["静息痛", "活动痛", "镇静评分", "活动状态", "恶心呕吐"]

# ===== 向量化（你要求的核心字段）=====
vector_cols = []

# 性别、ASA分级
if "性别" in df.columns:
    df["性别_num"] = df["性别"].apply(parse_gender).astype("Float64")
    vector_cols.append("性别_num")

if "ASA分级" in df.columns:
    df["ASA分级_num"] = df["ASA分级"].apply(parse_asa).astype("Float64")
    # 缺失值处理：ASA分级按众数填补
    asa_mode = df["ASA分级_num"].dropna().mode()
    if not asa_mode.empty:
        df["ASA分级_num"] = df["ASA分级_num"].fillna(asa_mode.iloc[0])
    vector_cols.append("ASA分级_num")

# 手术名称：10类手术类型分类
if "手术名称" in df.columns:
    df["手术类型类别"] = df["手术名称"].apply(classify_surgery_site)
    surgery_site_labels = [
        "妇科",
        "产科",
        "骨科-上肢",
        "骨科-下肢",
        "骨科-脊柱",
        "骨科-其他",
        "腹外科-上腹",
        "腹外科-下腹",
        "腹外科-其他",
        "其他",
    ]
    for label in surgery_site_labels:
        col_name = f"手术类型_{label}"
        df[col_name] = (df["手术类型类别"] == label).astype(int)
        vector_cols.append(col_name)

# 麻醉方法、镇痛方式 one-hot
for col in ["麻醉方法", "镇痛方式"]:
    vector_cols.extend(add_multilabel_onehot(df, col, min_count=10))

# 阻滞镇痛部位：按部位/技术重分类，单条记录可多标签
if "阻滞镇痛部位" in df.columns:
    block_site_categories = df["阻滞镇痛部位"].apply(classify_block_site_categories)
    block_site_labels = [
        "下肢外周神经阻滞",
        "上肢/颈丛神经阻滞",
        "腹壁平面阻滞",
        "胸壁平面阻滞",
        "椎管内镇痛",
        "椎旁阻滞",
        "局部浸润",
        "其他外周神经阻滞",
        "其他",
        "未记录/未实施",
    ]
    for label in block_site_labels:
        col_name = f"阻滞镇痛部位_{label}"
        df[col_name] = block_site_categories.apply(lambda s, t=label: int(t in s))
        vector_cols.append(col_name)

# 镇痛泵配方：结构化字段 + 文本剂量特征
for col in ["总量", "背景量"]:
    if col in df.columns:
        vector_cols.append(col)

if "镇痛泵配方" in df.columns:
    formula_text = df["镇痛泵配方"].apply(clean_text)
    drug_list = ["舒芬", "羟考酮", "曲马多", "帕洛诺司琼", "布托啡诺", "纳布啡", "地佐辛"]

    for drug in drug_list:
        has_col = f"镇痛泵配方_has_{sanitize_name(drug)}"
        dose_col = f"镇痛泵配方_{sanitize_name(drug)}_mg"
        df[has_col] = formula_text.str.contains(drug, regex=False).astype(int)
        df[dose_col] = extract_drug_dose_mg(formula_text, drug)
        # 规则：镇痛泵配方中未出现该药时，剂量填0；最终不保留NaN
        df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce").fillna(0.0)
        df.loc[df[has_col] == 0, dose_col] = 0.0
        # 业务修正：布托啡诺中的 1.0g 特殊值按 1mg 处理
        if drug == "布托啡诺":
            df.loc[df[dose_col] == 1000.0, dose_col] = 1.0
        vector_cols.extend([has_col, dose_col])

    volume_ml = pd.to_numeric(
        formula_text.str.extract(r"[,/]\s*([0-9]+(?:\.[0-9]+)?)\s*ml", expand=False),
        errors="coerce",
    )
    rate_ml_h = pd.to_numeric(
        formula_text.str.extract(r"([0-9]+(?:\.[0-9]+)?)\s*ml/h", expand=False),
        errors="coerce",
    )
    if "总量" in df.columns:
        df["总量"] = df["总量"].fillna(volume_ml)
    if "背景量" in df.columns:
        df["背景量"] = df["背景量"].fillna(rate_ml_h)
    # 业务修正：总量中的 1500ml 特殊值按 150ml 处理
    if "总量" in df.columns:
        df["总量"] = pd.to_numeric(df["总量"], errors="coerce")
        df.loc[df["总量"] == 1500.0, "总量"] = 150.0
    # 需求更新：总量/背景量缺失值默认分别填 200 / 4
    if "总量" in df.columns:
        df["总量"] = pd.to_numeric(df["总量"], errors="coerce").fillna(200.0)
    if "背景量" in df.columns:
        df["背景量"] = pd.to_numeric(df["背景量"], errors="coerce").fillna(4.0)

# 阻滞镇痛配方/用药：兼容两种列名
block_formula_col = None
if "阻滞镇痛配方" in df.columns:
    block_formula_col = "阻滞镇痛配方"
elif "阻滞镇痛用药" in df.columns:
    block_formula_col = "阻滞镇痛用药"

if block_formula_col:
    # 统一输出前缀，避免“配方/用药”列名差异影响下游
    block_feature_prefix = "阻滞镇痛用药"
    block_text = df[block_formula_col].apply(normalize_block_text)
    block_dose_cols = []

    # 每种药两列：有无(0/1) + 总剂量(优先mg，回退浓度*体积，再回退体积)
    block_drug_specs = [
        {"name": "罗哌卡因", "pattern": r"(?:罗哌卡因|罗哌)"},
        {"name": "布比卡因脂质体", "pattern": r"(?:布比卡因脂质体注射液|布比卡因脂质体|布比脂质体|脂质体布比卡因|脂胶体)"},
        {"name": "布比卡因", "pattern": r"(?:布比卡因)(?!脂质体)"},
        {"name": "利多卡因", "pattern": r"(?:利多卡因)"},
        {"name": "左旋布比卡因", "pattern": r"(?:左旋布比卡因)"},
    ]

    for spec in block_drug_specs:
        drug_name = spec["name"]
        drug_pattern = spec["pattern"]
        has_series = block_text.str.contains(drug_pattern, regex=True, na=False)
        # 数据中未出现的药物不创建特征列
        if has_series.sum() == 0:
            continue
        has_col = f"{block_feature_prefix}_has_{sanitize_name(drug_name)}"
        dose_col = f"{block_feature_prefix}_{sanitize_name(drug_name)}_dose"
        df[has_col] = has_series.astype(int)
        df[dose_col] = extract_block_drug_dose(block_text, drug_pattern)
        df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce").fillna(0.0)
        df.loc[df[has_col] == 0, dose_col] = 0.0
        block_dose_cols.append(dose_col)
        vector_cols.extend([has_col, dose_col])

    # 阻滞镇痛用药总量：所有阻滞药剂量加和；无用药填0
    block_total_dose_col = f"{block_feature_prefix}_总量"
    if block_dose_cols:
        df[block_total_dose_col] = df[block_dose_cols].sum(axis=1).fillna(0.0)
    else:
        df[block_total_dose_col] = 0.0
    vector_cols.append(block_total_dose_col)

    # 阻滞镇痛用药总体积（ml）
    total_vol_col = f"{block_feature_prefix}_总体积_ml"
    df[total_vol_col] = block_text.apply(lambda x: extract_volume_ml(x, prefer_final=True))
    df[total_vol_col] = pd.to_numeric(df[total_vol_col], errors="coerce").fillna(0.0)
    vector_cols.append(total_vol_col)

# 常见基础数值特征也保留进向量表
for col in ["年份", "年龄", "体重", "手术月份", "手术星期"]:
    if col in df.columns:
        vector_cols.append(col)

# 保留疼痛相关核心结局（手术当天 + 术后1/2/3天）
for day in outcome_days:
    for metric in outcome_metrics:
        col = f"{day}_{metric}"
        if col in df.columns:
            vector_cols.append(col)

vector_cols = list(dict.fromkeys(vector_cols))
df_vector = df[vector_cols].copy()

# 删除全零列
drop_zero_cols = ["镇痛方式_oh_其他", "阻滞镇痛部位_椎管内镇痛", "阻滞镇痛部位_局部浸润"]
df_vector = df_vector.drop(columns=drop_zero_cols, errors="ignore")

# 确保总量/背景量不为空（导出兜底）
default_fill = {"总量": 200.0, "背景量": 4.0}
for col in ["总量", "背景量"]:
    if col in df_vector.columns:
        df_vector[col] = pd.to_numeric(df_vector[col], errors="coerce").fillna(default_fill[col])
    else:
        df_vector[col] = default_fill[col]

df_vector.to_csv(VEC_PATH, index=False, encoding="utf-8-sig")

print(f"向量化完成，输出文件: {VEC_PATH}")
print(f"向量维度: {df_vector.shape}")
