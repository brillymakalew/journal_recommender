import os
import re
import json
from io import BytesIO

import pandas as pd
import streamlit as st
from openai import OpenAI

# Optional: TF-IDF prefilter
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ==========================
# OpenAI Client
# ==========================
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-5-mini"

st.set_page_config(page_title="Scopus Outlet & SDG Recommender", layout="wide")

# Quartile badge CSS
st.markdown(
    """
<style>
.q-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  color: white;
  margin-right: 6px;
}
.q1 { background-color: #15803d; } /* hijau */
.q2 { background-color: #1d4ed8; } /* biru */
.q3 { background-color: #f97316; } /* oranye */
.q4 { background-color: #dc2626; } /* merah */
</style>
""",
    unsafe_allow_html=True,
)

VISIBLE_COL_FILE = "journal_visible_columns.json"

# Session state defaults
if "role" not in st.session_state:
    st.session_state["role"] = None
if "fm_results" not in st.session_state:
    st.session_state["fm_results"] = None
if "fm_guidelines" not in st.session_state:
    st.session_state["fm_guidelines"] = ""
if "fm_selected_title" not in st.session_state:
    st.session_state["fm_selected_title"] = None

# global placeholder, akan diisi setelah login
outlet_df = None
sdg_dict = None


# ==========================
# Helper: column detector
# ==========================
def find_col(df, keywords):
    for kw in keywords:
        match = [c for c in df.columns if kw.lower() in c.lower()]
        if match:
            return match[0]
    return None


# ==========================
# Column visibility config
# ==========================
def load_visible_columns(all_columns):
    """Load list of columns that FM is allowed to see."""
    if os.path.exists(VISIBLE_COL_FILE):
        try:
            with open(VISIBLE_COL_FILE, "r", encoding="utf-8") as f:
                cols = json.load(f)
            cols = [c for c in cols if c in all_columns]
            if cols:
                return cols
        except Exception:
            pass

    # default columns if config not present
    default_candidates = [
        "Title",
        "Publisher",
        "ASJC Area",   # pakai teks, bukan code
        "Quartile",
        "Scope",
        "URL",
        "Coverage",
    ]
    return [c for c in default_candidates if c in all_columns]


def save_visible_columns(cols):
    with open(VISIBLE_COL_FILE, "w", encoding="utf-8") as f:
        json.dump(cols, f, ensure_ascii=False, indent=2)


def get_visible_columns(all_columns):
    return load_visible_columns(all_columns)


# ==========================
# ASJC mapping helper
# ==========================
def add_asjc_area_column(outlets: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah kolom 'ASJC Area' berdasarkan 'ASJC Code' menggunakan file ASJC1.xlsx.
    Satu jurnal bisa punya banyak ASJC code (misal '2738; 2742; 2204; 3612'),
    dan akan dipetakan ke beberapa kategori teks (dipisah dengan '; ').
    """
    if "ASJC Code" not in outlets.columns:
        outlets["ASJC Area"] = ""
        return outlets

    try:
        asjc_df = pd.read_excel("ASJC1.xlsx")
        asjc_df.columns = [c.strip() for c in asjc_df.columns]

        # Filter hanya baris CodeSystem = ASJC (kalau kolomnya ada)
        if "CodeSystem" in asjc_df.columns:
            asjc_df = asjc_df[
                asjc_df["CodeSystem"].astype(str).str.upper() == "ASJC"
            ]

        code_col = "Code"
        desc_col = "Description"
        if code_col not in asjc_df.columns or desc_col not in asjc_df.columns:
            outlets["ASJC Area"] = outlets["ASJC Code"]
            return outlets

        mapping = {}
        for _, row in asjc_df.iterrows():
            code_raw = row[code_col]
            if pd.isna(code_raw):
                continue
            code_str = str(code_raw)
            # buang .0 dari excel
            if code_str.endswith(".0"):
                code_str = code_str[:-2]
            code_str = code_str.strip()
            if not code_str:
                continue

            desc = str(row[desc_col]).strip()
            if not desc:
                continue

            # map 2 digit utama dan 4 digit xx00 ke deskripsi area
            two = code_str.zfill(2)[-2:]
            mapping[two] = desc
            mapping[two + "00"] = desc

        def map_asjc(val):
            if pd.isna(val):
                return ""
            s = str(val)
            # satu jurnal bisa punya banyak code dipisah ; , / spasi
            parts = re.split(r"[;,/]+", s)
            descs = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if p in mapping:
                    descs.append(mapping[p])
                    continue
                # coba 2 digit pertama
                if len(p) >= 2:
                    key2 = p[:2]
                    if key2 in mapping:
                        descs.append(mapping[key2])
            # hilangkan duplikat sambil jaga urutan
            unique_descs = []
            seen = set()
            for d in descs:
                if d and d not in seen:
                    seen.add(d)
                    unique_descs.append(d)
            return "; ".join(unique_descs)

        outlets["ASJC Area"] = outlets["ASJC Code"].apply(map_asjc)
    except Exception:
        # kalau gagal mapping, isi apa adanya
        outlets["ASJC Area"] = outlets["ASJC Code"]

    return outlets


# ==========================
# Load data + flags
# ==========================
def load_data():
    outlets = pd.read_excel("List Scopus Outlet.xlsx")
    sdg_keywords = pd.read_excel("SDGs Keyword.xlsx")

    outlets.columns = [c.strip() for c in outlets.columns]
    sdg_keywords.columns = [c.strip() for c in sdg_keywords.columns]

    # ------ Outlet columns ------
    title_col = find_col(outlets, ["title", "source title", "journal"])
    scope_col = find_col(outlets, ["scope", "aim"])
    publisher_col = find_col(outlets, ["publisher"])
    asjc_col = find_col(outlets, ["asjc", "subject"])
    quart_col = find_col(outlets, ["quartile", "best quartile", "sjr best"])
    link_col = find_col(outlets, ["scimago", "url", "link", "source url", "homepage"])

    if not title_col or not scope_col:
        raise ValueError(
            f"Tidak menemukan kolom judul/scope. Kolom tersedia: {list(outlets.columns)}"
        )

    rename_map = {title_col: "Title", scope_col: "Scope"}
    if publisher_col:
        rename_map[publisher_col] = "Publisher"
    if asjc_col:
        rename_map[asjc_col] = "ASJC Code"
    if quart_col:
        rename_map[quart_col] = "Quartile"
    if link_col:
        rename_map[link_col] = "URL"

    outlets = outlets.rename(columns=rename_map)

    # Tambahkan kolom ASJC Area (teks)
    outlets = add_asjc_area_column(outlets)

    # ------ Dangerous flags (admin) ------
    if os.path.exists("journal_flags.xlsx"):
        flags = pd.read_excel("journal_flags.xlsx")
        flags.columns = [c.strip() for c in flags.columns]
        if {"Title", "Dangerous"}.issubset(flags.columns):
            outlets = outlets.merge(
                flags[["Title", "Dangerous"]], on="Title", how="left"
            )

    if "Dangerous" not in outlets.columns:
        outlets["Dangerous"] = False

    # ------ SDG keywords ------
    sdg_col = find_col(sdg_keywords, ["sdg", "goal"])
    if not sdg_col:
        sdg_col = sdg_keywords.columns[0]

    keyword_cols = [c for c in sdg_keywords.columns if "keyword" in c.lower()]
    if not keyword_cols:
        keyword_cols = [c for c in sdg_keywords.columns if c != sdg_col]

    sdg_dict = {}
    for _, row in sdg_keywords.iterrows():
        sdg_name = str(row[sdg_col]).strip()
        kw_list = []
        for col in keyword_cols:
            text = str(row[col]).lower()
            parts = re.split(r"[;,/|]", text)
            kw_list.extend([p.strip() for p in parts if p.strip()])
        sdg_dict[sdg_name] = sorted(list(set(kw_list)))

    return outlets, sdg_dict


@st.cache_data(show_spinner=False)
def load_data_cached():
    # dibungkus cache supaya setelah pertama kali, load berikutnya jauh lebih cepat
    return load_data()


def save_flags(_outlet_df):
    flags = _outlet_df[["Title", "Dangerous"]]
    flags.to_excel("journal_flags.xlsx", index=False)


# ==========================
# SDG detection & mapping
# ==========================
def detect_sdgs_by_keywords(text):
    text_l = text.lower()
    result = {}
    for sdg_name, kw_list in sdg_dict.items():
        hits = sorted({kw for kw in kw_list if kw and kw in text_l})
        if hits:
            result[sdg_name] = hits
    return result


def ai_sdgs_numbers(abstract):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in UN Sustainable Development Goals (SDGs). "
                        "Given a research abstract, return ONLY the SDG numbers (1-17) "
                        "that are most relevant, separated by commas. Example: 3, 4, 9"
                    ),
                },
                {"role": "user", "content": abstract},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"⚠️ OpenAI Error (SDG detection): {e}")
        st.stop()


def map_sdg_numbers_to_names(sdg_numbers_str):
    nums = re.findall(r"\d+", sdg_numbers_str)
    nums = sorted(list(set(nums)))
    selected = []

    for n in nums:
        found = [
            name for name in sdg_dict.keys()
            if re.search(rf"\b{n}\b", str(name))
        ]
        if found:
            selected.extend(found)
        else:
            selected.append(f"SDG {n}")

    uniq = []
    seen = set()
    for x in selected:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def suggest_keywords_for_sdgs(sdg_names):
    """Disimpan kalau nanti mau dipakai di tempat lain."""
    keywords = []
    for name in sdg_names:
        if name in sdg_dict:
            keywords.extend(sdg_dict[name])
    return sorted(list(set(keywords)))


def select_additional_sdg_keywords(abstract, sdg_names, max_sdgs=3, max_keywords=30):
    """
    Pilih SDG utama (paling cocok) dan keyword tambahan:
    - SDG utama = SDG dengan paling banyak keyword yang SUDAH muncul di abstrak
    - Keyword tambahan = keyword dari SDG utama yang BELUM muncul di abstrak
    """
    text = abstract.lower()
    scored = []
    for name in sdg_names:
        kws = sdg_dict.get(name, [])
        hit_count = sum(1 for k in kws if k in text)
        scored.append((name, hit_count, kws))

    if not scored:
        return [], []

    # urutkan dari paling banyak hit
    scored.sort(key=lambda x: x[1], reverse=True)

    # ambil SDG utama max max_sdgs
    top_sdgs = [s for s, h, _ in scored[:max_sdgs]]

    # kumpulkan keyword tambahan dari SDG utama (yang belum ada di abstrak)
    additional = []
    for s, h, kws in scored:
        if s not in top_sdgs:
            continue
        for k in kws:
            if not k:
                continue
            if k in text:
                continue  # sudah ada di abstrak
            if k not in additional:
                additional.append(k)

    return top_sdgs, additional[:max_keywords]


def rewrite_abstract_with_keywords(original_abstract, keywords):
    keywords_sample = keywords[:30]

    prompt = f"""
Berikut adalah abstrak penelitian. Tolong tulis ulang abstrak ini dengan:

1. Tetap mempertahankan makna dan kontribusi ilmiahnya.
2. Menjaga bahasa yang digunakan sama dengan teks asli (Indonesia tetap Indonesia, Inggris tetap Inggris).
3. Mengintegrasikan beberapa keyword berikut secara natural (tidak wajib semua, gunakan yang paling relevan): 
   {", ".join(keywords_sample)}

Abstrak asli:
\"\"\"{original_abstract}\"\"\" 

Tulis ulang abstrak yang sudah diperbaiki:
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"⚠️ OpenAI Error (rewrite abstract): {e}")
        st.stop()


# ==========================
# TF-IDF prefilter & GPT ranking
# ==========================
def get_candidate_outlets(abstract, base_df, max_candidates=20):
    """Filter by Dangerous + ASJC, lalu TF-IDF (kalau ada)."""
    if base_df.empty:
        return base_df

    if not SKLEARN_AVAILABLE:
        st.warning(
            "⚠ scikit-learn tidak terpasang. "
            "Rekomendasi jurnal hanya menggunakan subset awal data."
        )
        return base_df.head(max_candidates).copy()

    scopes = base_df["Scope"].fillna("").astype(str).tolist()
    docs = [abstract] + scopes

    try:
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf = vectorizer.fit_transform(docs)
        sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        idx_sorted = sim.argsort()[::-1][:max_candidates]
        return base_df.iloc[idx_sorted].copy()
    except Exception as e:
        st.warning(f"⚠ Gagal TF-IDF: {e}. Pakai head() sebagai fallback.")
        return base_df.head(max_candidates).copy()


def ai_rank_outlets(abstract, base_df, max_candidates_for_gpt=20):
    """1) pre-filter base_df → kandidat; 2) ranking pakai GPT."""
    candidates = get_candidate_outlets(abstract, base_df, max_candidates_for_gpt)
    if candidates.empty:
        return candidates

    lines = []
    for _, row in candidates.iterrows():
        title = str(row.get("Title", "")).strip()
        scope = str(row.get("Scope", "")).strip()
        scope_short = scope[:400]
        if title:
            lines.append(f"{title} | {scope_short}")

    scopes_str = "\n".join(lines)

    prompt = f"""
You are a journal recommender that selects the most suitable Scopus-indexed journals 
based on their scope and the provided research abstract.

From the following list of journals (Title | Scope), score each journal from 0 to 100
for how suitable it is for the abstract, and return STRICT VALID JSON in the format:

[
  {{
    "title": "Exact journal title as in the list",
    "score": 0-100,
    "reason": "short explanation of why this journal fits"
  }}
]

ABSTRACT:
{abstract}

JOURNALS (Title | Scope):
{scopes_str}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
        except Exception:
            data = eval(raw)

        df_scores = pd.DataFrame(data)

        if "Title" not in df_scores.columns and "title" in df_scores.columns:
            df_scores = df_scores.rename(columns={"title": "Title"})

        merged = df_scores.merge(candidates, on="Title", how="left")
        if "score" in merged.columns:
            merged = merged.sort_values("score", ascending=False)
        return merged

    except Exception as e:
        st.error(f"⚠️ OpenAI Error (journal ranking): {e}")
        return pd.DataFrame()


def to_excel_bytes(df):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer


# ==========================
# Quartile badge helper
# ==========================
def quartile_badge(q):
    if q is None:
        return ""
    try:
        import math

        if isinstance(q, float) and math.isnan(q):
            return ""
    except Exception:
        pass

    q_str = str(q).upper().strip()
    if q_str not in {"Q1", "Q2", "Q3", "Q4"}:
        return ""
    cls = q_str.lower()
    return f'<span class="q-badge {cls}">{q_str}</span>'


# ==========================
# Submission Guidelines Checker
# ==========================
def clean_guideline_text(text: str) -> str:
    """
    Buang bagian akhir kalau model menambahkan kalimat seperti
    'Jika Anda mau, saya bisa ...' dsb.
    """
    if not text:
        return text

    lower = text.lower()
    stop_phrases = [
        "jika anda mau, saya bisa",
        "jika kamu mau, saya bisa",
        "if you want, i can",
    ]

    cut_index = None
    for phrase in stop_phrases:
        idx = lower.find(phrase)
        if idx != -1:
            if cut_index is None or idx < cut_index:
                cut_index = idx

    if cut_index is not None:
        return text[:cut_index].rstrip()

    return text


def generate_submission_guidelines(abstract, journal_row):
    title = str(journal_row.get("Title", ""))
    publisher = str(journal_row.get("Publisher", ""))
    scope = str(journal_row.get("Scope", ""))
    quartile = str(journal_row.get("Quartile", ""))
    coverage = str(journal_row.get("Coverage", "")) if "Coverage" in journal_row else ""

    prompt = f"""
Kamu adalah editor jurnal berpengalaman.

Berdasarkan informasi jurnal dan abstrak penelitian berikut, buatlah **dalam Bahasa Indonesia**:

1. Checklist hal-hal yang harus dicek penulis di laman "Instructions for Authors" sebelum submit
   (misalnya: jenis artikel yang diterima, batas jumlah kata/halaman, persyaratan metode, etika, gaya sitasi, APC/article processing charge, template, dsb).
2. Analisis singkat FIT / MISMATCH antara abstrak dan scope jurnal.
3. Beberapa saran singkat untuk meningkatkan kesiapan naskah sebelum dikirim ke jurnal ini.

Jawab dengan bullet points yang rapi, gunakan bahasa yang sopan namun to the point.
Jangan menawarkan bantuan lanjutan seperti "Jika Anda mau, saya bisa ...", cukup berhenti setelah memberikan checklist & saran.

INFO JURNAL:
- Title: {title}
- Publisher: {publisher}
- Quartile: {quartile}
- Coverage: {coverage}
- Scope: {scope}

ABSTRAK PENELITIAN:
{abstract}
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()
        cleaned = clean_guideline_text(raw)
        return cleaned
    except Exception as e:
        st.error(f"⚠️ OpenAI Error (submission guidelines): {e}")
        return ""


# ==========================
# LOGIN PAGE (pakai form)
# ==========================
def show_login_page():
    st.title("🔐 Scopus Outlet & SDG Recommender")

    st.markdown(
        """
Silakan login:

- Masukkan `password` untuk masuk sebagai **FM / user umum**  
- Masukkan `admin123` untuk masuk ke **halaman Admin**  
"""
    )

    with st.form("login_form"):
        pwd = st.text_input("Password:", type="password")
        submit = st.form_submit_button("Masuk")

    if submit:
        if pwd == "password":
            st.session_state["role"] = "fm"
        elif pwd == "admin123":
            st.session_state["role"] = "admin"
        else:
            st.error("Password salah.")


# ==========================
# ADMIN PAGE
# ==========================
def show_admin_page():
    global outlet_df

    with st.sidebar:
        st.write("Role: **ADMIN**")
        if st.button("Logout", key="logout_admin"):
            st.session_state["role"] = None

    st.title("🛠️ Admin – Manage Journal Knowledge")

    st.markdown(
        """
Di halaman ini admin dapat:

- Mencari jurnal  
- Melihat detail jurnal  
- Mem-*flag* jurnal sebagai **berbahaya** (tidak direkomendasikan ke FM)  
- Mengatur **kolom mana saja** yang ditampilkan ke FM  
"""
    )

    st.info(
        "Flag disimpan ke `journal_flags.xlsx`. Pengaturan kolom disimpan ke "
        "`journal_visible_columns.json`. Jurnal yang **Dangerous = True** "
        "tidak akan muncul di rekomendasi FM."
    )

    # --------- Pengaturan kolom untuk FM ----------
    st.subheader("⚙️ Pengaturan Kolom yang Ditampilkan ke FM")
    all_cols_for_config = [c for c in outlet_df.columns if c not in ["Dangerous"]]
    current_visible = get_visible_columns(all_cols_for_config)

    cols_selected = st.multiselect(
        "Pilih kolom yang boleh dilihat FM (detail & tabel ringkasan):",
        options=all_cols_for_config,
        default=current_visible,
    )

    if st.button("💾 Simpan Pengaturan Kolom"):
        save_visible_columns(cols_selected)
        st.success("Pengaturan kolom berhasil disimpan.")

    st.divider()

    # --------- Manajemen jurnal berbahaya ----------
    st.subheader("📚 Data Jurnal & Flag Berbahaya")

    search = st.text_input("Cari judul jurnal (optional):")
    df = outlet_df.copy()
    if search:
        df = df[df["Title"].str.contains(search, case=False, na=False)]

    df = df.sort_values("Title")
    titles = df["Title"].tolist()
    if not titles:
        st.warning("Tidak ada jurnal yang cocok dengan kata kunci.")
        return

    selected_title = st.selectbox("Pilih jurnal:", options=titles)
    detail_row = df[df["Title"] == selected_title].head(1)
    detail_series = detail_row.iloc[0]

    st.markdown(f"### Detail: {selected_title}")

    # Detail table (inline CSS)
    table_html = '<table style="width:100%; border-collapse:collapse;">'

    for field, value in detail_series.items():
        if pd.isna(value):
            val_str = ""
        else:
            val_str = str(value)
        # clickable URL
        if field.lower() in ["url", "link"] or val_str.startswith("http"):
            val_html = (
                f'<a href="{val_str}" target="_blank">{val_str}</a>' if val_str else ""
            )
        else:
            val_html = val_str

        table_html += (
            "<tr>"
            f'<td style="width:320px; font-weight:bold; white-space:nowrap; '
            'padding:6px 10px; border-bottom:1px solid #ddd; vertical-align:top;">'
            f"{field}</td>"
            f'<td style="padding:6px 10px; border-bottom:1px solid #ddd; '
            f'vertical-align:top;">{val_html}</td>'
            "</tr>"
        )

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("#### Flag status")
    current_flag = bool(detail_series.get("Dangerous", False))
    st.write(f"Status saat ini: **{'DANGEROUS' if current_flag else 'Normal'}**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚫 Flag sebagai Berbahaya"):
            outlet_df.loc[outlet_df["Title"] == selected_title, "Dangerous"] = True
            save_flags(outlet_df)
            st.success("Jurnal telah di-flag berbahaya.")
    with col2:
        if st.button("✅ Hapus Flag Berbahaya"):
            outlet_df.loc[outlet_df["Title"] == selected_title, "Dangerous"] = False
            save_flags(outlet_df)
            st.success("Flag berbahaya dihapus.")

    # Tabel jurnal berbahaya
    st.markdown("### 📛 Daftar Jurnal yang Di-flag Berbahaya")
    dangerous_df = outlet_df[outlet_df["Dangerous"] == True].copy()
    if dangerous_df.empty:
        st.write("Belum ada jurnal yang di-flag.")
    else:
        cols = [
            c
            for c in ["Title", "Publisher", "ASJC Area", "ASJC Code", "Quartile"]
            if c in dangerous_df.columns
        ]
        st.dataframe(
            dangerous_df[cols],
            use_container_width=True,
        )


# ==========================
# FM / USER PAGE
# ==========================
def show_fm_page():
    global outlet_df

    with st.sidebar:
        st.write("Role: **FM / USER**")
        if st.button("Logout", key="logout_fm"):
            st.session_state["role"] = None
            st.session_state["fm_results"] = None
            st.session_state["fm_guidelines"] = ""
            st.session_state["fm_selected_title"] = None

    st.title("📚 Scopus Outlet & SDG Recommender (FM View)")

    st.warning(
        "⚠️ Aplikasi ini hanya memberikan **rekomendasi**. "
        "FM tetap wajib mengecek secara mandiri di **Scopus / Elsevier Sources** "
        "untuk memastikan jurnal tersebut **masih terindeks** dan kebijakan terkini "
        "sebelum submit."
    )

    st.markdown(
        """
Masukkan abstrak penelitian kamu. Sistem akan:

1. Mendeteksi / merekomendasikan **SDGs** yang relevan  
2. Menampilkan **keyword apa saja yang terdeteksi per SDG**  
3. Menentukan **SDG utama** dan merekomendasikan **keyword tambahan** yang relevan  
4. Memberikan versi **abstrak yang sudah diperbaiki** dengan keyword tersebut  
5. Merekomendasikan **jurnal Scopus** yang paling cocok (dengan badge Q1–Q4 & filter ASJC)  
"""
    )

    abstract = st.text_area("Abstrak penelitian:", height=220)

    col_left, col_right = st.columns([1, 1])
    with col_right:
        top_n = st.slider(
            "Tampilkan berapa banyak jurnal rekomendasi?",
            min_value=3,
            max_value=10,
            value=3,
            step=1,
        )

    # Filter ASJC pakai teks area
    base_df = outlet_df[outlet_df["Dangerous"] != True].copy()

    asjc_options = []
    if "ASJC Area" in base_df.columns:
        areas_set = set()
        for val in base_df["ASJC Area"].dropna():
            for part in re.split(r"[;,]", str(val)):
                p = part.strip()
                if p:
                    areas_set.add(p)
        asjc_options = sorted(areas_set)

    selected_asjc = []
    if asjc_options:
        selected_asjc = st.multiselect(
            "Filter berdasarkan ASJC Area (optional):",
            options=asjc_options,
        )
        if selected_asjc:
            base_df = base_df[
                base_df["ASJC Area"]
                .fillna("")
                .apply(lambda s: any(a in s for a in selected_asjc))
            ]

    run_clicked = st.button("🚀 Jalankan Rekomendasi")

    # ========= JALANKAN REKOMENDASI & SIMPAN KE SESSION =========
    if run_clicked:
        if not abstract.strip():
            st.warning("Masukkan abstrak terlebih dahulu.")
            st.stop()

        if base_df.empty:
            st.error(
                "Tidak ada jurnal yang tersisa setelah filter ASJC / Dangerous. "
                "Coba kurangi filter atau cek data."
            )
            st.stop()

        # 1. SDG detection (keyword-based)
        sdg_hits = detect_sdgs_by_keywords(abstract)

        if sdg_hits:
            # semua SDG yang punya keyword di abstrak
            selected_sdg_names = list(sdg_hits.keys())
        else:
            # pakai AI kalau tidak ada keyword yang match
            sdg_nums_str = ai_sdgs_numbers(abstract)
            selected_sdg_names = map_sdg_numbers_to_names(sdg_nums_str)

        # 2. Pilih SDG utama & keyword tambahan (hanya dari SDG utama, dan belum ada di abstrak)
        primary_sdgs, sdg_keywords_suggested = select_additional_sdg_keywords(
            abstract, selected_sdg_names, max_sdgs=3, max_keywords=30
        )

        # fallback kalau sdg_keywords_suggested kosong total
        if not primary_sdgs:
            primary_sdgs = selected_sdg_names[:1]
        if not sdg_keywords_suggested:
            # ambil beberapa keyword dari SDG pertama saja sebagai fallback
            fallback_kws = []
            for name in primary_sdgs:
                fallback_kws.extend(sdg_dict.get(name, []))
            sdg_keywords_suggested = sorted(list(set(fallback_kws)))[:30]

        # 3. Rewrite abstrak (spinner)
        with st.spinner(
            "AI sedang memformulasi rekomendasi abstrak baru yang terintegrasi dengan SDGs…"
        ):
            improved_abstract = rewrite_abstract_with_keywords(
                original_abstract=abstract, keywords=sdg_keywords_suggested
            )

        # 4. Rekomendasi jurnal
        with st.spinner("AI sedang menganalisis kecocokan jurnal…"):
            ranking_df = ai_rank_outlets(
                improved_abstract, base_df, max_candidates_for_gpt=20
            )

        if ranking_df is None or ranking_df.empty:
            st.error("Gagal mendapatkan rekomendasi jurnal. Cek error di atas.")
            st.stop()

        # Simpan semuanya ke session_state
        st.session_state["fm_results"] = {
            "abstract": abstract,
            "sdg_hits": sdg_hits,
            "selected_sdg_names": selected_sdg_names,
            "primary_sdgs": primary_sdgs,
            "sdg_keywords_suggested": sdg_keywords_suggested,
            "improved_abstract": improved_abstract,
            "ranking_df": ranking_df,
        }
        st.session_state["fm_guidelines"] = ""
        st.session_state["fm_selected_title"] = None

    # ========= TAMPILKAN HASIL DARI SESSION =========
    results = st.session_state.get("fm_results")
    if results is None:
        st.info("Jalankan rekomendasi terlebih dahulu untuk melihat hasil.")
        return

    sdg_hits = results["sdg_hits"]
    selected_sdg_names = results["selected_sdg_names"]
    primary_sdgs = results.get("primary_sdgs", selected_sdg_names)
    sdg_keywords_suggested = results["sdg_keywords_suggested"]
    improved_abstract = results["improved_abstract"]
    ranking_df = results["ranking_df"]

    # ===== SDG section =====
    st.subheader("🌍 Rekomendasi SDGs, Deteksi Keyword, & Keyword Tambahan")

    if sdg_hits:
        st.success("SDGs terdeteksi dari isi abstrak (berdasarkan keyword yang muncul):")
        for sdg_name, hits in sdg_hits.items():
            st.markdown(f"- **{sdg_name}**")
            st.caption("  Keyword yang terdeteksi di abstrak:")
            st.write(", ".join(hits[:20]) + (" ..." if len(hits) > 20 else ""))
    else:
        st.info("SDG tidak terdeteksi langsung dari keyword; menggunakan analisis AI.")
        st.write("SDGs yang direkomendasikan AI:")
        st.markdown("- " + "\n- ".join(selected_sdg_names))

    if sdg_keywords_suggested:
        if primary_sdgs:
            st.markdown(
                f"**Keyword SDGs yang direkomendasikan untuk dimasukkan (tambahan, "
                f"berdasarkan SDG utama: {', '.join(primary_sdgs)}):**"
            )
        else:
            st.markdown("**Keyword SDGs yang direkomendasikan untuk dimasukkan (tambahan):**")

        max_show = 40
        show_list = sdg_keywords_suggested[:max_show]
        st.write(", ".join(show_list))
        if len(sdg_keywords_suggested) > max_show:
            st.caption(
                f"... dan {len(sdg_keywords_suggested) - max_show} keyword lainnya."
            )
    else:
        st.warning(
            "Tidak menemukan keyword tambahan untuk SDG yang dipilih. "
            "Cek struktur file SDGs Keyword."
        )

    # ===== Abstrak rekomendasi =====
    st.subheader("✍️ Abstrak Rekomendasi (dengan keyword SDGs)")

    st.text_area(
        "Abstrak versi rekomendasi (bisa di-copy & edit lagi):",
        value=improved_abstract,
        height=260,
    )

    st.divider()

    # ===== Rekomendasi jurnal =====
    st.subheader("🎯 Rekomendasi Jurnal Scopus")

    st.caption(
        "Rekomendasi didasarkan pada kecocokan antara **scope jurnal** dan "
        "**abstrak rekomendasi di atas**. "
        "Untuk menghemat token, sistem memfilter kandidat dengan TF-IDF terlebih dahulu. "
        "Jurnal yang di-*flag* berbahaya oleh admin tidak disertakan."
    )

    ranking_top = ranking_df.head(top_n)

    st.markdown("**Daftar jurnal yang direkomendasikan (urut dari yang paling cocok):**")
    for _, row in ranking_top.iterrows():
        title = row.get("Title", "(no title)")
        score = row.get("score", None)
        reason = row.get("reason", "")
        quart = row.get("Quartile", None)
        badge = quartile_badge(quart)

        if score is not None:
            try:
                score_str = f"{float(score):.1f}"
            except Exception:
                score_str = str(score)
            line = f"{badge}<b>{title}</b> — skor kecocokan: <b>{score_str}</b>"
        else:
            line = f"{badge}<b>{title}</b>"

        st.markdown(line, unsafe_allow_html=True)
        if reason:
            st.caption(f"└ alasan: {reason}")

    # Tabel ringkas: pakai konfigurasi kolom admin
    visible_cols = get_visible_columns(outlet_df.columns)
    base_cols = ["Title", "score", "reason"]
    cols_show = base_cols + [
        c for c in visible_cols if c in ranking_top.columns and c not in base_cols
    ]
    cols_show = [c for c in cols_show if c in ranking_top.columns]

    st.write("")
    st.markdown("**Tabel ringkasan rekomendasi:**")
    st.dataframe(ranking_top[cols_show], use_container_width=True)

    # ===== Detail outlet + submission guidelines =====
    st.subheader("🔍 Detail Outlet yang Dipilih")

    outlet_titles = ranking_top["Title"].dropna().tolist()
    if outlet_titles:
        st.session_state["fm_selected_title"] = st.selectbox(
            "Pilih salah satu outlet untuk melihat detailnya:",
            options=outlet_titles,
            key="fm_select_outlet",
        )
        selected_title = st.session_state["fm_selected_title"]

        if selected_title:
            detail_row = outlet_df[outlet_df["Title"] == selected_title].head(1)
            if not detail_row.empty:
                st.markdown(f"**Detail untuk:** {selected_title}")
                detail_series = detail_row.iloc[0]

                visible_cols = get_visible_columns(outlet_df.columns)
                if "Title" not in visible_cols:
                    visible_cols = ["Title"] + visible_cols

                table_html = '<table style="width:100%; border-collapse:collapse;">'

                for field in visible_cols:
                    if field not in detail_series.index:
                        continue
                    value = detail_series[field]
                    if pd.isna(value):
                        val_str = ""
                    else:
                        val_str = str(value)

                    if field.lower() in ["url", "link"] or val_str.startswith("http"):
                        val_html = (
                            f'<a href="{val_str}" target="_blank">{val_str}</a>'
                            if val_str
                            else ""
                        )
                    else:
                        val_html = val_str

                    table_html += (
                        "<tr>"
                        f'<td style="width:320px; font-weight:bold; white-space:nowrap; '
                        'padding:6px 10px; border-bottom:1px solid #ddd; vertical-align:top;">'
                        f"{field}</td>"
                        f'<td style="padding:6px 10px; border-bottom:1px solid #ddd; '
                        f'vertical-align:top;">{val_html}</td>'
                        "</tr>"
                    )

                table_html += "</table>"

                st.markdown(table_html, unsafe_allow_html=True)

                # Submission Guidelines Checker + spinner
                if st.button("📝 Generate Submission Guidelines Checklist"):
                    with st.spinner("AI sedang membuat catatan submission…"):
                        tips = generate_submission_guidelines(
                            improved_abstract, detail_series
                        )
                    st.session_state["fm_guidelines"] = tips
            else:
                st.warning("Tidak menemukan detail outlet di data asal.")
    else:
        st.warning("Tidak ada judul outlet yang bisa dipilih.")

    if st.session_state["fm_guidelines"]:
        st.markdown("#### Checklist & Catatan Submission")
        st.markdown(st.session_state["fm_guidelines"])

    # ===== Download =====
    st.subheader("📥 Download Data Rekomendasi Lengkap")
    excel_bytes = to_excel_bytes(ranking_df)
    st.download_button(
        "Download semua hasil rekomendasi (Excel)",
        data=excel_bytes,
        file_name="Scopus_Outlet_Recommendations.xlsx",
        mime=(
            "application/"
            "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
    )


# ==========================
# MAIN ROUTING
# ==========================
def main():
    global outlet_df, sdg_dict

    role = st.session_state["role"]
    if role is None:
        # Belum login → tidak usah load data dulu
        show_login_page()
        return

    # Sudah login: baru load data, dengan spinner info yang jelas
    with st.spinner("Mohon tunggu, sistem sedang memuat data jurnal & SDGs…"):
        try:
            outlet_df, sdg_dict = load_data_cached()
        except Exception as e:
            st.error(f"❌ Error saat load data: {e}")
            return

    if role == "admin":
        show_admin_page()
    else:
        show_fm_page()


if __name__ == "__main__":
    main()
