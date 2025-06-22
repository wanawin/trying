import streamlit as st
import os, unicodedata, re, importlib
from itertools import product, combinations
import pandas as pd

# ==============================
# Ensure wide layout and sidebar appears
# ==============================
try:
    st.set_page_config(layout="wide")
except Exception:
    pass
st.sidebar.title("🔧 DC-5 Midday Settings & Debug")
st.sidebar.write("✅ Sidebar initialization reached.")
st.sidebar.write("🔍 Debug: Sidebar is active and UI components should appear.")

# ==============================
# Helper functions for parsing manual filters
# ==============================
def strip_prefix(raw_name: str) -> str:
    return re.sub(r'^\s*\d+[\.)]\s*', '', raw_name).strip()

def normalize_name(raw_name: str) -> str:
    s = unicodedata.normalize('NFKC', raw_name)
    s = s.replace('≥', '>=').replace('≤', '<=').replace('\u2265', '>=').replace('\u2264', '<=')
    s = s.replace('→', '->').replace('\u2192', '->')
    s = s.replace('–', '-').replace('—', '-')
    s = s.replace('\u200B', '').replace('\u00A0', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()

def parse_manual_filters_txt(raw_text: str):
    entries = []
    skipped = []
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    current = None
    raw_block = []
    for raw_ln in lines:
        ln = raw_ln.strip()
        if not ln:
            if current:
                if current.get('name'):
                    entries.append(current)
                else:
                    skipped.append({'block': '\n'.join(raw_block)})
                current = None
                raw_block = []
            continue
        raw_block.append(raw_ln)
        norm_ln = unicodedata.normalize('NFKC', ln)
        low = norm_ln.lower()
        if low.startswith('type:'):
            if current is None:
                current = {'name': '', 'name_raw': '', 'type': '', 'logic': '', 'action': ''}
            current['type'] = norm_ln.split(':', 1)[1].strip()
        elif low.startswith('logic:'):
            if current is None:
                current = {'name': '', 'name_raw': '', 'type': '', 'logic': '', 'action': ''}
            after = norm_ln.split(':', 1)[1].strip()
            if after.lower().startswith('logic:'):
                after = after.split(':', 1)[1].strip()
            current['logic'] = after
        elif low.startswith('action:'):
            if current is None:
                current = {'name': '', 'name_raw': '', 'type': '', 'logic': '', 'action': ''}
            current['action'] = norm_ln.split(':', 1)[1].strip()
        else:
            if current:
                if current.get('name'):
                    entries.append(current)
                else:
                    skipped.append({'block': '\n'.join(raw_block[:-1])})
            clean = strip_prefix(norm_ln)
            name_norm = normalize_name(clean)
            current = {'name': name_norm, 'name_raw': clean, 'type': '', 'logic': '', 'action': ''}
            raw_block = [raw_ln]
    if current:
        if current.get('name'):
            entries.append(current)
        else:
            skipped.append({'block': '\n'.join(raw_block)})
    return entries, skipped

# ==============================
# Filter-application helpers
# ==============================
def seed_sum_matches_condition(seed_sum: int, condition_str: str) -> bool:
    s = condition_str.strip()
    m = re.match(r'[≤<=]\s*(\d+)', s)
    if m:
        try:
            return seed_sum <= int(m.group(1))
        except:
            return False
    m = re.match(r'(?:≥|>=)\s*(\d+)\s*or\s*higher', s, re.IGNORECASE)
    if m:
        try:
            return seed_sum >= int(m.group(1))
        except:
            return False
    m = re.match(r'(\d+)\s*[–\-]\s*(\d+)', s)
    if m:
        try:
            low, high = int(m.group(1)), int(m.group(2))
            return low <= seed_sum <= high
        except:
            return False
    if s.isdigit():
        try:
            return seed_sum == int(s)
        except:
            return False
    return False

def apply_sum_range_filter(combos, min_sum, max_sum):
    try:
        keep = [c for c in combos if min_sum <= sum(int(d) for d in c) <= max_sum]
    except:
        keep = combos[:]
    removed = [c for c in combos if c not in keep]
    return keep, removed

def apply_sum_eq_filter(combos, target_sum):
    try:
        keep = [c for c in combos if sum(int(d) for d in c) != target_sum]
        removed = [c for c in combos if sum(int(d) for d in c) == target_sum]
        return keep, removed
    except:
        return combos, []

def apply_keep_sum_range_if_seed_sum(combos, seed_sum, min_sum, max_sum, seed_condition_str):
    try:
        if seed_sum_matches_condition(seed_sum, seed_condition_str):
            return apply_sum_range_filter(combos, min_sum, max_sum)
        else:
            return combos, []
    except:
        return combos, []

def apply_conditional_seed_contains(combos, seed_digits, seed_digit, required_winners):
    try:
        if seed_digit in seed_digits:
            keep, removed = [], []
            for c in combos:
                found = any(str(rw) in c for rw in required_winners)
                if found:
                    keep.append(c)
                else:
                    removed.append(c)
            return keep, removed
        return combos, []
    except:
        return combos, []

# ==============================
# Generate combinations
# ==============================
def generate_combinations(seed, method="2-digit pair"):
    all_digits = '0123456789'
    combos = set()
    seed_str = str(seed)
    if not seed_str.isdigit() or len(seed_str) < 2:
        return []
    if method == "1-digit":
        for d in seed_str:
            for p in product(all_digits, repeat=4):
                combo = ''.join(sorted(d + ''.join(p)))
                combos.add(combo)
    else:
        pairs = set(''.join(sorted((seed_str[i], seed_str[j]))) for i in range(len(seed_str)) for j in range(i+1, len(seed_str)))
        for pair in pairs:
            for p in product(all_digits, repeat=3):
                combo = ''.join(sorted(pair + ''.join(p)))
                combos.add(combo)
    return sorted(combos)

# ==============================
# Streamlit App
# ==============================
st.title("DC-5 Midday Blind Predictor with External Aggressiveness Sorting and Custom Order")

# Sidebar inputs
seed = st.sidebar.text_input("5-digit seed:")
hot_digits = [d for d in st.sidebar.text_input("Hot digits (comma-separated):").replace(' ', '').split(',') if d]
cold_digits = [d for d in st.sidebar.text_input("Cold digits (comma-separated):").replace(' ', '').split(',') if d]
due_digits = [d for d in st.sidebar.text_input("Due digits (comma-separated):").replace(' ', '').split(',') if d]
method = st.sidebar.selectbox("Generation Method:", ["1-digit", "2-digit pair"])
enable_trap = st.sidebar.checkbox("Enable Trap V3 Ranking")

# Debug point: confirm inputs appear
st.sidebar.write(f"🔧 Debug: Seed input active, current seed='{seed}'")

# Trap V3 uploader
trap_uploaded = st.sidebar.file_uploader("Upload Trap V3 model file (dc5_trapv3_model.py)", type=['py'])
if trap_uploaded is not None:
    try:
        content = trap_uploaded.read()
        with open('dc5_trapv3_model.py', 'wb') as f:
            f.write(content)
        st.sidebar.success("Saved Trap V3 model file.")
        if 'dc5_trapv3_model' in globals():
            importlib.reload(globals()['dc5_trapv3_model'])
    except Exception as e:
        st.sidebar.error(f"Failed saving Trap V3 model file: {e}")

# Load manual filters - only via user upload or single candidate to avoid merging issues
uploaded = st.sidebar.file_uploader("Upload manual filters file (TXT)", type=['txt'])
raw_manual = None
if uploaded is not None:
    try:
        raw_manual = uploaded.read().decode('utf-8', errors='ignore')
        st.sidebar.success("Loaded uploaded manual filters file.")
    except Exception as e:
        st.sidebar.error(f"Failed reading uploaded file: {e}")
else:
    candidates = ["manual_filters_full.txt"]
    found = [fname for fname in candidates if os.path.exists(fname)]
    if len(found) == 1:
        try:
            raw_manual = open(found[0], 'r', encoding='utf-8').read()
            st.sidebar.info(f"Loaded manual filters from {found[0]}")
        except Exception as e:
            st.sidebar.error(f"Failed reading {found[0]}: {e}")
    else:
        st.sidebar.warning("No manual filter file found or multiple candidates present. Upload a TXT manually.")

parsed_entries = []
skipped_blocks = []
if raw_manual:
    try:
        parsed_entries, skipped_blocks = parse_manual_filters_txt(raw_manual)
    except Exception as e:
        parsed_entries = []
        skipped_blocks = []
        st.sidebar.error(f"Error parsing manual filters: {e}")
    st.write(f"Parsed {len(parsed_entries)} manual filter blocks")
    if skipped_blocks:
        st.warning(f"{len(skipped_blocks)} blocks skipped due to missing name.")
        with st.expander("Skipped blocks"):
            for sb in skipped_blocks:
                blk = sb.get('block','')
                st.code(blk[:300] + ("..." if len(blk)>300 else ""))
    if st.sidebar.checkbox("Show normalized filter names for debugging"):
        st.write("#### Parsed Filter Names:")
        for idx, pf in enumerate(parsed_entries):
            st.write(f"{idx}: RAW='{pf.get('name_raw','')}' -> NORM='{pf.get('name','')}' | Type: '{pf.get('type','')}' | Logic: '{pf.get('logic','')}'")
else:
    st.write("No manual filters loaded.")

# Aggressiveness CSV
agg_uploaded = st.sidebar.file_uploader("Upload aggressiveness CSV (columns 'filter' and 'score')", type=['csv'])
def load_aggressiveness_map(csv_path=None, uploaded_file=None):
    mapping = {}
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'filter' in df.columns and 'score' in df.columns:
                for _, row in df.iterrows():
                    norm = normalize_name(str(row['filter']))
                    try:
                        score = int(row['score'])
                    except:
                        continue
                    mapping[norm] = score
                st.sidebar.success("Loaded aggressiveness mapping from uploaded CSV.")
            else:
                st.sidebar.error("Uploaded CSV must have columns 'filter' and 'score'.")
        except Exception as e:
            st.sidebar.error(f"Failed loading uploaded aggressiveness CSV: {e}")
    elif csv_path and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'filter' in df.columns and 'score' in df.columns:
                for _, row in df.iterrows():
                    norm = normalize_name(str(row['filter']))
                    try:
                        score = int(row['score'])
                    except:
                        continue
                    mapping[norm] = score
                st.sidebar.success(f"Loaded aggressiveness mapping from {csv_path}.")
            else:
                st.sidebar.error(f"CSV {csv_path} must have columns 'filter' and 'score'.")
        except Exception as e:
            st.sidebar.error(f"Failed loading aggressiveness CSV: {e}")
    return mapping

if parsed_entries:
    agg_map = load_aggressiveness_map(csv_path="filter_ranking.csv", uploaded_file=agg_uploaded)
    if agg_map:
        try:
            parsed_entries.sort(key=lambda pf: agg_map.get(normalize_name(pf['name']), float('inf')))
            st.sidebar.info("Manual filters sorted by external aggressiveness (least→most). Missing appear last.")
        except Exception as e:
            st.sidebar.error(f"Error sorting filters: {e}")

# Apply filters - only when explicitly selected
session_pool = []
if seed:
    combos_initial = generate_combinations(seed, method)
    session_pool = combos_initial.copy()
    st.write(f"Generated {len(combos_initial)} combos before manual filters.")
    if parsed_entries:
        st.sidebar.markdown("### Manual Filter Selection")
    any_filtered = False
    special_combo = st.sidebar.text_input("Special combo check (enter 5-digit combo):")
    special_result = None
    norm_special = ''.join(sorted(str(special_combo).strip())) if special_combo else ''
    prev_special_set = {''.join(sorted(c)) for c in session_pool}
    for idx, pf in enumerate(parsed_entries):
        raw_name = pf.get('name_raw','') or ''
        logic = pf.get('logic','') or ''
        action = pf.get('action','') or ''
        # Only consider entries that look like filters
        if not re.search(r'\b(sum|seed|eliminate|filter|contains|range)\b', raw_name.lower()) and not logic:
            continue
        label_clean = re.sub(r"['\"\r\n]", " ", raw_name)
        label = label_clean[:50].strip() or f"Filter_{idx}"
        help_text = f"Type: {pf.get('type','')}\nLogic: {logic}\nAction: {action}"
        try:
            checked = st.sidebar.checkbox(label, key=f"filter_{idx}", help=help_text)
        except Exception:
            st.sidebar.error(f"Skipping filter {idx} due to label error.")
            continue
        if checked:
            any_filtered = True
            keep, removed = session_pool, []
            norm_name = normalize_name(raw_name)
            norm_logic = normalize_name(logic)
            # 1) Sum = N
            m_eq = re.search(r'sum\s*=\s*(\d+)', norm_name)
            if m_eq:
                try:
                    target = int(m_eq.group(1))
                    keep, removed = apply_sum_eq_filter(session_pool, target)
                except:
                    keep, removed = session_pool, []
            else:
                # 2) Sum < X or > Y patterns from raw_name or logic
                m_range1 = re.search(r'sum\s*<\s*(\d+)\s*or\s*>\s*(\d+)', raw_name.lower() + ' ' + logic.lower())
                m_range2 = re.search(r'sum\s*<\s*(\d+).*?>\s*(\d+)', raw_name.lower() + ' ' + logic.lower())
                if m_range1 or m_range2:
                    m = m_range1 or m_range2
                    try:
                        low = int(m.group(1)); high = int(m.group(2))
                        keep, removed = apply_sum_range_filter(session_pool, low, high)
                    except:
                        keep, removed = session_pool, []
                else:
                    # 3) Sum between X and Y, optionally conditional on seed sum
                    m_between = re.search(r'between\s*(\d+)\s*and\s*(\d+)', logic, re.IGNORECASE)
                    if m_between:
                        try:
                            low, high = int(m_between.group(1)), int(m_between.group(2))
                            seed_sum = sum(int(d) for d in seed if d.isdigit())
                            m_cond = re.search(r'if the seed sum is\s*([^\.]+)', logic, re.IGNORECASE)
                            if m_cond:
                                cond_str = m_cond.group(1).strip()
                                keep, removed = apply_keep_sum_range_if_seed_sum(session_pool, seed_sum, low, high, cond_str)
                            else:
                                keep, removed = apply_sum_range_filter(session_pool, low, high)
                        except:
                            keep, removed = session_pool, []
                    else:
                        # 4) Conditional seed contains patterns
                        m_cond2 = re.search(r'seed contains\s*(\d).*contain\s*([0-9, ]+)', raw_name, re.IGNORECASE)
                        if m_cond2:
                            try:
                                sd = int(m_cond2.group(1))
                                reqs_str = m_cond2.group(2)
                                reqs = [int(x) for x in re.findall(r'(\d)', reqs_str)]
                                seed_digits = [int(d) for d in seed if d.isdigit()]
                                keep, removed = apply_conditional_seed_contains(session_pool, seed_digits, sd, reqs)
                            except:
                                keep, removed = session_pool, []
                        else:
                            st.sidebar.warning(f"Could not apply logic for: '{label}'")
                            keep, removed = session_pool, []
            # Special combo check
            try:
                if norm_special and special_result is None:
                    new_set = {''.join(sorted(c)) for c in keep}
                    if norm_special in prev_special_set and norm_special not in new_set:
                        special_result = label
                    prev_special_set = new_set
            except:
                pass
            session_pool = keep
            st.write(f"Filter '{label}' removed {len(removed)} combos.")
    # special combo feedback
    if special_combo:
        try:
            if norm_special not in {''.join(sorted(c)) for c in combos_initial}:
                st.sidebar.info(f"Special combo '{special_combo}' was NOT generated from seed.")
            else:
                if special_result:
                    st.sidebar.info(f"Special combo '{special_combo}' was eliminated by filter: {special_result}")
                else:
                    if any_filtered:
                        st.sidebar.info(f"Special combo '{special_combo}' survived all selected manual filters.")
                    else:
                        st.sidebar.info(f"Special combo '{special_combo}' would not be eliminated by any selected manual filters.")
        except:
            pass
    st.write(f"**Remaining combos after manual filters:** {len(session_pool)}")
    if not any_filtered:
        st.info("No manual filters selected; Trap V3 will rank the unfiltered combos unless prevented.")
    with st.expander("Show remaining combinations"):
        for c in session_pool:
            st.write(c)

# Trap V3 Ranking
if enable_trap and seed and session_pool:
    try:
        st.write(f"Passing {len(session_pool)} combos into Trap V3.")
        if 'dc5_trapv3_model' in globals():
            trap_model = importlib.reload(globals()['dc5_trapv3_model'])
        else:
            import dc5_trapv3_model as trap_model
        ranked = trap_model.rank_combinations(session_pool, str(seed))
        st.write("## Trap V3 Ranking")
        st.write("Top combos:")
        for c in ranked[:20]: st.write(c)
        if len(ranked) > 20:
            with st.expander("Show all ranked combos"):
                for c in ranked: st.write(c)
    except ModuleNotFoundError:
        st.error("Trap V3 model not found: please upload dc5_trapv3_model.py via the sidebar or place it in the app directory.")
    except AttributeError:
        st.error("Trap V3 model loaded but missing 'rank_combinations' function.")
    except Exception as e:
        st.error(f"Trap V3 ranking failed: {e}")
