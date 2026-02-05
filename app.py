import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
import re
import json
import io
import string

# --- KONFIGURACIJA STRANICE ---
st.set_page_config(page_title="LabFit Mobile", layout="wide", page_icon="📈")

# --- INICIJALIZACIJA STANJA (SESSION STATE) ---
if 'df' not in st.session_state:
    # Početna tabela 15 redova, 4 kolone
    st.session_state.df = pd.DataFrame(index=range(15), columns=['A', 'B', 'C', 'D'])
    st.session_state.df[:] = ""

if 'col_meta' not in st.session_state:
    # Metadata (imena veličina i jedinice)
    st.session_state.col_meta = {col: {'qty': f'{col}', 'unit': '-'} for col in ['A', 'B', 'C', 'D']}

if 'plot_title' not in st.session_state:
    st.session_state.plot_title = "Grafik zavisnosti"

# --- FUNKCIJE ---

def add_new_column():
    """Logika za dodavanje nove kolone (E, F, G...)"""
    curr_cols = list(st.session_state.df.columns)
    # Generisanje sledećeg slova
    new_char = string.ascii_uppercase[len(curr_cols) % 26]
    # Ako pređe Z, dodaje broj (npr. A1, B1...)
    if len(curr_cols) >= 26: 
        new_char += str(len(curr_cols)//26)
    
    # Dodavanje u DataFrame
    st.session_state.df[new_char] = ""
    # Dodavanje u meta podatke
    st.session_state.col_meta[new_char] = {'qty': 'Nova', 'unit': '-'}

def remove_last_column():
    """Logika za brisanje poslednje kolone"""
    cols = list(st.session_state.df.columns)
    if len(cols) > 1: # Mora ostati bar jedna
        last_col = cols[-1]
        st.session_state.df.drop(columns=[last_col], inplace=True)
        if last_col in st.session_state.col_meta:
            del st.session_state.col_meta[last_col]

def save_project():
    data = {
        "df": st.session_state.df.to_json(orient="split"),
        "meta": st.session_state.col_meta,
        "plot_title": st.session_state.plot_title
    }
    return json.dumps(data)

def load_project(uploaded_file):
    try:
        data = json.load(uploaded_file)
        json_str = data["df"]
        st.session_state.df = pd.read_json(io.StringIO(json_str), orient="split")
        st.session_state.col_meta = data["meta"]
        st.session_state.plot_title = data.get("plot_title", "")
        st.success("Projekt učitan!")
        st.rerun()
    except Exception as e:
        st.error(f"Greška pri učitavanju: {e}")

def run_formula(target_col, formula_str, limit_rows):
    try:
        calc_df = st.session_state.df.iloc[:limit_rows].copy()
        calc_df = calc_df.apply(pd.to_numeric, errors='coerce')

        formula_str = formula_str.replace('^', '**')
        parsed_formula = re.sub(r"col\(([A-Za-z0-9_]+)\)", r"calc_df['\1']", formula_str)

        local_dict = {"np": np, "numpy": np, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "log": np.log, "calc_df": calc_df}
        
        try:
            res = float(formula_str.replace(',', '.'))
        except:
            res = eval(parsed_formula, {"__builtins__": None}, local_dict)

        if isinstance(res, (pd.Series, np.ndarray)):
            res = np.round(res, 5)

        st.session_state.df[target_col] = st.session_state.df[target_col].astype(object)
        st.session_state.df.loc[:limit_rows-1, target_col] = res
        
        st.success(f"Izračunato u koloni {target_col}!")
        return True
    except Exception as e:
        st.error(f"Greška u formuli: {e}")
        return False

# --- GLAVNI INTERFEJS ---

st.title("📱 LabFit Studio")
st.markdown("Fizika: Obrada podataka i grafici")

tab1, tab2, tab3 = st.tabs(["📝 Podaci", "📈 Grafik", "💾 Fajlovi"])

# === TAB 1: PODACI ===
with tab1:
    # --- MENADŽMENT KOLONA ---
    col_btn1, col_btn2, col_info = st.columns([1, 1, 3])
    with col_btn1:
        if st.button("➕ Dodaj kolonu"):
            add_new_column()
            st.rerun()
    with col_btn2:
        if st.button("➖ Obriši zadnju"):
            remove_last_column()
            st.rerun()
            
    # --- METADATA EDITOR ---
    with st.expander("⚙️ Imena veličina i jedinice", expanded=False):
        # Priprema podataka za editor
        meta_list = []
        for c in st.session_state.df.columns:
            meta = st.session_state.col_meta.get(c, {'qty': c, 'unit': '-'})
            meta_list.append({"Kolona": c, "Veličina": meta['qty'], "Jedinica": meta['unit']})
        
        meta_df = pd.DataFrame(meta_list)
        
        # Prikaz editora
        edited_meta = st.data_editor(
            meta_df, 
            key="meta_editor", 
            hide_index=True,
            use_container_width=True
        )
        
        # Čuvanje izmena nazad u session_state
        for index, row in edited_meta.iterrows():
            c = row['Kolona']
            st.session_state.col_meta[c] = {'qty': row['Veličina'], 'unit': row['Jedinica']}

    # --- GLAVNA TABELA ---
    st.markdown("### Tabela merenja")
    
    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        key="main_data_editor",
        use_container_width=True,
        height=400
    )
    
    if not edited_df.equals(st.session_state.df):
        st.session_state.df = edited_df

    # --- KALKULATOR ---
    st.markdown("---")
    with st.expander("🧮 Kalkulator (Formula)"):
        c1, c2 = st.columns([1, 2])
        # Osiguravamo da je selectbox uvek sinhronizovan sa trenutnim kolonama
        cols_now = list(st.session_state.df.columns)
        target_col = c1.selectbox("Rezultat u:", cols_now)
        formula_inp = c2.text_input("Formula:", placeholder="npr. col(A)/20 ili col(A)^2")
        
        c3, c4 = st.columns(2)
        row_limit = c3.number_input("Prvih N redova:", value=15, min_value=1)
        
        if c4.button("Izračunaj"):
            if run_formula(target_col, formula_inp, row_limit):
                st.rerun()

# === TAB 2: GRAFIK ===
with tab2:
    st.markdown("### Podešavanje ose")
    
    col_opts = list(st.session_state.df.columns)
    c1, c2 = st.columns(2)
    # Provera indeksa da ne pukne ako se obrišu kolone
    idx_x = 0 if len(col_opts) > 0 else None
    idx_y = 1 if len(col_opts) > 1 else 0
    
    x_col = c1.selectbox("X osa:", col_opts, index=idx_x)
    y_col = c2.selectbox("Y osa:", col_opts, index=idx_y)
    
    c3, c4 = st.columns(2)
    dx_col = c3.selectbox("Greška dX:", ["(Nema)"] + col_opts)
    dy_col = c4.selectbox("Greška dY:", ["(Nema)"] + col_opts)
    
    plot_title_input = st.text_input("Naslov grafika:", value=st.session_state.plot_title)
    st.session_state.plot_title = plot_title_input

    if st.button("NACRTAJ I FITUJ", type="primary", use_container_width=True):
        try:
            subset = pd.DataFrame()
            subset['x'] = pd.to_numeric(st.session_state.df[x_col], errors='coerce')
            subset['y'] = pd.to_numeric(st.session_state.df[y_col], errors='coerce')
            
            if dx_col != "(Nema)": subset['dx'] = pd.to_numeric(st.session_state.df[dx_col], errors='coerce')
            else: subset['dx'] = 1e-9
            
            if dy_col != "(Nema)": subset['dy'] = pd.to_numeric(st.session_state.df[dy_col], errors='coerce')
            else: subset['dy'] = 1.0

            subset = subset.dropna()

            if len(subset) < 2:
                st.error("Nedovoljno podataka za grafik (najmanje 2 tačke).")
            else:
                def lin_f(B, x): return B[0]*x + B[1]
                model = odr.Model(lin_f)
                
                if dx_col == "(Nema)" and dy_col == "(Nema)":
                    mydata = odr.RealData(subset['x'], subset['y'])
                else:
                    mydata = odr.RealData(subset['x'], subset['y'], sx=subset['dx'], sy=subset['dy'])

                output = odr.ODR(mydata, model, beta0=[1., 0.]).run()
                k, n = output.beta[0], output.beta[1]
                dk, dn = output.sd_beta[0], output.sd_beta[1]

                st.success(f"**Rezultat fita:** y = kx + n")
                st.markdown(f"#### k = {k:.5f} ± {dk:.5f}")
                st.markdown(f"#### n = {n:.5f} ± {dn:.5f}")

                fig, ax = plt.subplots()
                ax.set_title(st.session_state.plot_title, fontweight="bold")
                
                xerr = subset['dx'] if dx_col != "(Nema)" else None
                yerr = subset['dy'] if dy_col != "(Nema)" else None
                ax.errorbar(subset['x'], subset['y'], xerr=xerr, yerr=yerr, fmt='o', label='Merenja', color='blue', capsize=3)
                
                x_range = np.linspace(subset['x'].min(), subset['x'].max(), 100)
                ax.plot(x_range, k*x_range + n, 'r-', linewidth=2, label='Fit')

                meta_x = st.session_state.col_meta.get(x_col, {'qty': x_col, 'unit': '-'})
                meta_y = st.session_state.col_meta.get(y_col, {'qty': y_col, 'unit': '-'})
                ax.set_xlabel(f"{meta_x['qty']} [{meta_x['unit']}]")
                ax.set_ylabel(f"{meta_y['qty']} [{meta_y['unit']}]")
                
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Greška pri crtanju: {e}")

# === TAB 3: FAJLOVI ===
with tab3:
    st.markdown("### Upravljanje projektom")
    
    project_json = save_project()
    st.download_button(
        label="💾 Skini .lab fajl",
        data=project_json,
        file_name="moj_eksperiment.lab",
        mime="application/json"
    )

    st.markdown("---")
    st.markdown("#### Otvori (Upload)")
    uploaded_file = st.file_uploader("Izaberi .lab fajl", type=["lab", "json"])
    if uploaded_file is not None:
        if st.button("Učitaj izabrani fajl"):
            load_project(uploaded_file)
    
    st.markdown("---")
    st.markdown("#### Uvoz iz Excela/CSV")
    imp_file = st.file_uploader("Izaberi Excel ili CSV", type=["xlsx", "xls", "csv"])
    if imp_file:
        if st.button("Uvezi podatke"):
            try:
                if imp_file.name.endswith('.csv'):
                    ndf = pd.read_csv(imp_file)
                else:
                    ndf = pd.read_excel(imp_file)
                
                # Prilagođavanje veličine tabele
                st.session_state.df = pd.DataFrame(index=range(max(15, len(ndf))), columns=st.session_state.df.columns)
                # Ako excel ima više kolona nego mi, dodaj ih
                while len(st.session_state.df.columns) < len(ndf.columns):
                    add_new_column()
                
                st.session_state.df[:] = ""
                
                app_cols = list(st.session_state.df.columns)
                for i, col_name in enumerate(ndf.columns):
                    if i < len(app_cols):
                        target = app_cols[i]
                        st.session_state.col_meta[target]['qty'] = str(col_name)
                        vals = ndf[col_name].astype(str).replace("nan", "")
                        for r, val in enumerate(vals):
                            st.session_state.df.iloc[r, i] = val
                
                st.success("Uspešno uvezeno!")
                st.rerun()
            except Exception as e:
                st.error(f"Greška: {e}")
    
    st.markdown("---")
    if st.button("🗑️ Obriši sve (Novi projekat)"):
        st.session_state.df = pd.DataFrame(index=range(15), columns=['A', 'B', 'C', 'D'])
        st.session_state.df[:] = ""
        st.session_state.col_meta = {col: {'qty': f'{col}', 'unit': '-'} for col in ['A', 'B', 'C', 'D']}
        st.session_state.plot_title = "Grafik zavisnosti"
        st.rerun()
