from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="NEXUS | Business Intelligence", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

COLORS = {"navy": "#0B1220", "panel": "#111B2E", "accent": "#45D6C7", "blue": "#6C8CFF", "orange": "#FFB86B", "muted": "#8EA0B8", "text": "#F4F7FB", "danger": "#FF7185"}
PLOT_TEMPLATE = "plotly_dark"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background: {COLORS['navy']}; color: {COLORS['text']}; }}
[data-testid="stSidebar"] {{ background: #0E1728; border-right: 1px solid #1E2B43; }}
[data-testid="stSidebar"] * {{ color: #DCE5F2; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; }}
h1 {{ font-size: 2.4rem !important; }}
.block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
.kpi {{ background: linear-gradient(145deg, #15223A, #10192B); border: 1px solid #223552; border-radius: 16px; padding: 20px 22px; min-height: 125px; box-shadow: 0 12px 30px rgba(0,0,0,.12); }}
.kpi-label {{ color: {COLORS['muted']}; font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 600; }}
.kpi-value {{ color: {COLORS['text']}; font-family: 'Space Grotesk'; font-size: 1.85rem; font-weight: 700; margin-top: 8px; }}
.kpi-delta {{ color: {COLORS['accent']}; font-size: .82rem; margin-top: 5px; }}
.section-tag {{ color: {COLORS['accent']}; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; font-size: .72rem; margin-bottom: 4px; }}
.insight {{ background: #121E32; border-left: 3px solid {COLORS['accent']}; border-radius: 8px; padding: 14px 16px; margin: 9px 0; color: #DCE5F2; }}
.small-muted {{ color: {COLORS['muted']}; font-size: .85rem; }}
</style>
""", unsafe_allow_html=True)


def money(value: float) -> str:
    return f"{value:,.0f} zł".replace(",", " ")


def pct(value: float) -> str:
    return f"{value:.1f}%"


def make_demo_data(n: int = 4200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2026-12-31", freq="D")
    categories = {
        "Elektronika": [("Słuchawki Pro", 299, 150), ("Smartwatch X", 699, 410), ("Głośnik Mini", 249, 125)],
        "Dom i ogród": [("Ekspres Barista", 899, 560), ("Robot sprzątający", 1199, 740), ("Lampa Ambient", 189, 82)],
        "Sport": [("Mata treningowa", 149, 65), ("Rower miejski", 1799, 1180), ("Plecak Outdoor", 329, 155)],
        "Moda": [("Kurtka miejska", 449, 210), ("Sneakers Urban", 379, 175), ("Torba skórzana", 519, 245)],
        "Zdrowie": [("Masażer Pulse", 279, 125), ("Termometr Smart", 159, 66), ("Opaska Recovery", 229, 108)],
    }
    regions = ["Północ", "Południe", "Centrum", "Zachód", "Wschód"]
    channels = ["E-commerce", "Sklep stacjonarny", "Partner B2B"]
    category_names = list(categories)
    rows = []
    for _ in range(n):
        date = dates[rng.integers(0, len(dates))]
        month_factor = 1 + 0.18 * np.sin((date.month - 2) / 12 * 2 * np.pi) + (0.30 if date.month in (11, 12) else 0)
        category = rng.choice(category_names, p=[.25, .20, .20, .20, .15])
        product, base_price, base_cost = categories[category][rng.integers(0, 3)]
        region = rng.choice(regions, p=[.21, .22, .27, .18, .12])
        channel = rng.choice(channels, p=[.52, .28, .20])
        trend = 1 + (date.year - 2024) * .10
        channel_factor = {"E-commerce": 1.18, "Sklep stacjonarny": .92, "Partner B2B": 1.28}[channel]
        quantity = max(1, int(rng.poisson(3.3 * month_factor * trend * channel_factor)))
        unit_price = round(base_price * rng.normal(1, .035), 2)
        unit_cost = round(base_cost * rng.normal(1, .025), 2)
        revenue = round(quantity * unit_price, 2)
        margin = round(revenue - quantity * unit_cost, 2)
        rows.append([date, category, product, region, channel, quantity, unit_price, unit_cost, revenue, margin])
    return pd.DataFrame(rows, columns=["Data", "Kategoria", "Produkt", "Region", "Kanał", "Ilość", "Cena jednostkowa", "Koszt jednostkowy", "Przychód", "Marża"])


REQUIRED_COLUMNS = ["Data", "Kategoria", "Produkt", "Region", "Kanał", "Ilość", "Cena jednostkowa", "Koszt jednostkowy", "Przychód", "Marża"]
ALIASES = {"date": "Data", "transaction_date": "Data", "category": "Kategoria", "product": "Produkt", "region": "Region", "channel": "Kanał", "quantity": "Ilość", "unit_price": "Cena jednostkowa", "unit_cost": "Koszt jednostkowy", "revenue": "Przychód", "margin": "Marża"}


def prepare_uploaded(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: ALIASES.get(str(c).strip().lower(), c) for c in df.columns}).copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Brakujące kolumny: " + ", ".join(missing))
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    for c in ["Ilość", "Cena jednostkowa", "Koszt jednostkowy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Data", "Kategoria", "Produkt", "Region", "Kanał", "Ilość", "Cena jednostkowa", "Koszt jednostkowy"])
    df["Przychód"] = df["Ilość"] * df["Cena jednostkowa"]
    df["Marża"] = df["Ilość"] * (df["Cena jednostkowa"] - df["Koszt jednostkowy"])
    return df[REQUIRED_COLUMNS].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_demo_data() -> pd.DataFrame:
    return make_demo_data()


def card(label: str, value: str, delta: str = "") -> None:
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-delta">{delta}</div></div>', unsafe_allow_html=True)


def chart(fig: go.Figure, height: int = 340) -> None:
    fig.update_layout(template=PLOT_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#DCE5F2", margin=dict(l=12, r=12, t=45, b=12), height=height, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown("## NEXUS")
        st.caption("Business Intelligence / 2026")
        st.divider()
        st.markdown("### Filtry analityczne")
        min_date, max_date = df["Data"].min().date(), df["Data"].max().date()
        selected_dates = st.date_input("Zakres dat", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start, end = selected_dates
        else:
            start = end = selected_dates
        selected_regions = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
        selected_categories = st.multiselect("Kategoria", sorted(df["Kategoria"].unique()), default=sorted(df["Kategoria"].unique()))
        selected_channels = st.multiselect("Kanał sprzedaży", sorted(df["Kanał"].unique()), default=sorted(df["Kanał"].unique()))
        st.divider()
        st.caption("Brak pliku = syntetyczny dataset demonstracyjny (4 200 transakcji).")
    mask = (df["Data"].dt.date >= start) & (df["Data"].dt.date <= end) & df["Region"].isin(selected_regions) & df["Kategoria"].isin(selected_categories) & df["Kanał"].isin(selected_channels)
    return df.loc[mask].copy()


def executive(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-tag">01 / Executive overview</div>', unsafe_allow_html=True)
    st.title("NEXUS | Business Intelligence")
    st.write("Szybki obraz wyników sprzedaży — oparty na aktualnie wybranym wycinku danych.")
    revenue, profit = df["Przychód"].sum(), df["Marża"].sum()
    margin = profit / revenue * 100 if revenue else 0
    prev = df[df["Data"] < df["Data"].max() - pd.Timedelta(days=30)]["Przychód"].sum()
    current = df[df["Data"] >= df["Data"].max() - pd.Timedelta(days=30)]["Przychód"].sum()
    delta = (current / prev - 1) * 100 if prev else 0
    cols = st.columns(4)
    with cols[0]: card("Przychód całkowity", money(revenue), f"{len(df):,} transakcji".replace(",", " "))
    with cols[1]: card("Zysk brutto", money(profit), f"Marża operacyjna {pct(margin)}")
    with cols[2]: card("Średnia transakcja", money(df["Przychód"].mean() if len(df) else 0), f"{df['Ilość'].sum():,.0f} sztuk".replace(",", " "))
    with cols[3]: card("Dynamika 30 dni", f"{delta:+.1f}%", "vs wcześniejszy okres")
    st.write("")
    monthly = df.set_index("Data").resample("MS").agg(Przychód=("Przychód", "sum"), Marża=("Marża", "sum")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["Data"], y=monthly["Przychód"], mode="lines+markers", name="Przychód", line=dict(color=COLORS["accent"], width=3), fill="tozeroy", fillcolor="rgba(69,214,199,.12)"))
    fig.add_trace(go.Scatter(x=monthly["Data"], y=monthly["Marża"], mode="lines", name="Marża", line=dict(color=COLORS["orange"], width=2, dash="dot")))
    fig.update_layout(title="Dynamika przychodów i marży", yaxis_tickprefix=" zł ", hovermode="x unified")
    chart(fig, 360)
    c1, c2 = st.columns(2)
    with c1:
        cat = df.groupby("Kategoria", as_index=False)["Przychód"].sum().sort_values("Przychód")
        chart(px.bar(cat, x="Przychód", y="Kategoria", orientation="h", title="Sprzedaż według kategorii", color="Przychód", color_continuous_scale=["#27415D", COLORS["accent"]]), 320)
    with c2:
        reg = df.groupby("Region", as_index=False).agg(Przychód=("Przychód", "sum"), Marża=("Marża", "sum"))
        reg["Marża %"] = reg["Marża"] / reg["Przychód"] * 100
        chart(px.bar(reg.sort_values("Marża %"), x="Region", y="Marża %", title="Rentowność regionów", color="Marża %", color_continuous_scale=["#FF7185", COLORS["accent"]]), 320)
    channel = df.groupby("Kanał", as_index=False).agg(Przychód=("Przychód", "sum"), Marża=("Marża", "sum"))
    chart(px.bar(channel, x="Kanał", y=["Przychód", "Marża"], barmode="group", title="Porównanie kanałów sprzedaży", color_discrete_sequence=[COLORS["blue"], COLORS["orange"]]), 320)


def explorer(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-tag">02 / Data explorer</div>', unsafe_allow_html=True)
    st.title("Eksplorator danych")
    st.write(f"Przeanalizuj **{len(df):,}** rekordów po zastosowaniu filtrów.".replace(",", " "))
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Pobierz przefiltrowane dane CSV", csv, "nexus_filtered_sales.csv", "text/csv")
    st.dataframe(df.sort_values("Data", ascending=False), use_container_width=True, height=320, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Statystyki opisowe")
        st.dataframe(df[["Ilość", "Cena jednostkowa", "Koszt jednostkowy", "Przychód", "Marża"]].describe().T.round(2), use_container_width=True)
    with c2:
        st.subheader("Rozkład przychodów")
        chart(px.histogram(df, x="Przychód", nbins=35, color_discrete_sequence=[COLORS["accent"]], title="Transakcje według wartości"), 320)
    numeric = df.select_dtypes(include=np.number)
    if len(numeric.columns) > 1:
        st.subheader("Macierz korelacji")
        corr = numeric.corr().round(2)
        chart(px.imshow(corr, text_auto=True, color_continuous_scale="Teal", zmin=-1, zmax=1, title="Zależności między zmiennymi"), 430)
    q1, q3 = df["Przychód"].quantile([.25, .75]); iqr = q3 - q1
    outliers = df[(df["Przychód"] < q1 - 1.5 * iqr) | (df["Przychód"] > q3 + 1.5 * iqr)]
    st.info(f"Wykryto **{len(outliers)}** obserwacji odstających dla przychodu (reguła IQR).")


def forecast(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-tag">03 / Predictive analytics</div>', unsafe_allow_html=True)
    st.title("AI Sales Forecast")
    st.write("Model Random Forest uczy się wzorców sprzedaży na podstawie cech transakcji i sezonowości.")
    work = df.copy()
    work["Rok"] = work["Data"].dt.year; work["Miesiąc"] = work["Data"].dt.month; work["Dzień tygodnia"] = work["Data"].dt.dayofweek
    features = ["Kategoria", "Region", "Kanał", "Ilość", "Cena jednostkowa", "Koszt jednostkowy", "Rok", "Miesiąc", "Dzień tygodnia"]
    X, y = work[features], work["Przychód"]
    if len(work) < 30:
        st.warning("Do trenowania modelu potrzeba co najmniej 30 rekordów."); return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    categorical = ["Kategoria", "Region", "Kanał"]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical)], remainder="passthrough")
    model = Pipeline([("prep", pre), ("rf", RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1))])
    with st.spinner("Trenowanie modelu..."): model.fit(X_train, y_train)
    pred = model.predict(X_test); mae, r2 = mean_absolute_error(y_test, pred), r2_score(y_test, pred)
    a, b, c = st.columns(3)
    with a: card("MAE", money(mae), "średni błąd bezwzględny")
    with b: card("R²", f"{r2:.3f}", "wyjaśniona zmienność")
    with c: card("Zbiór testowy", f"{len(y_test)}", "transakcji poza treningiem")
    comparison = pd.DataFrame({"Rzeczywiste": y_test.values, "Prognozowane": pred}).reset_index(drop=True)
    chart(px.scatter(comparison, x="Rzeczywiste", y="Prognozowane", title="Wartości rzeczywiste vs prognozowane", trendline="ols", color_discrete_sequence=[COLORS["accent"]]), 380)
    st.subheader("Prognoza dla nowej transakcji")
    c1, c2, c3 = st.columns(3)
    with c1: category = st.selectbox("Kategoria", sorted(work["Kategoria"].unique()), key="fc")
    with c2: region = st.selectbox("Region", sorted(work["Region"].unique()), key="fr")
    with c3: channel = st.selectbox("Kanał", sorted(work["Kanał"].unique()), key="fh")
    c4, c5, c6 = st.columns(3)
    with c4: qty = st.number_input("Ilość", 1, 1000, 4, key="fq")
    with c5: price = st.number_input("Cena jednostkowa", 1.0, 10000.0, 399.0, key="fp")
    with c6: cost = st.number_input("Koszt jednostkowy", 1.0, 10000.0, 210.0, key="fk")
    if st.button("Uruchom prognozę", type="primary"):
        sample = pd.DataFrame([[category, region, channel, qty, price, cost, pd.Timestamp.now().year, pd.Timestamp.now().month, pd.Timestamp.now().dayofweek]], columns=features)
        st.success(f"Szacowany przychód: **{money(float(model.predict(sample)[0]))}**")
    st.caption("Interpretacja: MAE pokazuje przeciętną różnicę w zł, a R² — jak dobrze model odwzorowuje zmienność danych. Model jest demonstracyjny: nie uwzględnia kampanii, stanów magazynowych, konkurencji ani szoków rynkowych.")


def simulator(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-tag">04 / Scenario planning</div>', unsafe_allow_html=True)
    st.title("Business Simulator")
    st.write("Zbuduj scenariusz i zobacz, jak zmiana założeń wpływa na wynik finansowy.")
    c1, c2 = st.columns(2)
    with c1:
        customer = st.slider("Zmiana liczby klientów", -50, 100, 0, 5, format="%d%%") / 100
        price = st.slider("Zmiana cen produktów", -30, 30, 0, 1, format="%d%%") / 100
    with c2:
        cost = st.slider("Zmiana kosztów jednostkowych", -30, 50, 0, 1, format="%d%%") / 100
        volume = st.slider("Zmiana wolumenu sprzedaży", -50, 100, 0, 5, format="%d%%") / 100
    base_rev, base_profit = df["Przychód"].sum(), df["Marża"].sum()
    scen_rev = base_rev * (1 + customer) * (1 + price) * (1 + volume)
    scen_cost = (base_rev - base_profit) * (1 + customer) * (1 + cost) * (1 + volume)
    scen_profit, scen_margin = scen_rev - scen_cost, (scen_rev - scen_cost) / scen_rev * 100 if scen_rev else 0
    cols = st.columns(4)
    with cols[0]: card("Przychód bazowy", money(base_rev), "aktualny zakres filtrów")
    with cols[1]: card("Przychód scenariusza", money(scen_rev), f"{(scen_rev / base_rev - 1) * 100:+.1f}%")
    with cols[2]: card("Zysk scenariusza", money(scen_profit), f"{(scen_profit - base_profit) / abs(base_profit) * 100:+.1f}% vs baza")
    with cols[3]: card("Marża scenariusza", pct(scen_margin), f"baza: {pct(base_profit / base_rev * 100)}")
    comparison = pd.DataFrame({"Wskaźnik": ["Przychód", "Zysk"], "Bazowy": [base_rev, base_profit], "Scenariusz": [scen_rev, scen_profit]})
    chart(px.bar(comparison, x="Wskaźnik", y=["Bazowy", "Scenariusz"], barmode="group", title="Baza vs scenariusz", color_discrete_sequence=[COLORS["blue"], COLORS["accent"]]), 350)


def insights(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-tag">05 / Decision intelligence</div>', unsafe_allow_html=True)
    st.title("Business Insights")
    st.write("Wnioski generowane deterministycznie na podstawie obliczonych wskaźników — bez fikcyjnych rekomendacji AI.")
    if df.empty: st.warning("Brak danych dla bieżących filtrów."); return
    cat = df.groupby("Kategoria").agg(revenue=("Przychód", "sum"), margin=("Marża", "sum"), qty=("Ilość", "sum")); reg = df.groupby("Region").agg(revenue=("Przychód", "sum"), margin=("Marża", "sum"))
    top_cat, top_reg = cat["revenue"].idxmax(), (reg["margin"] / reg["revenue"]).idxmax()
    monthly = df.set_index("Data").resample("MS")["Przychód"].sum(); best, worst = monthly.idxmax(), monthly.idxmin()
    overall_margin = df["Marża"].sum() / df["Przychód"].sum() * 100
    st.markdown(f'<div class="insight"><b>Lider sprzedaży:</b> Kategoria <b>{top_cat}</b> generuje {money(cat.loc[top_cat, "revenue"])} przychodu.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight"><b>Najwyższa rentowność:</b> Region <b>{top_reg}</b> ma marżę na poziomie {pct(reg.loc[top_reg, "margin"] / reg.loc[top_reg, "revenue"] * 100)}.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight"><b>Sezonowość:</b> Najlepszy miesiąc to <b>{best.strftime("%B %Y")}</b>, a najsłabszy <b>{worst.strftime("%B %Y")}</b>.</div>', unsafe_allow_html=True)
    low_margin = cat.assign(rate=cat["margin"] / cat["revenue"] * 100).sort_values("rate").iloc[0]
    st.markdown(f'<div class="insight"><b>Obszar do poprawy:</b> Najniższą marżę ma kategoria <b>{low_margin.name}</b> ({pct(low_margin["rate"])}). Warto zweryfikować ceny, rabaty i koszty dostaw.</div>', unsafe_allow_html=True)
    if overall_margin < 25:
        st.warning(f"Marża całkowita wynosi {pct(overall_margin)} — poniżej progu 25%. Zalecana analiza miksu produktowego i polityki cenowej.")
    else:
        st.success(f"Marża całkowita wynosi {pct(overall_margin)} — zdrowy poziom w analizowanym zakresie.")
    st.subheader("Najważniejsze zależności")
    corr = df[["Ilość", "Cena jednostkowa", "Koszt jednostkowy", "Przychód", "Marża"]].corr()["Przychód"].drop("Przychód").sort_values(ascending=False)
    for name, value in corr.items(): st.write(f"**{name}** — korelacja z przychodem: `{value:+.2f}`")


def main() -> None:
    df = cached_demo_data()
    with st.sidebar:
        uploaded = st.file_uploader("Wgraj CSV (opcjonalnie)", type=["csv"], key="global_upload")
    if uploaded is not None:
        try:
            df = prepare_uploaded(pd.read_csv(uploaded))
            st.sidebar.success(f"Załadowano {len(df):,} rekordów".replace(",", " "))
        except Exception as exc:
            st.sidebar.error(f"Nie udało się wczytać pliku: {exc}")
    filtered = apply_global_filters(df)
    if filtered.empty:
        st.warning("Brak danych dla wybranych filtrów. Rozszerz zakres lub wybierz więcej wartości.")
        return
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Executive Dashboard", "Data Explorer", "AI Sales Forecast", "Business Simulator", "Business Insights"])
    with tab1: executive(filtered)
    with tab2: explorer(filtered)
    with tab3: forecast(filtered)
    with tab4: simulator(filtered)
    with tab5: insights(filtered)
    st.sidebar.divider(); st.sidebar.caption("NEXUS v1.0 • Python / Streamlit / Plotly")


if __name__ == "__main__":
    main()
