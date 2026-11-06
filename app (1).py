from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prognoza cen mieszkań", page_icon="🏠", layout="wide")

GEOJSON_URL = "https://raw.githubusercontent.com/jusuff/PolandGeoJson/main/data/poland.counties.json"
GUS_UNITS_URL = "https://bdl.stat.gov.pl/api/v1/units?level=5&page-size=500&format=json"
NBP_GOLD_URL = "https://api.nbp.pl/api/cenyzlota/last/1/?format=json"


def get_json(url: str, timeout: int = 30) -> Any:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=3600, show_spinner=False)
def load_sources() -> tuple[dict, pd.DataFrame, str, str]:
    geojson = get_json(GEOJSON_URL)
    features = geojson.get("features", [])
    units = pd.DataFrame([
        {"terc": str(f.get("properties", {}).get("terc", "")), "powiat": f.get("properties", {}).get("name", "Nieznany")}
        for f in features
    ]).query("terc != ''").drop_duplicates("terc").reset_index(drop=True)
    try:
        gus = get_json(GUS_UNITS_URL, 15)
        gus_status = f"dostępne ({len(gus.get('results', gus.get('data', [])))} jednostek)"
    except Exception as exc:
        gus_status = f"niedostępne ({type(exc).__name__})"
    try:
        nbp = get_json(NBP_GOLD_URL, 15)
        gold = float(nbp[0]["cena"])
        nbp_status = f"dostępne — złoto {gold:.2f} zł/g"
    except Exception as exc:
        gold = 518.0
        nbp_status = f"niedostępne ({type(exc).__name__}); użyto wartości neutralnej"
    return geojson, units, gus_status, nbp_status


@st.cache_data(show_spinner=False)
def make_data(units: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    def stable(text: str) -> int:
        return int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)
    bonuses = {"Warszawa": 9000, "Kraków": 5000, "Wrocław": 3500, "Gdańsk": 3500, "Poznań": 2800, "Gdynia": 2600, "Sopot": 4200, "Katowice": 1500, "Łódź": 1000, "Szczecin": 1200, "Lublin": 700, "Rzeszów": 500}
    data = units.copy()
    data["urbanizacja"] = data["powiat"].map(lambda x: .25 + (stable(x) % 76) / 100)
    data["premia_miasta"] = data["powiat"].map(lambda x: next((v for k, v in bonuses.items() if k.lower() in str(x).lower()), 0))
    data["cena_m2"] = (5200 + data["urbanizacja"] * 3100 + data["premia_miasta"] + rng.normal(0, 350, len(data))).clip(3500, 30000).round(0)
    model_data = data.loc[data.index.repeat(6)].reset_index(drop=True)
    model_data["metraz"] = rng.integers(30, 121, len(model_data))
    model_data["cena_m2_obserwowana"] = (model_data["cena_m2"] + rng.normal(0, 220, len(model_data))).clip(3000)
    model_data["cena_calkowita"] = (model_data["metraz"] * model_data["cena_m2_obserwowana"] + rng.normal(0, 18000, len(model_data))).clip(50000).round(0)
    return data, model_data


def train(model_data: pd.DataFrame):
    features = ["metraz", "cena_m2", "urbanizacja", "premia_miasta"]
    x_train, x_test, y_train, y_test = train_test_split(model_data[features], model_data["cena_calkowita"], test_size=.2, random_state=42)
    model = LinearRegression().fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, features, r2_score(y_test, pred), mean_absolute_error(y_test, pred), y_test, pred


def money(value: float) -> str:
    return f"{value:,.0f} zł".replace(",", " ")


try:
    geojson, units, gus_status, nbp_status = load_sources()
    data, model_data = make_data(units)
except Exception as exc:
    st.error(f"Nie udało się pobrać granic powiatów: {exc}")
    st.stop()

model, features, r2, mae, y_test, pred = train(model_data)

st.title("Inteligentna prognoza cen mieszkań w Polsce")
st.caption("Projekt demonstracyjny Data Science — ceny mieszkań są symulowane, ponieważ publiczne API nie dostarcza kompletnej tabeli cen dla wszystkich powiatów.")
with st.sidebar:
    st.header("Źródła danych")
    st.write(f"GUS BDL: {gus_status}")
    st.write(f"NBP: {nbp_status}")
    st.divider()
    selected_name = st.selectbox("Wybierz powiat", sorted(data["powiat"].unique()), index=0)
    area = st.slider("Powierzchnia mieszkania (m²)", 30, 120, 60)

selected = data.loc[data["powiat"] == selected_name].iloc[0]
input_row = pd.DataFrame([{"metraz": area, "cena_m2": selected["cena_m2"], "urbanizacja": selected["urbanizacja"], "premia_miasta": selected["premia_miasta"]}])
total_price = max(0, float(model.predict(input_row)[0]))
predicted_m2 = total_price / area

k1, k2, k3, k4 = st.columns(4)
k1.metric("Wybrany powiat", selected_name)
k2.metric("Powierzchnia", f"{area} m²")
k3.metric("Prognoza ceny za m²", money(predicted_m2))
k4.metric("Prognoza ceny całkowitej", money(total_price))

map_tab, chart_tab, model_tab = st.tabs(["Mapa powiatów", "Porównanie cen", "Model i metryki"])
with map_tab:
    st.subheader("Kliknij powiat na mapie lub wybierz go w panelu bocznym")
    map_fig = px.choropleth(data, geojson=geojson, locations="terc", featureidkey="properties.terc", color="cena_m2", color_continuous_scale="YlOrRd", hover_name="powiat", hover_data={"terc": False, "cena_m2": ":,.0f"}, labels={"cena_m2": "Cena za m²"}, title="Ceny za m² — dane symulowane")
    map_fig.update_geos(fitbounds="locations", visible=False)
    map_fig.update_layout(height=650, margin=dict(r=0, t=60, l=0, b=0))
    event = st.plotly_chart(map_fig, use_container_width=True, on_select="rerun", key="county_map")
    if event and getattr(event, "selection", None) and event.selection.get("point_indices"):
        clicked_index = event.selection["point_indices"][0]
        st.info(f"Kliknięto: {data.iloc[clicked_index]['powiat']}. Aby przeliczyć wynik, wybierz ten powiat także w panelu bocznym.")
with chart_tab:
    st.subheader("Ranking cen mieszkań w powiatach")
    top = data.nlargest(20, "cena_m2").sort_values("cena_m2")
    fig = px.bar(top, x="cena_m2", y="powiat", orientation="h", color="cena_m2", color_continuous_scale="Viridis", labels={"cena_m2": "Cena za m² (zł)", "powiat": "Powiat"})
    st.plotly_chart(fig, use_container_width=True)
with model_tab:
    a, b = st.columns(2)
    a.metric("R²", f"{r2:.3f}")
    b.metric("MAE", money(mae))
    comparison = pd.DataFrame({"Rzeczywista cena": y_test, "Prognozowana cena": pred})
    fig = px.scatter(comparison, x="Rzeczywista cena", y="Prognozowana cena", title="Wartości rzeczywiste i prognozowane")
    st.plotly_chart(fig, use_container_width=True)
    st.info("R² pokazuje dopasowanie modelu, a MAE średni błąd prognozy w złotych. Model ma charakter dydaktyczny i nie służy do profesjonalnej wyceny nieruchomości.")

st.divider()
st.warning("Ceny mieszkań zostały symulowane na podstawie urbanizacji i premii największych miast. Nie używaj wyników do decyzji inwestycyjnych ani formalnej wyceny.")
