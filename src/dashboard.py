import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests


# ---------------- LOAD DATA ----------------

df = pd.read_csv("outputs/cleaned_aqi.csv")
df['date'] = pd.to_datetime(df['date'])

# ---------------- LIVE AQI ----------------

def get_live_aqi(city):

    try:

        url = f"https://api.waqi.info/feed/{city}/?token=demo"

        r = requests.get(url, timeout=5).json()

        if r["status"] == "ok":

            return r["data"]["aqi"]

    except:

        return None


def aqi_color(value):

    if value is None:
        return "#94a3b8"

    if value <= 50:
        return "#22c55e"

    elif value <= 100:
        return "#eab308"

    elif value <= 150:
        return "#f97316"

    else:
        return "#ef4444"


# ---------------- WINDOW ----------------

root = tk.Tk()
root.title("Air Quality ML Dashboard")
root.geometry("1250x720")
root.configure(bg="#0b1120")

main = tk.Frame(root, bg="#0b1120")
main.pack(fill="both", expand=True, padx=20, pady=20)

title = tk.Label(main,
                 text="Air Quality Machine Learning Dashboard",
                 font=("Segoe UI", 22, "bold"),
                 fg="white",
                 bg="#0b1120")

title.pack(anchor="w", pady=10)

# ---------------- KPI CARDS ----------------

cards = tk.Frame(main, bg="#0b1120")
cards.pack(fill="x")

def create_card(text, value, color):

    frame = tk.Frame(cards,
                     bg="#111827",
                     width=160,
                     height=80)

    frame.pack(side="left",
               padx=6,
               pady=10)

    tk.Label(frame,
             text=text,
             fg="#9ca3af",
             bg="#111827").pack(anchor="w", padx=10, pady=5)

    label = tk.Label(frame,
                     text=value,
                     font=("Segoe UI",15,"bold"),
                     fg=color,
                     bg="#111827")

    label.pack(anchor="w", padx=10)

    return label


avg_pm25_card = create_card("Avg PM2.5",
                            round(df['PM2.5'].mean(),2),
                            "#22c55e")

max_pm25_card = create_card("Max PM2.5",
                            round(df['PM2.5'].max(),2),
                            "#ef4444")

min_pm25_card = create_card("Min PM2.5",
                            round(df['PM2.5'].min(),2),
                            "#38bdf8")

city_count_card = create_card("Cities",
                              df['City'].nunique(),
                              "#a78bfa")

records_card = create_card("Records",
                           len(df),
                           "#facc15")

avg_no2_card = create_card("Avg NO2",
                           round(df['NO2'].mean(),2),
                           "#fb923c")

live_aqi_card = create_card("Live AQI",
                            "...",
                            "#38bdf8")

# ---------------- LAYOUT ----------------

content = tk.Frame(main, bg="#0b1120")
content.pack(fill="both", expand=True)

sidebar = tk.Frame(content,
                   bg="#111827",
                   width=220)

sidebar.pack(side="left",
             fill="y",
             padx=10,
             pady=10)

graph_area = tk.Frame(content,
                      bg="#111827")

graph_area.pack(side="right",
                fill="both",
                expand=True,
                padx=10,
                pady=10)

# ---------------- CITY FILTER ----------------

tk.Label(sidebar,
         text="Select City",
         fg="white",
         bg="#111827",
         font=("Segoe UI",10,"bold")).pack(pady=12)

city_var = tk.StringVar()

city_dropdown = ttk.Combobox(sidebar,
                             values=["All"] + sorted(df['City'].unique()),
                             textvariable=city_var)

city_dropdown.set("All")

city_dropdown.pack(pady=5, padx=10)


def get_data():

    if city_var.get() == "All":
        return df

    return df[df['City'] == city_var.get()]

# ---------------- LIVE AQI UPDATE ----------------

def update_live_aqi():

    city = city_var.get()

    if city == "All":

        live_aqi_card.config(text="Select City",
                             fg="#94a3b8")

    else:

        value = get_live_aqi(city)

        if value is None:

            live_aqi_card.config(text="N/A",
                                 fg="#94a3b8")

        else:

            live_aqi_card.config(text=value,
                                 fg=aqi_color(value))

    root.after(60000, update_live_aqi)


city_dropdown.bind("<<ComboboxSelected>>",
                   lambda e: update_live_aqi())

# ---------------- GRAPH STYLE ----------------

def style_plot(fig, ax):

    fig.patch.set_facecolor("#0b1120")

    ax.set_facecolor("#111827")

    ax.tick_params(colors='white')

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    ax.title.set_color('white')

    for spine in ax.spines.values():
        spine.set_color("white")


# ---------------- GRAPH RENDER ----------------

current_canvas = None

def clear_graph():

    global current_canvas

    if current_canvas:
        current_canvas.get_tk_widget().destroy()


def display_plot(fig):

    global current_canvas

    current_canvas = FigureCanvasTkAgg(fig,
                                       master=graph_area)

    current_canvas.draw()

    current_canvas.get_tk_widget().pack(fill="both",
                                        expand=True)


# ---------------- GRAPHS ----------------

def trend_graph():

    clear_graph()

    data = get_data()

    daily = data.groupby('date')['PM2.5'].mean()

    fig, ax = plt.subplots()

    ax.plot(daily,
            color="#8b5cf6",
            linewidth=2)

    ax.set_title("PM2.5 Trend Over Time")

    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5")

    style_plot(fig, ax)

    display_plot(fig)


def distribution_graph():

    clear_graph()

    data = get_data()

    fig, ax = plt.subplots()

    sns.histplot(data['PM2.5'],
                 color="#6366f1",
                 ax=ax)

    ax.set_title("PM2.5 Distribution")

    style_plot(fig, ax)

    display_plot(fig)


def correlation_graph():

    clear_graph()

    data = get_data()

    fig, ax = plt.subplots()

    sns.heatmap(data[['PM2.5','PM10','NO2','SO2']].corr(),
                annot=True,
                cmap="magma",
                ax=ax)

    ax.set_title("Correlation Heatmap")

    style_plot(fig, ax)

    display_plot(fig)


def cluster_graph():

    clear_graph()

    data = get_data()

    features = data[['PM2.5','PM10','NO2','SO2']]

    scaled = StandardScaler().fit_transform(features)

    clusters = KMeans(n_clusters=3).fit_predict(scaled)

    fig, ax = plt.subplots()

    ax.scatter(data['PM2.5'],
               data['PM10'],
               c=clusters,
               cmap="viridis")

    ax.set_title("Pollution Clusters")

    ax.set_xlabel("PM2.5")
    ax.set_ylabel("PM10")

    style_plot(fig, ax)

    display_plot(fig)


def city_compare_graph():

    clear_graph()

    city_data = df.groupby('City')['PM2.5'].mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots()

    city_data.plot(kind="bar",
                   color="#4f46e5",
                   ax=ax)

    ax.set_title("Top 10 Most Polluted Cities")

    ax.set_xlabel("City")
    ax.set_ylabel("PM2.5")

    style_plot(fig, ax)

    display_plot(fig)

def show_future_aqi():

    city = city_var.get()

    if city == "All":
        return

    city_data = df[df['City'] == city]

    daily = city_data.groupby('date')['PM2.5'].mean()

    if len(daily) < 20:
        return

    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(daily, order=(5,1,0))

    model_fit = model.fit()

    forecast = model_fit.forecast(steps=10)

    # new window
    win = tk.Toplevel(root)

    win.title(f"{city} - Future AQI")

    win.geometry("600x500")

    win.configure(bg="#0b1120")

    # title
    tk.Label(win,
             text=f"{city} AQI Forecast (Next 10 Days)",
             font=("Segoe UI",14,"bold"),
             fg="white",
             bg="#0b1120").pack(pady=10)

    # graph
    fig, ax = plt.subplots()

    future_dates = pd.date_range(daily.index[-1],
                                 periods=10)

    ax.plot(future_dates,
            forecast,
            color="#22c55e",
            linewidth=2)

    ax.set_title("Predicted AQI")

    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")

    fig.patch.set_facecolor("#0b1120")

    ax.set_facecolor("#111827")

    ax.tick_params(colors='white')

    ax.title.set_color("white")

    ax.xaxis.label.set_color("white")

    ax.yaxis.label.set_color("white")

    canvas = FigureCanvasTkAgg(fig, master=win)

    canvas.draw()

    canvas.get_tk_widget().pack(fill="both", expand=True)

    # values table
    frame = tk.Frame(win, bg="#0b1120")

    frame.pack(pady=10)

    for i, value in enumerate(forecast):

        tk.Label(frame,
                 text=f"Day {i+1}: {round(value,2)}",
                 fg="#38bdf8",
                 bg="#0b1120").pack()

# ---------------- BUTTONS ----------------

def create_button(text, command):

    btn = tk.Button(sidebar,
                    text=text,
                    command=command,
                    bg="#1f2937",
                    fg="white",
                    relief="flat",
                    font=("Segoe UI",10),
                    padx=10,
                    pady=8)

    btn.pack(fill="x",
             padx=12,
             pady=6)


create_button("Trend", trend_graph)

create_button("Distribution", distribution_graph)

create_button("Correlation", correlation_graph)

create_button("Clusters", cluster_graph)

create_button("City Comparison", city_compare_graph)

create_button("Future AQI", show_future_aqi)
# start live AQI loop
update_live_aqi()

root.mainloop()