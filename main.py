import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


def load():
    df = pl.read_excel(st.secrets.url)
    X_train = df.drop("Class")
    y_train = df["Class"].map(lambda x: 1 if x == "Cammeo" else 0)
    n_samples = y_train.value_counts().max() - y_train.value_counts().min()
    resampled = resample(
        X_train.iloc[X_train.index[y_train == 1], :],
        n_samples=n_samples,
        replace=True,
        random_state=4399,
    )
    X_train_resampled = pl.concat([X_train, resampled])
    y_train_resampled = y_train.append(
        pl.Series(1).repeat(n_samples), ignore_index=True
    )
    scaler = StandardScaler()
    X_train_scaled = pl.DataFrame(
        scaler.fit_transform(X_train_resampled), columns=X_train_resampled.columns
    )
    return X_train_scaled, y_train_resampled, scaler


@st.cache_resource
def fit():
    X, y, scaler = load()
    pca = PCA(n_components=3)
    components = pca.fit_transform(X)
    svm_linear = svm.SVC(
        kernel="linear",
        probability=False,  # вывод оценки вероятности
        random_state=4399,
    )
    svm_linear.fit(X, y)
    y = y.map(lambda x: "Cammeo" if x == 1 else "Osmancik")
    return {
        "columns": X.columns,
        "y": y,
        "scaler": scaler,
        "model": svm_linear,
        "pca": pca,
        "components": components,
    }


def update():
    X_test = pl.DataFrame(
        np.array(
            [
                (
                    area,
                    perimeter,
                    major_axis_length,
                    minor_axis_length,
                    eccentricity,
                    convex_area,
                    extent,
                )
            ]
        ),
        columns=cache["columns"],
    )
    X_test_scaled = pl.DataFrame(
        cache["scaler"].transform(X_test), columns=cache["columns"]
    )
    if cache["model"].predict(X_test_scaled) == 0:
        st.header("A predicted specie is :blue[_Osmancik_] :upside_down_face:")
    else:
        st.header("A predicted specie is :orange[_Cammeo_] :slightly_smiling_face:")
    draw(X_test_scaled)


def draw(X_test):
    obj = cache["pca"].transform(X_test)
    total_var = cache["pca"].explained_variance_ratio_.sum() * 100
    fig = px.scatter_3d(
        np.vstack([cache["components"], obj]),
        x=0,
        y=1,
        z=2,
        color=np.hstack([cache["y"], "Candidate"]),
        title=f"Total Explained Variance: {total_var:.1f}%",
        labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
        width=800,
        height=600,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=25, b=0),
    )
    fig


cache = fit()
st.title("Rice Classifier")
st.sidebar.title("Morphological Features of Grain")
area = st.sidebar.slider(
    label=cache["columns"][0], min_value=7000, max_value=20000, step=200, value=10000
)
perimeter = st.sidebar.slider(
    label=cache["columns"][1], min_value=300, max_value=600, step=20, value=380
)
major_axis_length = st.sidebar.slider(
    label=cache["columns"][2], min_value=100, max_value=300, step=10, value=160
)
minor_axis_length = st.sidebar.slider(
    label=cache["columns"][3], min_value=50, max_value=120, step=10, value=80
)
eccentricity = st.sidebar.slider(
    label=cache["columns"][4], min_value=0.7, max_value=1.0, step=0.05, value=0.85
)
convex_area = st.sidebar.slider(
    label=cache["columns"][5], min_value=7000, max_value=20000, step=200, value=12000
)
extent = st.sidebar.slider(
    label=cache["columns"][6], min_value=0.3, max_value=1.0, step=0.05, value=0.8
)
update()
