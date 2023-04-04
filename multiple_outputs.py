# Training models with multiple outputs.
# i.e X ~ y,z

# %%
import pandas as pd
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import plotly.express as px

# %%
ee_dataset_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
)


def get_data() -> pd.DataFrame:
    response = requests.get(ee_dataset_url)
    file_bytes = io.BytesIO(response.content)
    return pd.read_excel(file_bytes)


def get_x(df: pd.DataFrame) -> pd.DataFrame:
    return df[[x for x in df.columns.values if x[0] != "Y"]]


def get_y(df: pd.DataFrame) -> pd.DataFrame:
    return df[[x for x in df.columns.values if x[0] == "Y"]]


def make_model(num_features: int) -> tf.keras.Model:
    input_layer = tf.keras.layers.Input(shape=(num_features,), name="input_layer")
    first_dense = tf.keras.layers.Dense(
        units="128", activation="relu", name="first_dense"
    )(input_layer)
    second_dense = tf.keras.layers.Dense(
        units="128", activation="relu", name="second_dense"
    )(first_dense)

    y1_out = tf.keras.layers.Dense(units="1", name="y1_output")(second_dense)

    third_dense = tf.keras.layers.Dense(
        units="64", activation="relu", name="third_dense"
    )(second_dense)

    y2_out = tf.keras.layers.Dense(units="1", name="y2_output")(third_dense)

    return tf.keras.models.Model(inputs=input_layer, outputs=[y1_out, y2_out])


def get_metrics(model, test_x, test_y):
    loss, y1_loss, y2_loss, y1_rmse, y2_rmse = model.evaluate(x=test_x, y=test_y)
    return "Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(
        loss, y1_loss, y1_rmse, y2_loss, y2_rmse
    )


# %%
if __name__ == "__main__":
    # %%
    df = get_data()
    df.head(10)

    # %%
    train, test = train_test_split(df, test_size=0.2)

    # %%
    train_x = get_x(train)
    train_y = get_y(train)

    test_x = get_x(test)
    test_y = get_y(test)
    # %%
    # Normaliser for normalising the data
    normalizer = StandardScaler()
    normalizer = normalizer.fit(train_x)
    # %%
    # Normalise the train set
    norm_train_x = pd.DataFrame(
        normalizer.transform(train_x), columns=train_x.columns.values
    )
    norm_train_x

    # %%
    norm_test_x = pd.DataFrame(
        normalizer.transform(test_x), columns=test_x.columns.values
    )
    norm_test_x
    # %%
    model = make_model(norm_train_x.columns.shape[0])

    # %%
    model.summary()
    # %%
    tf.keras.utils.plot_model(model)
    # %%
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

    # %%
    model.compile(
        optimizer=optimizer,
        loss={"y1_output": "mse", "y2_output": "mse"},
        metrics={
            "y1_output": tf.keras.metrics.RootMeanSquaredError(),
            "y2_output": tf.keras.metrics.RootMeanSquaredError(),
        },
    )

    # %%
    # Train the model
    history = model.fit(
        norm_train_x,
        train_y,
        epochs=500,
        batch_size=10,
        validation_data=(norm_test_x, test_y),
    )
    # %%
    # Print the model metrics
    print(get_metrics(model, test_x, test_y))
    # %%
    y_pred = model.predict(test_x)
    y_pred
    # %%
    px.scatter(x=test_y["Y1"].tolist(), y=y_pred[0][:, 0])
# %%
