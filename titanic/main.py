import altair as alt
import streamlit as st
import polars as pl
import datetime as dt
import numpy as np
import random
import statsmodels.api as sm
from xgboost import XGBClassifier
from patsy import dmatrices, dmatrix

def print_data_frame(df: pl.DataFrame):
    column_config = {}
    for i, (col_name, col_type) in enumerate(df.schema.items()):
        column_config[i + 1] = {"label": col_name}
        # Sadly only goes down to millisecond accuracy. For more use st.write(df) instead if
        # you need it. This limitation comes from the underlying MomentJS library
        if isinstance(col_type, (dt.datetime, pl.Datetime)):
            column_config[i + 1] = st.column_config.DatetimeColumn(
                col_name, format="YYYY-MM-DD HH:mm:ss.SSS"
            )
    st.dataframe(df, column_config=column_config)

# def modify_df(df: pl.DataFrame):


st.write("Astrids titanic app")


df = pl.read_csv('data/train.csv')

df = df.with_columns(
    pl.concat_str(
        [pl.col("Pclass"),
         pl.col("Sex")
         ],
         separator=" ",
    ).alias("Class Sex"),
    )


df = df.with_columns(
    pl.when(pl.col('Age')>6).then(0).otherwise(pl.col('Age')).fill_null(strategy="zero").alias("Age_kids")
)

# print_data_frame(df)

pandas_df = df.to_pandas()

y, X = dmatrices('Survived ~ Fare + Age + Sex + Pclass + SibSp + Parch + Embarked', data=pandas_df, return_type='dataframe')

# Fit the model
model = sm.OLS(y, X)
results = model.fit_regularized(alpha=0.5, L1_wt=0.0)
# st.write(results.summary())

coefficients = np.array(results.params)
Survived_fitted = X @ coefficients

pandas_df['Survived_fitted'] = Survived_fitted

st.dataframe(pandas_df)

scatter = (
    alt.Chart(pandas_df).mark_point()
    .encode(
        x='Fare:Q',
        y='Survived:Q'
    )
)

line = (
    alt.Chart(pandas_df).mark_line()
    .encode(
       x='Fare:Q',
       y='Survived_fitted:Q' 
    )
)

chart4 = scatter + line
st.altair_chart(chart4, use_container_width=True)

#Modeling the testdata
df_test = pl.read_csv('data/test.csv')

df_test = df_test.with_columns(
    pl.when(pl.col('Age')>6).then(0).otherwise(pl.col('Age')).fill_null(strategy="zero").alias("Age_kids")
)

pandas_df_test = df_test.to_pandas()
X_test = dmatrix('Fare + Age + Sex + Pclass + SibSp + Parch + Embarked ', data=pandas_df_test, return_type='dataframe')
Survived_test = ((X_test @ coefficients)>0.6)


st.write(Survived_test)

pandas_df_test['Survived_test'] = Survived_test
pandas_df_test['Survived_test'] =pandas_df_test['Survived_test'].fillna(False)
pandas_df_test['Survived_test'] =pandas_df_test['Survived_test'].astype(int)

pandas_df_test[['PassengerId','Survived_test']].to_csv('answer.csv', index=False)
color_scale = alt.Scale(domain=[0, 1], range=['red', 'green'])

chart1 = (
    alt.Chart(df).mark_point()
    .encode(
        x="Pclass:O",
        y='Age:Q',
        color=alt.Color("Survived:N", scale=color_scale),
        tooltip = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked' ]
    )
)
st.altair_chart(chart1, use_container_width=True)

chart2 = (
    alt.Chart(df).mark_point()
    .encode(
        x="Class Sex:N",
        y='Age:Q',
        color=alt.Color("Survived:N", scale=color_scale),
        tooltip = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked' ]
    )
)
st.altair_chart(chart2, use_container_width=True)

chart3 = (
    alt.Chart(df).mark_bar()
    .encode(
        x=alt.X('Age:Q', bin=alt.Bin(step=3), title='Age'),
        y='count()',
        color=alt.Color("Survived:N", scale=color_scale),
        tooltip = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked' ]
    )
)
st.altair_chart(chart3, use_container_width=True)






