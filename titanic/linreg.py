import altair as alt
import streamlit as st
import polars as pl
import datetime as dt
import numpy as np
import random
import statsmodels.api as sm

# Polynomial practice
x = [ e/10 for e in range(-100,100)]
Y_m = [10+x_n*5+x_n*x_n*.5+random.uniform(-20,20) for x_n in x]
X = np.array([[x**p for x in x] for p in range(0,5)]).T

#Statsmodels regression
model = sm.OLS(Y_m, X)
results = model.fit()
st.write(results.summary())
coefficients = results.params
coefficients_vector = np.array(coefficients)
Y_smfit = X @ coefficients_vector

# Manual regression
beta = np.linalg.inv(X.T @ X) @ X.T @ Y_m
Y_manualfit = X @ beta

df_linreg = pl.DataFrame({'x': x, 'Y_m': Y_m, 'Y_manualfit': Y_manualfit, 'Y_smfit':Y_smfit})

#Plotting
scatter = (
    alt.Chart(df_linreg).mark_point()
    .encode(
        x='x:Q',
        y='Y_m:Q'
    )
)

line1 = (
    alt.Chart(df_linreg).mark_line()
    .encode(
       x='x:Q',
       y='Y_manualfit:Q',
       color=alt.value('green')
    )
)

line2 = (
    alt.Chart(df_linreg).mark_line(strokeDash=[5, 5])
    .encode(
       x='x:Q',
       y='Y_smfit:Q',
       color=alt.value('red')
    )
)

chart4 = scatter  + line1 + line2
st.altair_chart(chart4, use_container_width=True)
st.write(beta)