import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def ols_lR(df, x, y='ln_kappa', log=False):
    df_clean = df[[x, y]].dropna()

    # Extract data
    x_clean = df_clean[x]
    y_clean = df_clean[y]

    if log==True:
        df_clean = df_clean[df_clean[x] != 0]
        df_clean[f'log_{x}'] = np.log(x_clean)
        x_clean = df_clean[f'log_{x}']

    # Fit the model
    x_clean = sm.add_constant(x_clean)
    model = sm.OLS(y_clean, x_clean).fit()
    print(model.summary())

    return model, df_clean, x_clean, y_clean


def plot_regression_results_with_coefficients(model, df_clean, x_clean, y_clean, y='ln_kappa'):
    
    y_pred = model.predict(x_clean)
    fig, axs = plt.subplots(2, 3, figsize=(15*0.66, 8*0.66), gridspec_kw={'height_ratios': [1, 4], 'width_ratios': [1, 4, 2]})
    axs = axs.flatten()
    
    axs[0].axis('off')
    axs[2].axis('off')
    axs[1].hist(df_clean[x_clean.columns[1]], alpha=0.5, color='purple')
    axs[1].set_title(f'Distribucion de {x_clean.columns[1]}', color='purple')
    axs[1].set_xlabel(x_clean.columns[1])  
    axs[1].set_ylabel('Frecuencia')  
    
    axs[3].hist(y_clean, alpha=0.5, orientation='horizontal', color='darkgreen')
    axs[3].invert_xaxis()
    axs[3].set_title(f'Distribucion de {y}', color='darkgreen')
    axs[3].set_xlabel('Frecuencia')  
    
    axs[4].scatter(df_clean[x_clean.columns[1]], y_clean, label="Datos", color='blue')
    axs[4].plot(df_clean[x_clean.columns[1]], y_pred, color='red', label="Linea de regresion")
    axs[4].set_title(f'{x_clean.columns[1]} vs {y}')
    axs[4].set_xlabel(x_clean.columns[1])
    axs[4].set_ylabel(y)
    axs[4].legend()
    coefs = model.params[1:]  # Skip intercept
    ci_lower = model.conf_int().loc[x_clean.columns[1], 0] 
    ci_upper = model.conf_int().loc[x_clean.columns[1], 1]  
    ci_error = [[coefs.values - ci_lower], [ci_upper - coefs.values]]  # Error bars for the CI

    # Reshape ci_error to be (2, n) format
    ci_error = np.array(ci_error).reshape(2, -1)
    
    bars = axs[5].bar(coefs.index, coefs.values, yerr=ci_error, capsize=5, color='orange')
    axs[5].set_title('Coeficiente con intervalo de confianza')

    p_values = model.pvalues[1:]  
    for bar, coef, p_value in zip(bars, coefs, p_values):
        height = bar.get_height()
        axs[5].text(bar.get_x() + bar.get_width()/2.0 + 0.2, height, f'p={p_value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
