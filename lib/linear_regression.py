
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.utils import shuffle
import statsmodels.api as sm


def get_bootstrap(data, n):
    values = []
    for _ in range(n):
        df = data.sample(n=len(data), replace=True)
        values.append(np.mean(df))
    return values

def get_percentile(data, init, final):
    inferior = np.percentile(data, q = init)
    superior = np.percentile(data, q = final)
    return [inferior, superior]

def get_confidence_interval(df):
    n = len(df)
    sample_mean = np.mean(df)
    sample_variance = np.var(df, ddof = 1)
    inferior = sample_mean - 1.96*np.sqrt(sample_variance/n)
    superior = sample_mean + 1.96*np.sqrt(sample_variance/n)
    return [inferior, superior]

def ab_test(df_lower_hyp, df_upper_hyp):
    upper_bound_lower_hyp = get_confidence_interval(df_lower_hyp)[1]
    lower_bound_upper_hyp = get_confidence_interval(df_upper_hyp)[0]
    if lower_bound_upper_hyp > upper_bound_lower_hyp:
        return True
    else:
        return False

def get_approval_rate_comparison(df, teaching_level, regions):
    data_1 = df[df['regiao'] == regions[0].upper()][teaching_level]
    data_2 = df[df['regiao'] == regions[1].upper()][teaching_level]

    if teaching_level == "tx_aprov_ef_1":
        x = "nos anos iniciais do ensino fundamental"
    elif teaching_level == "tx_aprov_ef_2":
        x = "nos anos finais do ensino fundamental"
    else:
        x = "no ensino médio"

    print(f"Existe evidência de que a taxa de aprovação {x} no {regions[1]} é"
          f"maior do que no {regions[0]}?")
    if ab_test(data_1, data_2) == True:
        print("Sim")
    else:
        print("Não")
    
    print(f"Intervalos de confiança:\n {regions[0]}",
          get_confidence_interval(data_1),
          f"{regions[1]}", get_confidence_interval(data_2), "\n")

def get_histogram(df, mode):
    # Income vs approval rate
    if mode == 'dist_per_capita_gdp':
        key='pib_per_capita'
        title='Distribuição do PIB per capita dos municípios brasileiros'
        vline_labels=['PIB per capita médio', 'PIB per capita mediano']
        xlabel='PIB per capita'

    elif mode == 'bootstrap_ef1_gdp':
        keys=['pib_per_capita', 'tx_aprov_ef_1']
        title=('Bootstrap da diferença na taxa de aprovação no Ensino '
                'Fundamental I entre municípios ricos e pobres')

    elif mode == 'bootstrap_ef2_gdp':
        keys=['pib_per_capita', 'tx_aprov_ef_2']
        title=('Bootstrap da diferença na taxa de aprovação no Ensino '
                'Fundamental II entre municípios ricos e pobres')

    elif mode == 'bootstrap_em_gdp':
        keys=['pib_per_capita', 'tx_aprov_em']
        title=('Bootstrap da diferença na taxa de aprovação no Ensino '
                'Médio entre municípios ricos e pobres')

    # Internet connection speed vs approval rate
    elif mode == 'dist_internet_connection':
        title=('Distribuição do acesso à Internet acima de 10 mb/s dos '
                'municípios brasileiros')
        vline_labels=['Número de dispositivos/habitante médio',
                      'Número de dispositivos/habitante mediano']
        xlabel='Número de dispositivos por habitante'

    elif mode == 'bootstrap_ef1_internet_connection':
        keys=['acima_10_mbs', 'tx_aprov_ef_1']
        title=('Bootstrap da diferença na taxa de aprovação no Ensino '
                'Fundamental I entre municípios com bom e fraco acesso à Internet')

    elif mode == 'bootstrap_ef2_internet_connection':        
        keys=['acima_10_mbs', 'tx_aprov_ef_2']
        title=('Bootstrap da diferença na taxa de aprovação no Ensino '
                'Fundamental II entre municípios com bom e fraco acesso à Internet')
    
    elif mode == 'bootstrap_em_internet_connection':
        keys=['acima_10_mbs', 'tx_aprov_em']
        title=('Bootstrap da diferença na taxa de aprovação no Ensino '
                'Médio entre municípios com bom e fraco acesso à Internet')        

    if 'dist' in mode:
        _get_var_distr_histogram(
            df=df,
            key=key,
            title=title,
            vline_labels=vline_labels,
            xlabel=xlabel
        )

    else:
        _get_var_bootstrap_histogram(
            df=df,
            keys=keys,
            title=title
        )

def _get_var_distr_histogram(df, key, title, vline_labels, xlabel):
    outliers = np.percentile(df[key], 99)
    plt.hist(df[key], ec='black', density=True, bins=15, range=(0, outliers))
    plt.title(title)
    plt.axvline(np.mean(df[key]), color='red', label=vline_labels[0])
    plt.axvline(np.median(df[key]), color='black', label=vline_labels[1])
    plt.xlabel(xlabel)
    plt.ylabel('Densidade')
    plt.legend()
    plt.show()

def _get_var_bootstrap_histogram(df, keys, title):
    df = df[keys].copy().dropna()
    median = np.median(df[keys[0]])
    upper_df = df[df[keys[0]] > median].reset_index()[keys[1]]
    lower_df = df[df[keys[0]] <= median].reset_index()[keys[1]][0:-1]
    diff_df = upper_df - lower_df
    bootstrap = get_bootstrap(diff_df, 10000)
    ic_lower, ic_upper = get_percentile(bootstrap, 2.5, 97.5)
    plt.hist(bootstrap, ec='black', density=True, bins=15)
    plt.title(title)
    plt.axvline(ic_upper, color='red', label='Percentil 97.5%')
    plt.axvline(ic_lower, color='green', label='Percentil 2.5%')
    plt.xlabel('Diferença na taxa de aprovação')
    plt.ylabel('Densidade')
    plt.legend()
    plt.show()

def get_univariate_regression(df, mode):
    if mode == 'ef1_internet':
        endog = df['tx_aprov_ef_1']
    
    elif mode == 'ef2_internet':
        endog = df['tx_aprov_ef_2']

    elif mode == 'em_internet':
        endog = df['tx_aprov_em']

    return sm.OLS(
        endog=endog,
        exog=sm.add_constant(df['0_a_2_mbs']),
        missing='drop'
    ).fit().summary()

def get_univariate_regression_plot(df, mode):
    y_key = 'tx_aprov_' + mode[:-4]

    if 'reg' in mode:        
        education_level = {
            'ef1_reg': 'Ensino Fundamental I',
            'ef2_reg': 'Ensino Fundamental II',
            'reg_reg': 'Ensino Médio'
        }[mode]

        _get_regression_reg_plot(
            df=df,
            x='0_a_2_mbs',
            y=y_key,
            title=('Acesso a internet e taxa de aprovação nos anos iniciais do '
                   + education_level),
            xlabel='Número de dispositivos de 0-2 mb/s por habitante',
            ylabel='Taxa de aprovação no ' + education_level
        )

    else:
        _get_regression_resid_plot(
            df=df,
            x='0_a_2_mbs',
            y=y_key,
            title='Distribuição dos erros do modelo',
            xlabel='Número de dispositivos de 0-2 mb/s por habitante',
            ylabel=r'$\epsilon_i$'
        )

def _get_regression_reg_plot(df, x, y, title, xlabel, ylabel):
    sns.regplot(
        x=x,
        y=y,
        data=df,
        n_boot=10000,
        line_kws={
            'color': 'red',
            'lw': 4
        },
        scatter_kws={
            'edgecolor': 'k',
            's': 80,
            'alpha': 0.8
        }
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def _get_regression_resid_plot(df, x, y, title, xlabel, ylabel):
    sns.residplot(
        x=x,
        y=y,
        data=df,
        line_kws={
            'color': 'red',
            'lw': 4
        },
        scatter_kws={
            'edgecolor': 'k',
            's': 80,
            'alpha': 0.8
        },
        dropna=True
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

#### Continuar da seção multivariada (2.2)