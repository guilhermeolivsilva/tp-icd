
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

def print_approval_rate_comparison(df, teaching_level):
    regions = _get_region_labels()
    regions.remove('Sudeste')
    
    for region in regions:
        _get_approval_rate_comparison(df, teaching_level, [region, 'Sudeste'])

def _get_region_labels():
    return ['Centro-Oeste', 'Nordeste', 'Norte', 'Sudeste', 'Sul']

def _get_approval_rate_comparison(df, teaching_level, regions):
    data_1 = df[df['regiao'] == regions[0].upper()][teaching_level]
    data_2 = df[df['regiao'] == regions[1].upper()][teaching_level]

    if teaching_level == "tx_aprov_ef_1":
        x = "nos anos iniciais do ensino fundamental"
    elif teaching_level == "tx_aprov_ef_2":
        x = "nos anos finais do ensino fundamental"
    else:
        x = "no ensino médio"

    if ab_test(data_1, data_2) == True:
        print(f"Existe evidência de que a taxa de aprovação {x} no {regions[1]} é "
          f"maior do que no {regions[0]}.")
    else:
        print(f"Não existe evidência de que a taxa de aprovação {x} no {regions[1]} é "
          f"maior do que no {regions[0]}.")
    
    print(f"Intervalos de confiança:\n {regions[0]}\t\t",
          get_confidence_interval(data_1),
          f"\n {regions[1]}\t\t", get_confidence_interval(data_2), "\n")

def get_histogram(df, mode):
    # Income vs approval rate
    if mode == 'dist_per_capita_gdp':
        key='pib_per_capita'
        title='Distribuição do PIB per capita dos municípios brasileiros'
        vline_labels=['PIB per capita médio', 'PIB per capita mediano']
        xlabel='PIB per capita'

    elif mode == 'bootstrap_ef_1_gdp':
        keys=['pib_per_capita', 'tx_aprov_ef_1']
        title=('Bootstrap da diferença na taxa de aprovação no Ensino '
                'Fundamental I entre municípios ricos e pobres')

    elif mode == 'bootstrap_ef_2_gdp':
        keys=['pib_per_capita', 'tx_aprov_ef_2']
        title=('Bootstrap da diferença na taxa de aprovação no Ensino '
                'Fundamental II entre municípios ricos e pobres')

    elif mode == 'bootstrap_em_gdp':
        keys=['pib_per_capita', 'tx_aprov_em']
        title=('Bootstrap da diferença na taxa de aprovação no Ensino '
                'Médio entre municípios ricos e pobres')

    # Internet connection speed vs approval rate
    elif mode == 'dist_internet_connection':
        key='acima_10_mbs'
        title=('Distribuição do acesso à Internet acima de 10 mb/s dos '
                'municípios brasileiros')
        vline_labels=['Número de dispositivos/habitante médio',
                      'Número de dispositivos/habitante mediano']
        xlabel='Número de dispositivos por habitante'

    elif mode == 'bootstrap_ef_1_internet_connection':
        keys=['acima_10_mbs', 'tx_aprov_ef_1']
        title=('Bootstrap da diferença na taxa de aprovação no Ensino '
                'Fundamental I entre municípios com bom e fraco acesso à Internet')

    elif mode == 'bootstrap_ef_2_internet_connection':        
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