
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

    elif mode == 'bootstrap_ef_1_internet_connection':
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
    if mode == 'ef_1_internet':
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
            'ef_1_reg': 'Ensino Fundamental I',
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

def get_train_test_validation(df):
    df = shuffle(df, random_state=3647).reset_index(drop=True)

    training_sample_range = round(len(df.index) * 0.8)
    validation_sample_range = round(len(df.index) * 0.9)

    df_train = df.loc[
        0:training_sample_range
    ].reset_index(drop=True)
    df_train = _get_additional_columns(df_train)
    df_train = _get_regional_dummies(df_train)
    
    df_validation = df.loc[
        training_sample_range:validation_sample_range
    ].reset_index(drop=True)
    df_validation = _get_regional_dummies(df_validation)

    df_test = df.loc[
        validation_sample_range:len(df.index)
    ].reset_index(drop=True)

    return df_train, df_validation, df_test

def _get_additional_columns(df):
    df['0_a_2_mbs_squared'] = df['0_a_2_mbs']**2
    df['0_a_2_mbs_*_num_operadoras'] = df['0_a_2_mbs'] * df['num_operadoras']
    return df

def _get_regional_dummies(df):
    regional_dummies = pd.get_dummies(df['regiao'])
    df['Centro-Oeste'] = regional_dummies['CENTRO-OESTE']
    df['Nordeste'] = regional_dummies['NORDESTE']
    df['Norte'] = regional_dummies['NORTE']
    df['Sudeste'] = regional_dummies['SUDESTE']
    df['Sul'] = regional_dummies['SUL']

    return df

def get_multivariate_regression(df, model, education_level):
    approval_model, ideb_model, saeb_mat_model, saeb_port_model = _get_multivariate_regression_models(df, model, education_level)

    print(approval_model.summary())
    print(ideb_model.summary())
    print(saeb_mat_model.summary())
    print(saeb_port_model.summary())


def _get_multivariate_regression_models(df, model, education_level):
    exog = _get_multivariate_regression_columns(df, model)
    
    approval_model_endog = df['tx_aprov_' + education_level]
    approval_model = sm.OLS(
        endog=approval_model_endog,
        exog=exog,
        missing='drop'
    ).fit()

    ideb_model_endog = df['ideb_ef_' + education_level]
    ideb_model = sm.OLS(
        endog=ideb_model_endog,
        exog=exog,
        missing='drop'
    ).fit()

    saeb_mat_model_endog = df['saeb_mat_' + education_level]
    saeb_mat_model = sm.OLS(
        endog=saeb_mat_model_endog,
        exog=exog,
        missing='drop'
    ).fit()

    saeb_port_model_endog = df['saeb_port_' + education_level]
    saeb_port_model = sm.OLS(
        endog=saeb_port_model_endog,
        exog=exog,
        missing='drop'
    ).fit()

    return approval_model, ideb_model, saeb_mat_model, saeb_port_model

def _get_multivariate_regression_columns(df, model):
    df['intercepto'] = 1
    common_variables = [
        'intercepto',
        'num_operadoras',
        'exp_vida',
        'analfabetismo',
        'tx_pobreza', 
        'pib_per_capita',
        'mortalidade_inf',
        'maternidade_inf',
        'Centro-Oeste',
        'Nordeste',
        'Norte',
        'Sul',
        'Sudeste'
    ]
    
    if model == 'main':
        exp_variables = common_variables + [
            '0_a_2_mbs'
        ]

    elif model == 'interact':
        exp_variables = common_variables + [
            '0_a_2_mbs',
            '0_a_2_mbs_*_num_operadoras'
        ]

    elif model == 'non_linear':
        exp_variables = common_variables + [
            '0_a_2_mbs',
            '0_a_2_mbs_squared'
        ]

    elif model == 'all_speed_ranges':
        exp_variables = common_variables + [
            '0_a_2_mbs',
            '2.5_a_5_mbs',
            '6_a_10_mbs',
            '11_a_15_mbs',
            '15_a_25_mbs',
            'acima_25_mbs'
        ]

    elif model == 'dual_speed_ranges':
        exp_variables = common_variables + [
            '0_a_10_mbs',
            'acima_10_mbs'
        ]
    
    return df[exp_variables]

def get_forecast_plot(df_train, df_validation, education_level):
    title = 'Valores observados e preditos da taxa de aprovação no ' + {
        'ef_1': 'Ensino Fundamental I',
        'ef_2': 'Ensino Fundamental II',
        'em': 'Ensino Médio'
    }[education_level]

    y_pred, y_real = _get_forecast_analysis(
        df_train=df_train,
        df_validation=df_validation,
        education_level=education_level
    )

    plt.scatter(y_pred, y_real)
    plt.title(title)
    plt.xlabel('Valor predito')
    plt.ylabel('Valor observado')
    plt.show()

def _get_forecast_analysis(df_train, df_validation, education_level):
    df_validation = _get_multivariate_regression_columns(
        df=df_validation,
        model='dual_speed_ranges'
    )

    approval_model = _get_multivariate_regression_models(
        df=df_train,
        model='dual_speed_ranges',
        education_level=education_level
    )[0]

    y_pred = approval_model.predict(df_validation)
    y_real = df_validation['tx_aprov_' + education_level][df_validation.index]

    return y_pred, y_real
