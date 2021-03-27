
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.utils import shuffle
import statsmodels.api as sm


def get_univariate_regression(df, mode):
    if mode == 'ef_1_internet':
        endog = df['tx_aprov_ef_1']
    
    elif mode == 'ef_2_internet':
        endog = df['tx_aprov_ef_2']

    elif mode == 'em_internet':
        endog = df['tx_aprov_em']

    return sm.OLS(
        endog=endog,
        exog=sm.add_constant(df['0_a_2_mbs']),
        missing='drop'
    ).fit().summary()

def get_univariate_regression_plot(df, mode):
    if 'reg' in mode:        
        education_level = {
            'ef_1_reg': 'Ensino Fundamental I',
            'ef_2_reg': 'Ensino Fundamental II',
            'em_reg': 'Ensino Médio'
        }[mode]

        _get_regression_reg_plot(
            df=df,
            x='0_a_2_mbs',
            y='tx_aprov_' + mode[:-4],
            title=('Acesso a internet e taxa de aprovação nos anos iniciais do '
                   + education_level),
            xlabel='Número de dispositivos de 0 a 2 Mb/s por habitante',
            ylabel='Taxa de aprovação no ' + education_level
        )

    else:
        _get_regression_resid_plot(
            df=df,
            x='0_a_2_mbs',
            y='tx_aprov_' + mode,
            title='Distribuição dos erros do modelo',
            xlabel='Número de dispositivos de 0 a 2 Mb/s por habitante',
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

    ideb_model_endog = df['ideb_' + education_level]
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

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_real)
    plt.title(title)
    plt.xlabel('Valor predito')
    plt.ylabel('Valor observado')
    plt.xlim(50, 100)
    plt.ylim(50, 100)
    plt.show()

def _get_forecast_analysis(df_train, df_validation, education_level):
    df_validation_ = _get_multivariate_regression_columns(
        df=df_validation,
        model='dual_speed_ranges'
    )

    approval_model = _get_multivariate_regression_models(
        df=df_train,
        model='dual_speed_ranges',
        education_level=education_level
    )[0]

    y_pred = approval_model.predict(df_validation_)
    y_real = df_validation['tx_aprov_' + education_level][df_validation.index]

    return y_pred, y_real
