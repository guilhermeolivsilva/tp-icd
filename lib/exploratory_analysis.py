
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_dataset_description(df, dataset):
    if dataset == 'education':
        cols = _get_education_cols()
    elif dataset == 'internet':
        cols = _get_internet_cols(df)
    elif dataset == 'social':
        cols = _get_social_cols()
    
    display(df[cols].drop_duplicates(subset=cols).describe())

def _get_education_cols():
    return [
        'tx_aprov_ef_1',
        'saeb_mat_ef_1',
        'saeb_port_ef_1',
        'ideb_ef_1',
        'tx_aprov_ef_2',
        'saeb_mat_ef_2',
        'saeb_port_ef_2',
        'ideb_ef_2',
        'tx_aprov_em',
        'saeb_mat_em',
        'saeb_port_em',
        'ideb_em'
    ]

def _get_internet_cols(df):
    return [x for x in df.columns.tolist() if 'mbs' in x and x not in ['0_a_10_mbs', 'acima_10_mbs']]

def _get_social_cols():
    return [
        'populacao',
        'exp_vida',
        'analfabetismo',
        'tx_pobreza',
        'pib_per_capita',
        'mortalidade_inf',
        'maternidade_inf'
    ]

def get_education_plot(df, mode):
    education_cols = _get_education_cols()

    if mode == 'ideb':
        _get_education_histogram(
            df=df.drop_duplicates(subset=education_cols),
            key='ideb',
            title=u'IDEB por nível de educação',
            xlabel=u'IDEB',
            bins=[
                np.linspace(0, 10, 21),
                np.linspace(0, 10, 21),
                np.linspace(0, 10, 21)
            ]
        )

        _get_education_histogram_decile(
            df=df.drop_duplicates(subset=education_cols),
            key='ideb',
            decil=0.9,
            title=u'Escolas no decil superior (>= 90%) do IDEB',
            xlabel=u'Escolas por região'
        )

        _get_education_histogram_decile(
            df=df.drop_duplicates(subset=education_cols),
            key='ideb',
            decil=-0.1,
            title=u'Escolas no decil inferior (<= 10%) do IDEB',
            xlabel=u'Escolas por região'
        )

    elif mode == 'approval_rate':
        _get_education_histogram(
            df=df.drop_duplicates(subset=education_cols),
            key='tx_aprov',
            title=u'Distribuição das taxas de aprovação por nível de educação',
            xlabel=u'Taxa de aprovação'
        )
    
    elif mode == 'saeb_mat':
        _get_education_histogram(
            df=df.drop_duplicates(subset=education_cols),
            key='saeb_mat',
            title=u'Rendimento em matemática',
            xlabel=[
                u'Notas (125-350)',
                u'Notas (200-425)',
                u'Notas (225-475)'
            ],
            bins=[
                np.linspace(125, 350, 21),
                np.linspace(200, 425, 21),
                np.linspace(225, 475, 21)
            ]
        )

        _get_education_histogram_decile(
            df=df.drop_duplicates(subset=education_cols),
            key='saeb_mat',
            decil=0.9,
            title=u'Escolas no decil superior (>= 90%) do SAEB para Matemática',
            xlabel=u'Escolas por região'
        )

        _get_education_histogram_decile(
            df=df.drop_duplicates(subset=education_cols),
            key='saeb_mat',
            decil=-0.1,
            title=u'Escolas no decil inferior (<= 10%) do SAEB para Matemática',
            xlabel=u'Escolas por região'
        )

    elif mode == 'saeb_port':
        _get_education_histogram(
            df=df.drop_duplicates(subset=education_cols),
            key='saeb_port',
            title=u'Rendimento em língua portuguesa',
            xlabel=[
                u'Notas (0-350)',
                u'Notas (200-400)',
                u'Notas (225-425)'
            ],
            bins=[
                np.linspace(0, 350, 21),
                np.linspace(200, 400, 21),
                np.linspace(225, 425, 21)
            ]
        )

        _get_education_histogram_decile(
            df=df.drop_duplicates(subset=education_cols),
            key='saeb_port',
            decil=0.9,
            title=u'Escolas no decil superior (>= 90%) do SAEB para Língua Portuguesa',
            xlabel=u'Escolas por região'
        )

        _get_education_histogram_decile(
            df=df.drop_duplicates(subset=education_cols),
            key='saeb_port',
            decil=-0.1,
            title=u'Escolas no decil inferior (<= 10%) do SAEB para Língua Portuguesa',
            xlabel=u'Escolas por região'
        )


def _get_education_histogram(df, key, title, xlabel, bins=None):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    
    if bins is None:
        bins = []
        for i in range(3):
            bins.append(np.linspace(60, 100, 11))

    ax[0].hist(df[key + '_ef_1'], edgecolor='black', bins=bins[0])
    ax[0].axvline(df[key + '_ef_1'].mean(), color='red', linestyle='--')
    ax[0].axvline(df[key + '_ef_1'].median(), color='black', linestyle='--')
    ax[0].legend(['Média', 'Mediana'])
    ax[0].set_title(u'Ensino Fundamental I')

    ax[1].hist(df[key + '_ef_2'], edgecolor='black', bins=bins[1])
    ax[1].axvline(df[key + '_ef_2'].mean(), color='red', linestyle='--')
    ax[1].axvline(df[key + '_ef_2'].median(), color='black', linestyle='--')
    ax[1].legend(['Média', 'Mediana'])
    ax[1].set_title(u'Ensino Fundamental II')

    ax[2].hist(df[key + '_em'], edgecolor='black', bins=bins[2])
    ax[2].axvline(df[key + '_em'].mean(), color='red', linestyle='--')
    ax[2].axvline(df[key + '_em'].median(), color='black', linestyle='--')
    ax[2].legend(['Média', 'Mediana'])
    ax[2].set_title(u'Ensino Médio')
    
    if type(xlabel) is list:
        for i in range(3):
            ax[i].set_xlabel(xlabel[i])
            ax[i].set_ylabel(u'Frequência')
    else:
        for i in range(3):
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(u'Frequência')
        
    fig.suptitle(title, fontsize=14)
    plt.show()

def _get_education_histogram_decile(df, key, decil, title, xlabel):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    
    if decil > 0:
        df_ef_1 = df[df[key + '_ef_1'] >= df[key + '_ef_1'].quantile(decil)]
        df_ef_1['regiao'].value_counts().plot(kind='bar', ax=ax[0])
        ax[0].set_title(u'Ensino Fundamental I')

        df_ef_2 = df[df[key + '_ef_2'] >= df[key + '_ef_2'].quantile(decil)]
        df_ef_2['regiao'].value_counts().plot(kind='bar', ax=ax[1])
        ax[1].set_title(u'Ensino Fundamental II')

        df_em = df[df[key + '_em'] >= df[key + '_em'].quantile(decil)]
        df_em['regiao'].value_counts().plot(kind='bar', ax=ax[2])
        ax[2].set_title(u'Ensino Médio')
        
    else:
        df_ef_1 = df[df[key + '_ef_1'] <= df[key + '_ef_1'].quantile(-1*decil)]
        df_ef_1['regiao'].value_counts().plot(kind='bar', ax=ax[0])
        ax[0].set_title(u'Ensino Fundamental I')

        df_ef_2 = df[df[key + '_ef_2'] <= df[key + '_ef_2'].quantile(-1*decil)]
        df_ef_2['regiao'].value_counts().plot(kind='bar', ax=ax[1])
        ax[1].set_title(u'Ensino Fundamental II')

        df_em = df[df[key + '_em'] <= df[key + '_em'].quantile(-1*decil)]
        df_em['regiao'].value_counts().plot(kind='bar', ax=ax[2])
        ax[2].set_title(u'Ensino Médio')
    
    if type(xlabel) is list:
        for i in range(3):
            ax[i].set_xlabel(xlabel[i])
            ax[i].set_ylabel(u'Frequência')
    else:
        for i in range(3):
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(u'Frequência')
        
    fig.suptitle(title, fontsize=14)
    plt.show()

def get_internet_plot(df, mode):
    if mode == 'device_distribution':
        _get_internet_histogram_devices(
            df=df,
            title='Média municipal de dispositivos/habitante por faixa de velocidade'
        )

    elif mode == 'isp_distribution':
        _get_internet_histogram_isp(
            df=df,
            title=u'Média municipal de prestadoras de serviço de Internet por região'
        )

def _get_internet_histogram_devices(df, title, xlabel=u'Região', ylabel=u'Disp./hab.'):
    fig, ax = plt.subplots(5, figsize=(17, 25), constrained_layout=True, sharey=True)
    
    internet_cols = _get_internet_cols(df)
    internet_labels = _get_internet_labels()
    regions = _get_region_labels()
    ind = np.arange(6)
    width = 0.4
    
    i = 0
    for region in regions:
        ax[i].bar(
            ind,
            df[(df['regiao']==region) & (df['type']=='broadband')][internet_cols].mean(),
            width,
            label=u'Internet fixa'
        )
        ax[i].bar(
            ind + width,
            df[(df['regiao']==region) & (df['type']=='mobile')][internet_cols].mean(),
            width,
            label=u'Internet móvel'
        )
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].set_title(region)
        ax[i].set_xticks(ind + width/2)
        ax[i].set_xticklabels(internet_labels)
        ax[i].legend()
        i += 1
    
    fig.suptitle(title, fontsize=14)
    plt.show()

def _get_internet_labels():
    return [
        '0 a 2 Mb/s',
        '2.5 a 5 Mb/s',
        '6 a 10 Mb/s',
        '11 a 15 Mb/s',
        '15 a 25 Mb/s',
        '25+ Mb/s'
    ]

def _get_region_labels():
    return ['CENTRO-OESTE', 'NORDESTE', 'NORTE', 'SUDESTE', 'SUL']

def _get_internet_histogram_isp(df, title, xlabel=u'Região', ylabel=u'Operadoras únicas'):
    fig, ax = plt.subplots(figsize=(17, 5))
    
    regions = _get_region_labels()
    
    ax.bar(regions, df.groupby('regiao')['num_operadoras'].mean())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.title(title)
    plt.show()

def get_social_plot(df, mode):
    social_cols = _get_social_cols()

    if mode == 'population':
        _get_social_histogram(
            df=df[social_cols + ['regiao']].drop_duplicates(),
            key='populacao',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Dezenas de milhões de invidíduos',
            title=u'População por região do Brasil, em dezenas de milhões de pessoas',
            regional=True,
            agg_method='sum'
        )

    elif mode == 'life_expectancy':
        _get_social_histogram(
            df=df[social_cols].drop_duplicates(),
            key='exp_vida',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Observações',
            title=u'Expectativa de vida ao nascer, por faixa de idade'
        )
        _get_social_histogram(
            df=df[social_cols + ['regiao']].drop_duplicates(),
            key='exp_vida',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Anos',
            title=u'Expectativa de vida ao nascer, por região',
            regional=True,
            agg_method=lambda x: np.average(x, weights=df.loc[x.index, 'populacao'])
        )

    elif mode == 'analfabetism':
        _get_social_histogram(
            df=df[social_cols].drop_duplicates(),
            key='analfabetismo',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Observações',
            title=u'Taxa de analfabetismo, por faixas'
        )
        _get_social_histogram(
            df=df[social_cols + ['regiao']].drop_duplicates(),
            key='analfabetismo',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Taxa (%)',
            title=u'Taxa média de analfabetismo por região, ponderada pela população',
            regional=True,
            agg_method=lambda x: np.average(x, weights=df.loc[x.index, 'populacao'])
        )

    elif mode == 'poverty':
        _get_social_histogram(
            df=df[social_cols].drop_duplicates(),
            key='tx_pobreza',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Observações',
            title=u'Taxa de pobreza, por faixas'
        )
        _get_social_histogram(
            df=df[social_cols + ['regiao']].drop_duplicates(),
            key='tx_pobreza',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Taxa (%)',
            title=u'Taxa média de pobreza por região, ponderada pela população',
            regional=True,
            agg_method=lambda x: np.average(x, weights=df.loc[x.index, 'populacao'])
        )

    elif mode == 'gdp_per_capita':
        _get_social_histogram(
            df=df[social_cols].drop_duplicates(),
            key='pib_per_capita',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Observações',
            title=u'PIB per capita, por faixas de renda',
            bins=np.linspace(0, 200, 50)
        )

        _get_social_histogram(
            df=df[social_cols + ['regiao']].drop_duplicates(),
            key='pib_per_capita',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Milhares de BRL',
            title=u'PIB per capita médio por região, ponderado pela população',
            regional=True,
            agg_method=lambda x: np.average(x, weights=df.loc[x.index, 'populacao'])
        )

    elif mode == 'child_mortality':
        _get_social_histogram(
            df=df[social_cols].drop_duplicates(),
            key='mortalidade_inf',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Observações',
            title=u'Mortalidade infantil, por faixas'
        )

        _get_social_histogram(
            df=df[social_cols + ['regiao']].drop_duplicates(),
            key='mortalidade_inf',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Média ponderada de casos/1000 hab.',
            title=u'Taxa média de mortalidade infantil por região, ponderada pela população',
            regional=True,
            agg_method=lambda x: np.average(x, weights=df.loc[x.index, 'populacao'])
        )

    elif mode == 'adolescent_pregnancy':
        _get_social_histogram(
            df=df[social_cols].drop_duplicates(),
            key='maternidade_inf',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Observações',
            title=u'Maternidade infantil, por faixas'
        )

        _get_social_histogram(
            df=df[social_cols + ['regiao']].drop_duplicates(),
            key='maternidade_inf',
            xlabel=u'Nota: são considerados apenas municípios com observações na tabela internet_connection',
            ylabel=u'Média ponderada de casos/1000 hab.',
            title=u'Taxa média de maternidade infantil por região, ponderada pela população',
            regional=True,
            agg_method=lambda x: np.average(x, weights=df.loc[x.index, 'populacao'])
        )


def _get_social_histogram(df, key, title, xlabel, ylabel=u'Frequência',
                            legend=None, regional=False, agg_method=None, bins=None):

    df = df.dropna()
    fig, ax = plt.subplots(figsize=(15, 5))
    
    if not regional:
        if bins is not None:
            ax.hist(df[key], edgecolor='black', bins=bins)
        else:
            ax.hist(df[key], edgecolor='black')
        ax.axvline(df[key].mean(), color='red', linestyle='--')
        ax.axvline(df[key].median(), color='black', linestyle='--')
        ax.legend(['Média', 'Mediana'])
    else:
        df.groupby('regiao').agg({
            key: agg_method
        }).plot.bar(ax=ax)
        if legend:
            ax.legend([legend])
        else:
            ax.get_legend().remove()
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
        
    plt.title(title, fontsize=14)
    plt.show()

def get_internet_education_corr(df):
    cols_corr = _get_internet_cols(df)\
                 + ['num_operadoras']\
                 + [x for x in _get_education_cols() if '_em' in x]
    
    display(df[cols_corr].corr())

    sns.pairplot(
        df[cols_corr],
        diag_kws={'edgecolor':'k'},
        plot_kws={'alpha':0.5, 'edgecolor':'k'}
    )
    plt.show()
