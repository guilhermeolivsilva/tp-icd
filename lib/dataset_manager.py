
from unidecode import unidecode

import numpy as np
import pandas as pd


def get_working_dataset(presentation=True, datasets_path='datasets/'):
    if presentation:
        return pd.read_parquet(datasets_path + 'dataset.parquet')
    
    else:
        datasets_path += 'raw/'
    
    # Geography
    ibge_dataset = get_ibge_dataset(datasets_path)
    
    # Internet access
    internet_connection_dataset = get_internet_dataset(
        datasets_path,
        ibge_dataset
    )
    
    # Education
    education_dataset = get_education_dataset(datasets_path)
    
    # Social
    social_dataset = get_social_dataset(datasets_path)

    df = education_dataset.merge(ibge_dataset.drop(columns=['cod_ap', 'cod_setor']).drop_duplicates(), on='cod_mun', how='inner')\
                          .merge(social_dataset, on=['nom_mun', 'sig_uf'], how='inner')\
                          .merge(internet_connection_dataset, on=['cod_mun', 'nom_mun', 'sig_uf', 'regiao'], how='inner')

    cols_to_normalize = [x for x in df.columns.tolist() if 'mbs' in x]
    for col in cols_to_normalize:
        df[col] = df[col] / df['populacao']

    df = df.reindex(columns=(
        ['cod_mun', 'nom_mun', 'sig_uf', 'regiao'] + list(
            [x for x in df.columns.tolist() if x not in ['cod_mun', 'nom_mun', 'sig_uf', 'regiao']]
        )
    ))

    return df

def get_ibge_dataset(raw_datasets_path):
    print('Loading and cleaning IBGE data...')
    ibge_dataset = pd.read_parquet(
        raw_datasets_path + 'dicionario_ibge.parquet'
    )
    ibge_dataset['nom_mun'] = ibge_dataset['nom_mun'].apply(
        lambda x: unidecode(x)
    )
    ibge_dataset['regiao'] = ibge_dataset['sig_uf'].apply(get_region_from_uf)

    print('Done')
    return ibge_dataset

def get_region_from_uf(x):
    if x in ['AM', 'RR', 'AP', 'PA', 'TO', 'RO', 'AC']:
        return 'NORTE'
    if x in ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA']:
        return 'NORDESTE'
    if x in ['MT', 'MS', 'GO', 'DF']:
        return 'CENTRO-OESTE'
    if x in ['MG', 'SP', 'RJ', 'ES']:
        return 'SUDESTE'
    if x in ['SC', 'PR', 'RS']:
        return 'SUL'

def get_internet_dataset(raw_datasets_path, ibge_dataset):
    print('Loading and cleaning Internet access data...')
    internet_connection_dataset = pd.read_parquet(
        raw_datasets_path + 'internet_connection.parquet'
    )
    internet_connection_dataset.columns = internet_connection_dataset.columns\
                                                                      .str\
                                                                      .replace(',', '.')
    internet_connection_dataset = internet_connection_dataset.rename(
        columns=dict(
            zip(
                [x for x in internet_connection_dataset.columns if 'tcp_range' in x and '.' not in x],
                [x + str('.0') for x in internet_connection_dataset.columns if 'tcp_range' in x and '.' not in x]
            )
        )
    )

    print('Done')
    return group_internet_data(internet_connection_dataset.merge(
        ibge_dataset,
        how='inner',
        on='cod_setor'
    ))

def sum_tcp_range(df, start, stop):
    col_sum = pd.Series(np.zeros(len(df)))
    
    while start <= stop:
        try:
            col_sum += df['tcp_range_' + str(start)]
        except KeyError:
            pass
        start += 0.5
        
    return col_sum

def group_internet_data(internet_connection_dataset):
    tcp_range_cols = [x for x in internet_connection_dataset.columns.tolist() if 'tcp_range' in x]
    agg_dict = {key: 'sum' for key in tcp_range_cols}
    agg_dict['asn_name'] = 'nunique'

    internet_connection_grp = internet_connection_dataset.groupby([
        'cod_mun',
        'type',
        'nom_mun',
        'sig_uf',
        'regiao'
    ]).agg(agg_dict).reset_index().rename(columns={
        'asn_name': 'num_operadoras'
    })

    internet_connection_grp['0_a_2_mbs'] = sum_tcp_range(internet_connection_grp, 0, 2)
    internet_connection_grp['2.5_a_5_mbs'] = sum_tcp_range(internet_connection_grp, 2.5, 5)
    internet_connection_grp['6_a_10_mbs'] = sum_tcp_range(internet_connection_grp, 6, 10)
    internet_connection_grp['11_a_15_mbs'] = sum_tcp_range(internet_connection_grp, 11, 15)
    internet_connection_grp['15_a_25_mbs'] = sum_tcp_range(internet_connection_grp, 16, 25)
    internet_connection_grp['acima_25_mbs'] = sum_tcp_range(internet_connection_grp, 26, 50)
    internet_connection_grp['0_a_10_mbs'] = sum_tcp_range(internet_connection_grp, 0, 10)
    internet_connection_grp['acima_10_mbs'] = sum_tcp_range(internet_connection_grp, 10, 50)

    internet_connection_grp = internet_connection_grp.drop(
        columns=[x for x in internet_connection_grp if 'tcp_range' in x]
    )

    cols_to_treat = [x for x in internet_connection_grp.columns.tolist() if 'mbs' in x]

    for col in cols_to_treat:   
        internet_connection_grp[col] = internet_connection_grp[col].fillna(0)
        internet_connection_grp[col] = internet_connection_grp[col].astype(float)
        
    internet_connection_grp['num_operadoras'] = internet_connection_grp['num_operadoras'].fillna(0).astype(float)

    return internet_connection_grp


def get_education_dataset(raw_datasets_path):
    print('Loading and cleaning education data...')
    ## IDEB - Ensino Fundamental I
    ideb_ef_1_dataset = pd.read_excel(
        raw_datasets_path + 'divulgacao_anos_iniciais_municipios_2019.xlsx',
        header=[6, 7, 8]
    )
    ideb_ef_1_dataset = rename_and_drop_cols(
        df=ideb_ef_1_dataset,
        cols_to_keep=[
            'Código do Município',
            'Taxa de Aprovação - 2019  5º',
            'Nota SAEB - 2019 Matemática',
            'Nota SAEB - 2019 Língua Portuguesa',
            'IDEB\n2019\n(N x P)'
        ],
        rename_dict={
            'Código do Município': 'cod_mun',
            'Taxa de Aprovação - 2019  5º': 'tx_aprov_ef_1',
            'Nota SAEB - 2019 Matemática': 'saeb_mat_ef_1',
            'Nota SAEB - 2019 Língua Portuguesa': 'saeb_port_ef_1',
            'IDEB\n2019\n(N x P)': 'ideb_ef_1'
        },
        _remove_multilevel_index=True,
        public_only=True
    )

    ## IDEB - Ensino Fundamental II
    ideb_ef_2_dataset = pd.read_excel(
        raw_datasets_path + 'divulgacao_anos_finais_municipios_2019.xlsx',
        header=[6, 7, 8]
    )
    ideb_ef_2_dataset = rename_and_drop_cols(
        df=ideb_ef_2_dataset,
        cols_to_keep=[
            'Código do Município',
            'Taxa de Aprovação - 2019  9º',
            'Nota SAEB - 2019 Matemática',
            'Nota SAEB - 2019 Língua Portuguesa',
            'IDEB\n2019\n(N x P)'
        ],
        rename_dict={
            'Código do Município': 'cod_mun',
            'Taxa de Aprovação - 2019  9º': 'tx_aprov_ef_2',
            'Nota SAEB - 2019 Matemática': 'saeb_mat_ef_2',
            'Nota SAEB - 2019 Língua Portuguesa': 'saeb_port_ef_2',
            'IDEB\n2019\n(N x P)': 'ideb_ef_2'
        },
        _remove_multilevel_index=True,
        public_only=True
    )

    ## IDEB - Ensino Medio
    ideb_em_dataset = pd.read_excel(
        raw_datasets_path + 'divulgacao_ensino_medio_municipios_2019.xlsx',
        header=[6, 7, 8]
    )
    ideb_em_dataset = rename_and_drop_cols(
        df=ideb_em_dataset,
        cols_to_keep=[
            'Código do Município',
            'Taxa de Aprovação - 2019  Total',
            'Nota SAEB - 2019 Matemática',
            'Nota SAEB - 2019 Língua Portuguesa',
            'IDEB\n2019\n(N x P)'
        ],
        rename_dict={
            'Código do Município': 'cod_mun',
            'Taxa de Aprovação - 2019  Total': 'tx_aprov_em',
            'Nota SAEB - 2019 Matemática': 'saeb_mat_em',
            'Nota SAEB - 2019 Língua Portuguesa': 'saeb_port_em',
            'IDEB\n2019\n(N x P)': 'ideb_em'
        },
        _remove_multilevel_index=True,
        public_only=True
    )

    education_dataset = ideb_ef_1_dataset.merge(ideb_ef_2_dataset, on='cod_mun', how='inner')\
                                         .merge(ideb_em_dataset, on='cod_mun', how='inner')

    cols_to_treat = [
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
    
    for col in cols_to_treat:
        education_dataset[col] = education_dataset[col].replace('-', np.nan)
        education_dataset[col] = education_dataset[col].replace('ND', np.nan)
        education_dataset[col] = education_dataset[col].astype(float)

    print('Done')
    return education_dataset

def get_social_dataset(raw_datasets_path):
    print('Loading and cleaning social indicators...')
    
    social_dataset = pd.read_csv(
        raw_datasets_path + 'indicadores_sociais.csv',
        sep=','
    )
    census = pd.read_excel(
        raw_datasets_path + 'pop_censo_2010.xlsx'
    )

    social_dataset = social_dataset.merge(census, on='Territorialidades', how='inner')
    social_dataset = rename_and_drop_cols(
        df=social_dataset,
        cols_to_keep=[
            'Territorialidades',
            'População total 2010',
            'Esperança de vida ao nascer 2010',
            'Taxa de analfabetismo - 15 anos ou mais de idade 2010',
            '% de pobres 2010', 
            'Produto Interno Bruto per capita 2016',
            'Taxa de mortalidade infantil 2017',
            '% de meninas de 10 a 14 anos de idade que tiveram filhos 2017'
        ],
        rename_dict={
            'Territorialidades': 'nom_mun',
            'População total 2010': 'populacao',
            'Esperança de vida ao nascer 2010': 'exp_vida',
            'Taxa de analfabetismo - 15 anos ou mais de idade 2010': 'analfabetismo',
            '% de pobres 2010': 'tx_pobreza',
            'Produto Interno Bruto per capita 2016': 'pib_per_capita',
            'Taxa de mortalidade infantil 2017': 'mortalidade_inf',
            '% de meninas de 10 a 14 anos de idade que tiveram filhos 2017': 'maternidade_inf'
        }
    )

    social_dataset['sig_uf'] = social_dataset['nom_mun'].apply(
        lambda x: str(x)[-3:-1]
    )
    social_dataset['nom_mun'] = social_dataset['nom_mun'].apply(
        lambda x: unidecode(str(x)[:-5].upper())
    )

    cols_to_treat = [
        'populacao',
        'exp_vida',
        'analfabetismo',
        'tx_pobreza',
        'pib_per_capita',
        'mortalidade_inf',
        'maternidade_inf'
    ]

    for col in cols_to_treat:
        social_dataset[col] = social_dataset[col].fillna(np.nan)
        social_dataset[col] = social_dataset[col].astype(float)

    print('Done')
    return social_dataset

def remove_multilevel_index(df):
    for i in range(len(df.columns.values)):
        col = list(df.columns.values[i])

        for j in range(3):
            col[j] = str(col[j])

        for j in range(3):
            if 'Unnamed' in col[j]:
                col[j] = ""

        col = tuple(col)
        df.columns.values[i] = col

    df.columns = [' '.join(col).strip() for col in df.columns.values]
    
    return df

def rename_and_drop_cols(df, cols_to_keep, rename_dict,
                         _remove_multilevel_index=False, public_only=False):
    if _remove_multilevel_index:
        df = remove_multilevel_index(df)
    
    if public_only:
        df = df[df['Rede'] == 'Pública']

    return df[cols_to_keep].rename(columns=rename_dict).reset_index(drop=True)
