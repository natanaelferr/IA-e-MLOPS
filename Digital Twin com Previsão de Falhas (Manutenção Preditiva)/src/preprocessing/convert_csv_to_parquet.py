import pandas as pd
from paths import Paths
import os

DATABASE_FD001 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD001_normalized.csv'
DATABASE_FD002 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD002_normalized.csv'
DATABASE_FD003 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD003_normalized.csv'
DATABASE_FD004 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD004_normalized.csv'

ARQUIVO_PARQUET_FD001 = Paths.PROCESSED_DIR / 'train_FD001.parquet'
ARQUIVO_PARQUET_FD002 = Paths.PROCESSED_DIR / 'train_FD002.parquet'
ARQUIVO_PARQUET_FD003 = Paths.PROCESSED_DIR / 'train_FD003.parquet'
ARQUIVO_PARQUET_FD004 = Paths.PROCESSED_DIR / 'train_FD004.parquet'



def convert_csv_to_parquet(fileToConvert, fileConverted):
    # Read desired file
    df = pd.read_csv(fileToConvert, sep=';', header=None, engine="python")

    # Convert to parquet
    df.to_parquet(fileConverted, engine="pyarrow", index=False)
    print(f"Arquivo convertido com sucesso para: {fileConverted}")


def read_the_parquet_file(file):
    df = pd.read_parquet(file)

    for i, row in df.iterrows():
        print(row.tolist())
        if i >= 9:  # imprime apenas as 10 primeiras
          break

def verify_if_file_has_string(file):
    # Detecta o formato de arquivo se csv ou parquet
    #df = pd.read_csv(file, sep=';', header=None, engine="python")
    df = pd.read_parquet(file)

    # Diagnóstico
    print("Colunas detectadas (nomes como inteiros):")
    print(df.columns.tolist())

    # Verifica colunas com pelo menos um valor string
    colunas_com_string = []
    for col in df.columns:
        mask = df[col].apply(lambda x: isinstance(x, str))
        if mask.any():
            exemplos = df[col][mask].unique()[:5]  # até 5 exemplos
            colunas_com_string.append((col, len(df[col][mask]), exemplos))

    # Mostra resultado
    if colunas_com_string:
        print("⚠️ Colunas com pelo menos um valor string:")
        for col, total, exemplos in colunas_com_string:
            print(f" - Coluna {col} | {total} valores string")
            print(f"   Exemplos: {list(exemplos)}")
    else:
        print("✅ Nenhuma coluna contém valores string.")

#convert_csv_to_parquet(DATABASE_FD001, ARQUIVO_PARQUET_FD001)
#read_the_parquet_file(ARQUIVO_PARQUET_FD001)
verify_if_file_has_string(ARQUIVO_PARQUET_FD001)

