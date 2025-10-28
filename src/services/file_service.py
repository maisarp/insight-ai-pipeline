from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np


class InputFileService:
    """Gerencia validação e conversão do arquivo fornecido pelo usuário."""

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

    def __init__(self, conversion_dir: Optional[Path] = None) -> None:
        """Inicializa o serviço definindo o diretório de conversão padrão."""
        self.conversion_dir = conversion_dir
        if self.conversion_dir:
            self.conversion_dir.mkdir(parents=True, exist_ok=True)

    def normalize_path(self, raw_path: str) -> Path:
        """Normaliza o caminho informado, expandindo variáveis e diacríticos."""
        expanded = raw_path.strip().strip('"').strip("'")
        expanded = expanded.replace("\\", "/")
        path = Path(expanded).expanduser()
        return path

    def validate_input(self, path: Path) -> Tuple[bool, str]:
        """Valida existência e extensão suportada do arquivo."""
        if not path.exists():
            return False, "Arquivo não encontrado. Verifique o caminho informado."

        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            return False, "Formato não suportado. Use arquivos .csv, .xlsx ou .xls."

        if path.stat().st_size == 0:
            return False, "Arquivo vazio. Forneça um arquivo contendo registros."

        return True, ""
    
    def _has_meaningful_data(self, dataframe: pd.DataFrame) -> bool:
        """
        Verifica se o dataframe contém dados significativos.
        
        Args:
            dataframe (pd.DataFrame): DataFrame a ser validado.
        
        Returns:
            bool: True se há pelo menos uma célula com dado significativo.
        """
        if dataframe.empty:
            return False
        
        # Remove linhas completamente vazias
        df_no_empty_rows = dataframe.dropna(how='all')
        
        if df_no_empty_rows.empty:
            return False
        
        # Verifica se existe pelo menos uma célula com dado não-vazio e não-nan
        for col in df_no_empty_rows.columns:
            for value in df_no_empty_rows[col]:
                # Ignora NaN/None
                if pd.isna(value):
                    continue
                # Converte para string e verifica se não está vazio
                str_value = str(value).strip()
                if str_value and str_value.lower() not in {'nan', 'none', ''}:
                    return True
        
        return False


    def ensure_csv(self, path: Path) -> Tuple[Path, bool]:
        """Garante que o arquivo esteja em CSV, convertendo quando necessário."""
        extension = path.suffix.lower()
        if extension == ".csv":
            return path, False

        # Tenta ler Excel com tratamento de encoding robusto
        try:
            dataframe = pd.read_excel(path, engine='openpyxl')
        except Exception:
            try:
                # Fallback para engine padrão
                dataframe = pd.read_excel(path)
            except Exception as e:
                raise ValueError(f"Não foi possível ler o arquivo Excel: {str(e)}")
        
        # Valida se o arquivo tem dados significativos
        if not self._has_meaningful_data(dataframe):
            raise ValueError(
                "O arquivo não contém dados válidos para análise. "
                "Todas as linhas estão vazias ou não possuem informações significativas."
            )

        target_dir = self.conversion_dir or path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{path.stem}_convertido.csv"
        target_path = target_dir / base_name

        suffix_index = 1
        while target_path.exists():
            target_path = target_dir / f"{path.stem}_convertido_{suffix_index}.csv"
            suffix_index += 1

        dataframe.to_csv(target_path, index=False, encoding="utf-8-sig")
        return target_path, True
