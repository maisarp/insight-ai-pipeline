import pandas as pd
import numpy as np
import os
import sys
import io
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configura o encoding para UTF-8 no Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class DataProcessorClustering:
    """
    Processador de dados otimizado para clustering com validação robusta.
    
    Atributos:
        file_path (str): Caminho do arquivo de dados.
        dataframe (pd.DataFrame): Dados carregados.
        required_columns (list): Colunas obrigatórias para análise.
        processed_data (pd.DataFrame): Dados processados para clustering.
        scaler (StandardScaler): Normalizador de dados.
    """
    
    # Colunas obrigatórias para análise
    REQUIRED_COLUMNS = [
        'Gênero',
        'Idade',
        'Estado civil',
        'Bolsa Família',
        'Grau de escolaridade',
        'Situação escolar',
        'Situação de Trabalho',
        'Remuneração',
        'Procurando trabalho',
        'Faz ou fez uso de drogas',
        'Data de Cadastro'
    ]
    
    # Colunas identificadoras esperadas (pelo menos uma deve existir e ter dados)
    IDENTIFIER_COLUMNS = [
        'Nome completo',
        'Nome',
        'CPF',
        'RG',
        'ID',
        'Identificador',
        'Matrícula'
    ]
    
    # Mapeamentos para encoding
    EDUCATION_BASE_MAP = {
        # Sem escolaridade
        '': 0,
        'sem escolaridade': 0,
        'sem escolaridade - nao alfabetizado': 0,
        'educacao infantil - educacao infantil': 0,
        'educacao infantil': 0,
        # Ensino fundamental
        'sem escolaridade - ensino tecnico': 1,
        'educacao de jovens e adultos (eja) - ensino fundamental (eja)': 1,
        'ensino fundamental - 1 ano': 1,
        'ensino fundamental - 3 ano': 1,
        'ensino fundamental - 4 ano': 1,
        'ensino fundamental - 5 ano': 1,
        'ensino fundamental - 6 ano': 1,
        'ensino fundamental - 7 ano': 1,
        'ensino fundamental - 8 ano': 1,
        'ensino fundamental - 9 ano': 1,
        # Ensino médio
        'ensino medio - 1 ano': 2,
        'ensino medio - 2 ano': 2,
        'ensino medio - 3 ano': 2,
        'ensino medio': 2,
        # Ensino superior
        'ensino superior - graduacao': 3,
        'ensino superior': 3
    }
    
    def __init__(self, file_path: str, verbose: bool = True):
        """
        Inicializa o processador.
        
        Args:
            file_path (str): Caminho do arquivo Excel ou CSV.
        """
        self.file_path = file_path
        self.dataframe = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.missing_columns = []
        self.verbose = verbose
        
        # Peso reduzido para a variável de dependência química
        self.feature_weights = {
            'substance_dependency': 0.5,  # Peso reduzido para dependência química
            'age': 1.0,
            'education_level': 1.0,
            'family_benefit': 1.0,
            'studying': 1.0,
            'looking_for_job': 1.0,
            'employment_signal': 0.4,
            'program_duration': 1.0,
            'marital_status_single': 1.0,
            'marital_status_married': 1.0,
            'income': 1.0
        }

    def _print(self, message: str, force: bool = False) -> None:
        """Exibe mensagens condicionadas à configuração de verbosidade."""
        if self.verbose or force:
            print(message)
    
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
    
    def _build_missing_columns_error(self, missing_columns: List[str]) -> str:
        """
        Constrói mensagem de erro formatada para colunas faltantes.
        
        Args:
            missing_columns: Lista de nomes de colunas faltantes.
            
        Returns:
            str: Mensagem de erro formatada.
        """
        all_required = [
            "• Gênero",
            "• Idade", 
            "• Estado civil",
            "• Bolsa Família",
            "• Grau de escolaridade",
            "• Situação escolar",
            "• Situação de Trabalho",
            "• Remuneração",
            "• Procurando trabalho",
            "• Faz ou fez uso de drogas",
            "• Data de Cadastro"
        ]
        
        return (
            f"\n{'='*80}\n"
            f"❌ ERRO: O arquivo não possui todas as colunas necessárias\n"
            f"{'='*80}\n\n"
            f"Coluna(s) faltando no seu arquivo:\n" +
            "\n".join([f"  ✗ {col}" for col in missing_columns]) +
            f"\n\n{'─'*80}\n\n"
            f"📋 COLUNAS OBRIGATÓRIAS (todas devem estar presentes):\n\n" +
            "\n".join([f"  {col}" for col in all_required]) +
            f"\n\n{'─'*80}\n\n"
            f"💡 DICA: Verifique se sua planilha possui EXATAMENTE esses nomes de colunas.\n"
            f"   Os nomes devem ser idênticos, incluindo acentos e espaços.\n\n"
            f"{'='*80}\n"
        )
    
    def load_data(self) -> 'DataProcessorClustering':
        """
        Carrega dados do arquivo.
        
        Returns:
            self: Para encadeamento de métodos.
        
        Raises:
            FileNotFoundError: Se arquivo não existe.
            ValueError: Se formato não suportado.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")
        
        lower = self.file_path.lower()
        
        try:
            if lower.endswith('.csv'):
                # Tenta diferentes encodings e delimitadores para CSV
                encodings = ['utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
                delimiters = [',', ';', '\t']
                self.dataframe = None
                last_error = None
                
                for encoding in encodings:
                    for delimiter in delimiters:
                        try:
                            df = pd.read_csv(
                                self.file_path, 
                                encoding=encoding, 
                                delimiter=delimiter,
                                on_bad_lines='skip'  # Pula linhas problemáticas
                            )
                            # Verifica se o CSV foi lido corretamente (tem mais de 1 coluna)
                            if not df.empty and len(df.columns) > 1:
                                self.dataframe = df
                                self._print(f"[OK] CSV carregado com encoding: {encoding}, delimitador: '{delimiter}'")
                                break
                        except Exception as e:
                            last_error = e
                            continue
                    
                    if self.dataframe is not None:
                        break
                
                if self.dataframe is None:
                    raise RuntimeError(
                        f"Não foi possível ler o CSV com nenhuma combinação de encoding/delimitador. "
                        f"Último erro: {last_error}"
                    )
            
            # Valida se o CSV tem dados significativos
            if not self._has_meaningful_data(self.dataframe):
                raise ValueError(
                    "O arquivo não contém dados válidos para análise. "
                    "Todas as linhas estão vazias ou não possuem informações significativas."
                )
                    
            elif lower.endswith(('.xlsx', '.xls')):
                # Para arquivos Excel, usa engine openpyxl que lida melhor com encoding
                try:
                    self.dataframe = pd.read_excel(self.file_path, engine='openpyxl')
                except Exception:
                    # Fallback para engine padrão
                    self.dataframe = pd.read_excel(self.file_path)
                
                # Valida se o Excel tem dados significativos
                if not self._has_meaningful_data(self.dataframe):
                    raise ValueError(
                        "O arquivo não contém dados válidos para análise. "
                        "Todas as linhas estão vazias ou não possuem informações significativas."
                    )
            else:
                raise ValueError('Formato não suportado. Use .xlsx, .xls ou .csv')
            
            self._print(f"[OK] Dados carregados: {len(self.dataframe)} registros, {len(self.dataframe.columns)} colunas")
            return self
            
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar arquivo: {str(e)}")
    
    def validate_columns(self, strict: bool = False) -> Tuple[bool, List[str]]:
        """
        Valida se as colunas obrigatórias estão presentes.
        
        Args:
            strict (bool): Se True, lança exceção. Se False, apenas avisa.
        
        Returns:
            tuple: (is_valid, missing_columns)
        
        Raises:
            ValueError: Se strict=True e colunas faltando.
        """
        if self.dataframe is None:
            raise RuntimeError("Dados não carregados. Execute load_data() primeiro.")
        
        # Normaliza nomes das colunas do arquivo
        df_columns_normalized = [col.strip().lower() for col in self.dataframe.columns]
        required_normalized = [col.strip().lower() for col in self.REQUIRED_COLUMNS]
        
        # Identifica colunas faltando
        self.missing_columns = []
        for req_col in self.REQUIRED_COLUMNS:
            req_normalized = req_col.strip().lower()
            if req_normalized not in df_columns_normalized:
                self.missing_columns.append(req_col)
        
        if self.missing_columns:
            if strict:
                # Usa método centralizado para construir mensagem de erro
                error_msg = self._build_missing_columns_error(self.missing_columns)
                raise ValueError(error_msg)
            else:
                # Modo não-strict: aviso simples
                self._print(f"\n⚠ Atenção: Algumas colunas obrigatórias estão faltando:", force=True)
                for col in self.missing_columns:
                    self._print(f"  ✗ {col}", force=True)
                self._print("\n💡 O sistema tentará continuar, mas a qualidade da análise pode ser afetada.", force=True)
                self._print("   Recomendamos usar um arquivo com todas as colunas obrigatórias.\n", force=True)
                return False, self.missing_columns

        self._print("✓ Todas as colunas obrigatórias estão presentes")
        return True, []
    
    def validate_identifier_columns(self) -> Tuple[bool, Optional[str]]:
        """
        Valida se existe pelo menos uma coluna identificadora com dados válidos.
        
        Returns:
            tuple: (is_valid, identifier_found or error_message)
        
        Raises:
            ValueError: Se nenhuma coluna identificadora válida for encontrada.
        """
        if self.dataframe is None:
            raise RuntimeError("Dados não carregados. Execute load_data() primeiro.")
        
        # Normaliza nomes das colunas do arquivo
        df_columns_normalized = {col.strip().lower(): col for col in self.dataframe.columns}
        
        # Procura por colunas identificadoras
        found_identifiers = []
        for id_col in self.IDENTIFIER_COLUMNS:
            id_normalized = id_col.strip().lower()
            if id_normalized in df_columns_normalized:
                original_col_name = df_columns_normalized[id_normalized]
                
                # Verifica se a coluna tem pelo menos um valor não-vazio
                has_data = False
                for value in self.dataframe[original_col_name]:
                    if pd.notna(value) and str(value).strip():
                        str_val = str(value).strip().lower()
                        if str_val not in {'nan', 'none', ''}:
                            has_data = True
                            break
                
                if has_data:
                    found_identifiers.append(original_col_name)
        
        if not found_identifiers:
            # Constrói mensagem de erro detalhada
            error_msg = (
                f"\n{'='*80}\n"
                f"❌ ERRO: Nenhuma coluna identificadora válida encontrada\n"
                f"{'='*80}\n\n"
                f"Para realizar a análise, é necessário ter pelo menos UMA coluna\n"
                f"identificadora com dados válidos (não-vazios).\n\n"
                f"{'─'*80}\n\n"
                f"📋 COLUNAS IDENTIFICADORAS ACEITAS (pelo menos uma deve existir):\n\n" +
                "\n".join([f"  • {col}" for col in self.IDENTIFIER_COLUMNS]) +
                f"\n\n{'─'*80}\n\n"
                f"💡 DICAS:\n"
                f"   1. Adicione uma coluna com nome, CPF, RG ou outro identificador\n"
                f"   2. Verifique se a coluna identificadora não está vazia\n"
                f"   3. Os nomes devem ser idênticos aos listados acima\n\n"
                f"{'='*80}\n"
            )
            raise ValueError(error_msg)
        
        self._print(f"✓ Coluna(s) identificadora(s) encontrada(s): {', '.join(found_identifiers)}")
        return True, found_identifiers[0]
    
    def normalize_text(self, text: str) -> str:
        """
        Normaliza texto para matching robusto.
        
        Args:
            text (str): Texto a normalizar.
        
        Returns:
            str: Texto normalizado.
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).strip().lower()
        # Remove acentos comuns
        replacements = {
            'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
            'é': 'e', 'ê': 'e',
            'í': 'i',
            'ó': 'o', 'ô': 'o', 'õ': 'o',
            'ú': 'u', 'ü': 'u',
            'ç': 'c'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def encode_escolaridade(self, value: str) -> int:
        """
        Codifica grau de escolaridade em valor ordinal simplificado.
        
        Args:
            value (str): Grau de escolaridade informado.
        
        Returns:
            int: Código ordinal (0=Sem escolaridade, 1=Fundamental,
                 2=Médio, 3=Superior).
        """
        if pd.isna(value) or not str(value).strip():
            return 0
        
        normalized = self.normalize_text(value)
        if normalized in self.EDUCATION_BASE_MAP:
            return self.EDUCATION_BASE_MAP[normalized]
        
        # Ajuste para valores com símbolos como "º" ou variações de acento
        normalized = normalized.replace('º', '').replace('°', '')
        if normalized in self.EDUCATION_BASE_MAP:
            return self.EDUCATION_BASE_MAP[normalized]
        
        # Busca parcial por palavras-chave
        if 'superior' in normalized or 'graduacao' in normalized or 'graduacao' in normalized:
            return 3
        if 'medio' in normalized:
            return 2
        if 'fundamental' in normalized or 'tecnico' in normalized or 'eja' in normalized:
            return 1

        self._print(f"⚠ Valor não reconhecido na coluna 'Grau de escolaridade': '{value}' (usando valor padrão: sem escolaridade)", force=True)
        return 0
    
    def parse_binary(self, value) -> int:
        """
        Converte valores diversos para binário (0/1).
        
        Args:
            value: Valor a converter.
        
        Returns:
            int: 0 ou 1.
        """
        if pd.isna(value) or not str(value).strip():
            return 0
            
        normalized = str(value).lower().strip()
        normalized = normalized.replace('\xa0', ' ')
        normalized = ' '.join(normalized.split())

        # Mapeia variações de sim/não para 1/0
        if normalized in ['sim', 's', '1', 'true', 'verdadeiro', 'yes', 'y', 'sim - r$ 600,00', 'sim - r$ 300,00']:
            return 1
        elif normalized in ['não', 'nao', 'n', '0', 'false', 'falso', 'no', '']:
            return 0

        if 'sim' in normalized:
            return 1
        if 'não' in normalized or 'nao' in normalized:
            return 0
            
        # Se não for nenhum dos valores conhecidos, assume que é não
        return 0
    
    def parse_remuneracao(self, value) -> float:
        """
        Extrai valor numérico de remuneração.
        
        Args:
            value: Valor a converter.
        
        Returns:
            float: Valor numérico.
        """
        if pd.isna(value) or not str(value).strip():
            return 0.0
            
        # Remove formatação brasileira
        cleaned_text = str(value).strip()
        cleaned_text = cleaned_text.replace('R$', '').replace('r$', '').replace(' ', '')
        cleaned_text = cleaned_text.replace('.', '').replace(',', '.').strip()
        
        try:
            return float(cleaned_text)
        except:
            self._print(f"⚠ Valor não reconhecido na coluna 'Remuneração': '{value}' (usando valor padrão: R$ 0,00)", force=True)
            return 0.0
    
    def prepare_features_for_clustering(self) -> pd.DataFrame:
        """
        Prepara features otimizadas para clustering.
        
        Returns:
            pd.DataFrame: Dados processados e normalizados.
        """
        if self.dataframe is None:
            raise RuntimeError("Dados não carregados. Execute load_data() primeiro.")

        self._print("\n📊 Preparando dados para análise...")
        
        df = self.dataframe.copy()
        features = {}
        
        # Normaliza nomes das colunas
        df.columns = [col.strip() for col in df.columns]
        
        # 1. Idade (numérico)
        if 'Idade' in df.columns:
            # Converte para numérico, tratando valores inválidos
            features['age'] = pd.to_numeric(df['Idade'], errors='coerce').fillna(0)
        
        # 2. Estado civil (one-hot para os principais)
        if 'Estado civil' in df.columns:
            civil_status = df['Estado civil'].apply(
                lambda value: '' if pd.isna(value) else ' '.join(str(value).strip().lower().split())
            )

            single_options = {'solteiro(a)', 'solteiro', 'solteira'}
            married_options = {
                'casado(a)',
                'casado',
                'casada',
                'união estável',
                'uniao estavel',
            }

            features['marital_status_single'] = civil_status.isin(single_options).astype(int)
            features['marital_status_married'] = civil_status.isin(married_options).astype(int)
        
        # 3. Benefício social (binário)
        if 'Bolsa Família' in df.columns:
            features['family_benefit'] = df['Bolsa Família'].apply(self.parse_binary)
        
        # 4. Nível educacional (ordinal)
        if 'Grau de escolaridade' in df.columns:
            features['education_level'] = df['Grau de escolaridade'].apply(self.encode_escolaridade)
        
        # 5. Situação educacional (binário: estudando ou não)
        if 'Situação escolar' in df.columns:
            # Mapeia para valores binários
            education_status_map = {
                'cursando': 1,  # Considera como estudando
                'interrompido': 0,
                'concluído': 0,
                'concluido': 0,
                '': 0  # Considera vazio como não estudando
            }
            
            # Aplica o mapeamento, convertendo para minúsculas e removendo espaços extras
            school_status = df['Situação escolar'].astype(str).str.strip().str.lower().fillna('')
            features['studying'] = school_status.map(lambda x: education_status_map.get(x, 0))

        # 6. Situação de trabalho (escala 0 a 1)
        if 'Situação de Trabalho' in df.columns:
            work_status = df['Situação de Trabalho'].astype(str).str.strip().str.lower().fillna('')
            work_status = work_status.replace({'não informado': '', 'nao informado': ''})

            unemployed_options = {
                'desempregado',
                'desempregado(a)',
                'desempregado (a)',
                'sem emprego',
                'sem trabalho',
                'dona de casa',
                ''
            }

            informal_options = {
                'trabalhador por conta propria (bico, autonomo)',
                'trabalhador por conta propria',
                'trabalhador informal',
                'autonomo',
                'autônomo',
                'empregado sem carteira de trabalho assinada',
                'trabalhador por conta própria',
                'bico'
            }

            formal_options = {
                'empregado com carteira de trabalho assinada',
                'empregado com carteira assinada',
                'empregado clt',
                'clt',
                'aprendiz',
                'estagiario',
                'estagiário'
            }

            def map_employment(value: str) -> float:
                """Mapeia situação de trabalho para escala de 0 a 1."""
                if value in formal_options:
                    return 1.0
                if value in informal_options:
                    return 0.6
                if value in unemployed_options:
                    return 0.0
                return 0.0

            features['employment_signal'] = work_status.map(map_employment).fillna(0.0)

        # 7. Renda (numérico)
        if 'Remuneração' in df.columns:
            features['income'] = df['Remuneração'].apply(self.parse_remuneracao)

        # 8. Busca por emprego (binário)
        if 'Procurando trabalho' in df.columns:
            features['looking_for_job'] = df['Procurando trabalho'].apply(self.parse_binary)

        # 9. Dependência química (binário com terminologia adequada)
        if 'Faz ou fez uso de drogas' in df.columns:
            def parse_substance_dependency(value):
                """
                Identifica se há dependência química com base no uso de substâncias.
                
                Args:
                    value: Valor da coluna 'Faz ou fez uso de drogas'
                    
                Returns:
                    int: 1 se houver evidência de dependência, 0 caso contrário
                """
                if pd.isna(value) or not str(value).strip() or str(value).strip().lower() == 'não possui':
                    return 0
                # Considera como dependente se houver qualquer menção a drogas
                return 1
                
            features['substance_dependency'] = df['Faz ou fez uso de drogas'].astype(str).apply(parse_substance_dependency)

        # 10. Tempo no programa (em meses) - calculado a partir da Data de Cadastro
        if 'Data de Cadastro' in df.columns:
            from datetime import datetime
            
            # Converte para datetime (formato brasileiro: DD/MM/YYYY)
            registration_dates = pd.to_datetime(df['Data de Cadastro'], format='%d/%m/%Y', errors='coerce')
            
            # Calcula diferença em dias
            today = datetime.now()
            days_in_program = (today - registration_dates).dt.days
            
            # Converte para meses (arredonda para 1 casa decimal)
            features['program_duration'] = (days_in_program / 30.0).round(1).fillna(0)

            self._print(
                f"  ✓ Coluna 'Data de Cadastro' processada: tempo mínimo={features['program_duration'].min():.1f} meses, "
                f"tempo máximo={features['program_duration'].max():.1f} meses, "
                f"tempo médio={features['program_duration'].mean():.1f} meses"
            )
        
        # Cria DataFrame
        self.processed_data = pd.DataFrame(features)
        self.feature_names = list(features.keys())

        self._print(f"\n✓ Dados preparados com sucesso!")
        self._print(f"  • Total de registros processados: {len(self.processed_data)}")
        self._print(f"  • Total de características analisadas: {len(self.feature_names)}")

        return self.processed_data
    
    def normalize_features(self, fit: bool = True) -> np.ndarray:
        """
        Normaliza features usando StandardScaler, aplicando pesos específicos.
        
        Args:
            fit (bool): Se True, ajusta o scaler. Se False, apenas transforma.
        
        Returns:
            np.ndarray: Dados normalizados e ponderados.
        """
        if self.processed_data is None:
            raise RuntimeError("Features não preparadas. Execute prepare_features_for_clustering() primeiro.")
        
        # Valida features esperadas vs presentes (apenas quando NÃO está ajustando)
        if not fit and hasattr(self.scaler, 'feature_names_in_'):
            expected_features = set(self.scaler.feature_names_in_)
            current_features = set(self.processed_data.columns)
            
            missing = expected_features - current_features
            if missing:
                # Mapeia nomes técnicos para nomes de colunas
                feature_map = {
                    'age': 'Idade',
                    'marital_status_single': 'Estado civil',
                    'marital_status_married': 'Estado civil',
                    'family_benefit': 'Bolsa Família',
                    'education_level': 'Grau de escolaridade',
                    'studying': 'Situação escolar',
                    'employment_signal': 'Situação de Trabalho',
                    'income': 'Remuneração',
                    'looking_for_job': 'Procurando trabalho',
                    'substance_dependency': 'Faz ou fez uso de drogas',
                    'program_duration': 'Data de Cadastro'
                }
                
                missing_readable = []
                for feat in missing:
                    readable_name = feature_map.get(feat, feat)
                    if readable_name not in missing_readable:  # Evita duplicatas
                        missing_readable.append(readable_name)
                
                # Reutiliza o método centralizado para construir a mensagem
                error_msg = self._build_missing_columns_error(missing_readable)
                raise ValueError(error_msg)
        
        # Aplica pesos às features
        weighted_data = self.processed_data.copy()
        for feature in self.processed_data.columns:
            weight = self.feature_weights.get(feature, 1.0)
            weighted_data[feature] = self.processed_data[feature] * weight
        
        # Aplica a normalização
        if fit:
            normalized = self.scaler.fit_transform(weighted_data)
            self._print("✓ Dados normalizados e ponderados com sucesso (ajuste inicial)")
        else:
            normalized = self.scaler.transform(weighted_data)
            self._print("✓ Dados normalizados usando padrão do treinamento")
        
        return normalized
    
    def get_feature_summary(self) -> Dict:
        """
        Retorna resumo estatístico das features.
        
        Returns:
            dict: Estatísticas descritivas.
        """
        if self.processed_data is None:
            raise RuntimeError("Features não preparadas.")
        
        return {
            'total_registros': len(self.processed_data),
            'total_features': len(self.feature_names),
            'features': self.feature_names,
            'estatisticas': self.processed_data.describe().to_dict(),
            'valores_faltando': self.processed_data.isnull().sum().to_dict()
        }


# ---
# Nota de Transparência e Responsabilidade
#
# Descrição: Este arquivo contém seções de código que foram geradas
#            ou assistidas por IA.
#
# Auditoria: Todo o código foi revisado, testado e validado por
#            uma desenvolvedora humana.
#
# Tag:       @ai_generated
# Dev:       Maisa Pires
# ---
