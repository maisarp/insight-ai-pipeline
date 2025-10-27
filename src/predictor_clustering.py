import json
import pandas as pd
import joblib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from src.clustering_engine import ClusteringEngine
from src.data_processor_clustering import DataProcessorClustering


class PredictorClustering:
    """
    Preditor baseado em clustering para classificar prontidão de acolhidos.
    
    Atributos:
        engine (ClusteringEngine): Motor de clustering carregado.
        processor (DataProcessorClustering): Processador de dados.
        scaler: Normalizador carregado.
        predictions (list): Resultados das análises.
        id_data (pd.DataFrame): Dados identificadores preservados.
    """
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        id_columns: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Inicializa o preditor.
        
        Args:
            model_path (str): Caminho do modelo de clustering.
            scaler_path (str): Caminho do scaler.
            id_columns (list): Colunas identificadoras a preservar.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler não encontrado: {scaler_path}")

        self.verbose = verbose
        self.engine = ClusteringEngine.load_model(model_path, verbose=verbose)
        self.scaler = joblib.load(scaler_path)
        self.id_columns = id_columns or []
        self.id_data = None
        self.processor = None
        self.predictions = []
        self.training_metadata = self._load_training_metadata(model_path)

        self._print(f"✓ Scaler carregado de: {scaler_path}")
        if self.id_columns:
            self._print(f"✓ Colunas identificadoras: {', '.join(self.id_columns)}")

    def _print(self, message: str, force: bool = False) -> None:
        """Exibe mensagem condicionalmente conforme configuração de verbosidade."""
        if self.verbose or force:
            print(message)
    
    def load_and_process_data(self, data_path: str) -> 'PredictorClustering':
        """
        Carrega e processa novos dados para análise.
        
        Args:
            data_path (str): Caminho do arquivo com novos dados.
        
        Returns:
            self: Para encadeamento de métodos.
        """
        self._print("\n⏳ Carregando dados para análise...")
        
        # Carrega dados originais
        if data_path.endswith('.csv'):
            original_data = pd.read_csv(data_path, encoding='utf-8-sig')
        elif data_path.endswith(('.xlsx', '.xls')):
            original_data = pd.read_excel(data_path)
        else:
            raise ValueError('Formato não suportado. Use .xlsx, .xls ou .csv')
        
        # Preserva identificadores
        available_ids: List[str] = []
        if self.id_columns:
            available_ids = [col for col in self.id_columns if col in original_data.columns]
            if available_ids:
                self.id_data = original_data[available_ids].copy()
                self._print(f"✓ Identificadores preservados: {', '.join(available_ids)}")
            else:
                self._print("⚠ Nenhuma coluna identificadora encontrada", force=True)
                self.id_data = None
        else:
            self.id_data = None
        
        # Processa dados
        self.processor = DataProcessorClustering(data_path, verbose=self.verbose)
        self.processor.load_data()
        self.processor.validate_columns(strict=False)
        self.processor.prepare_features_for_clustering()
        
        # Normaliza usando scaler treinado
        self.processor.scaler = self.scaler
        self.processor.normalize_features(fit=False)
        
        return self
    
    def _load_training_metadata(self, model_path: str) -> Dict[str, object]:
        """Carrega metadados do treinamento quando disponíveis."""
        metadata_path = Path(model_path).with_name('clustering_metadata.json')
        if not metadata_path.exists():
            return {}

        try:
            with metadata_path.open('r', encoding='utf-8') as handler:
                payload = json.load(handler)
            if isinstance(payload, dict):
                return payload
        except Exception as exc:  # pragma: no cover - falhas não devem interromper fluxo
            self._print(f"⚠ Não foi possível carregar os metadados do treinamento: {exc}", force=True)

        return {}

    def get_training_metadata(self) -> Dict[str, object]:
        """Retorna os metadados conhecidos sobre o treinamento do modelo."""
        return dict(self.training_metadata)

    def predict(self) -> 'PredictorClustering':
        """
        Realiza análises nos dados carregados.
        
        Returns:
            self: Para encadeamento de métodos.
        """
        if self.processor is None or self.processor.processed_data is None:
            raise RuntimeError("Dados não carregados. Execute load_and_process_data() primeiro.")

        self._print("\n⏳ Realizando análise...")
        
        # Normaliza dados
        features_normalized = self.processor.scaler.transform(self.processor.processed_data)
        
        # Prediz clusters
        cluster_ids = self.engine.predict(features_normalized)
        
        # Gera análises detalhadas
        self.predictions = []
        for idx, cluster_id in enumerate(cluster_ids):
            # Label semântico
            label = self.engine.get_cluster_label(cluster_id)
            
            # Confiança
            confidence = self.engine.calculate_confidence(
                features_normalized[idx].reshape(1, -1),
                cluster_id
            )
            
            # Nível de risco baseado no label
            if label == "Apto":
                risk = "Baixo"
            elif label == "Acompanhamento":
                risk = "Médio"
            else:
                risk = "Alto"
            
            prediction = {
                'cluster_id': int(cluster_id),
                'classification': label,
                'confidence': confidence,
                'risk_level': risk
            }
            
            self.predictions.append(prediction)
        
        self._print(f"✓ Análise concluída: {len(self.predictions)} caso(s) analisado(s)")
        
        # Resumo
        from collections import Counter
        class_counts = Counter([p['classification'] for p in self.predictions])
        self._print("\n📊 Resumo:")
        for label, count in class_counts.items():
            pct = count / len(self.predictions) * 100
            self._print(f"  {label}: {count} ({pct:.1f}%)")
        
        return self
    
    def generate_report(self) -> pd.DataFrame:
        """
        Gera DataFrame com relatório completo.
        
        Returns:
            pd.DataFrame: Relatório formatado.
        """
        if not self.predictions:
            raise RuntimeError("Análises não realizadas. Execute predict() primeiro.")
        
        # Cria DataFrame base
        report = pd.DataFrame({
            'ID': range(1, len(self.predictions) + 1)
        })
        
        # Adiciona identificadores
        if self.id_data is not None:
            for col in self.id_data.columns:
                report[col] = self.id_data[col].values
        
        # Adiciona análises
        report['Classificação'] = [p['classification'] for p in self.predictions]
        report['Confiança'] = [f"{p['confidence']:.1f}%" for p in self.predictions]
        report['Nível_de_Risco'] = [p['risk_level'] for p in self.predictions]
        report['Cluster_ID'] = [p['cluster_id'] for p in self.predictions]
        
        return report
    
    def save_predictions_to_txt(self, output_path: str = None) -> str:
        """
        Salva análises em arquivo TXT legível.
        
        Args:
            output_path (str): Caminho de saída.
        
        Returns:
            str: Caminho do arquivo salvo.
        """
        if not self.predictions:
            raise RuntimeError("Análises não realizadas. Execute predict() primeiro.")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"predictions_{timestamp}.txt"
        
        report = self.generate_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE PRONTIDÃO\n")
            f.write("Sistema Inteligente de Apoio à Decisão\n")
            f.write("="*70 + "\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Total de casos analisados: {len(self.predictions)}\n")
            f.write("="*70 + "\n\n")
            
            # Resumo geral
            from collections import Counter
            class_counts = Counter([p['classification'] for p in self.predictions])
            risk_counts = Counter([p['risk_level'] for p in self.predictions])
            
            f.write("RESUMO GERAL\n")
            f.write("-"*70 + "\n")
            
            f.write("\nClassificação:\n")
            for label in ["Apto", "Acompanhamento", "Não Apto"]:
                count = class_counts.get(label, 0)
                pct = count / len(self.predictions) * 100 if self.predictions else 0
                f.write(f"  {label:20s}: {count:4d} ({pct:5.1f}%)\n")
            
            f.write("\nNível de Risco:\n")
            for risk in ["Baixo", "Médio", "Alto"]:
                count = risk_counts.get(risk, 0)
                pct = count / len(self.predictions) * 100 if self.predictions else 0
                f.write(f"  Risco {risk:15s}: {count:4d} ({pct:5.1f}%)\n")
            
            f.write("-"*70 + "\n\n")
            
            # Detalhamento por caso
            f.write("DETALHAMENTO POR CASO\n")
            f.write("="*70 + "\n\n")
            
            for idx, row in report.iterrows():
                f.write(f"Caso #{row['ID']:03d}\n")
                
                # Identificadores
                if self.id_data is not None:
                    for col in self.id_data.columns:
                        if col in row:
                            f.write(f"  {col}: {row[col]}\n")
                
                # Resultados
                f.write(f"  Classificação: {row['Classificação']}\n")
                f.write(f"  Confiança: {row['Confiança']}\n")
                f.write(f"  Nível de Risco: {row['Nível_de_Risco']}\n")
                f.write(f"  Cluster: {row['Cluster_ID']}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("="*70 + "\n")
            
            # Nota técnica
            f.write("\n📌 NOTA TÉCNICA\n")
            f.write("-"*70 + "\n")
            f.write("Este relatório foi gerado por um sistema de Inteligência Artificial\n")
            f.write("baseado em clustering (KMeans). O modelo identifica padrões nos dados\n")
            f.write("e agrupa acolhidos com perfis similares.\n\n")
            f.write("A classificação é uma recomendação e deve ser validada por profissionais\n")
            f.write("qualificados antes de tomar decisões finais.\n")
            f.write("-"*70 + "\n")
        
        self._print(f"✓ Relatório TXT salvo em: {output_path}")
        return output_path
    
    def save_predictions_to_csv(self, output_path: str = None) -> str:
        """
        Salva análises em arquivo CSV.
        
        Args:
            output_path (str): Caminho de saída.
        
        Returns:
            str: Caminho do arquivo salvo.
        """
        if not self.predictions:
            raise RuntimeError("Análises não realizadas. Execute predict() primeiro.")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"predictions_{timestamp}.csv"
        
        report = self.generate_report()
        report.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        self._print(f"✓ Relatório CSV salvo em: {output_path}")
        return output_path
    
    def get_summary_stats(self) -> Dict:
        """
        Retorna estatísticas resumidas para clusters e riscos.
        
        Returns:
            dict: Estatísticas agregadas das análises.
        """
        if not self.predictions:
            raise RuntimeError("Análises não realizadas. Execute predict() primeiro.")

        from collections import Counter

        label_counts = Counter([p['classification'] for p in self.predictions])
        risk_counts = Counter([p['risk_level'] for p in self.predictions])

        total = len(self.predictions)

        cluster_payload = {}
        for label, count in label_counts.items():
            percentage = (count / total * 100) if total else 0.0
            cluster_payload[label] = {
                'count': count,
                'percentage': percentage
            }

        risk_payload = {}
        for label, count in risk_counts.items():
            percentage = (count / total * 100) if total else 0.0
            risk_payload[label] = {
                'count': count,
                'percentage': percentage
            }

        average_confidence = sum([p['confidence'] for p in self.predictions]) / total if total > 0 else 0

        return {
            'total': total,
            'clusters': cluster_payload,
            'risks': risk_payload,
            'average_confidence': average_confidence
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
