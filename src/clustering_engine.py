import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ClusteringEngine:
    """
    Motor de clustering para descobrir perfis de acolhidos.
    
    Atributos:
        model (KMeans): Modelo de clustering treinado.
        n_clusters (int): Número de clusters.
        cluster_profiles (dict): Perfis interpretados de cada cluster.
        feature_names (list): Nomes das features usadas.
    """
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Inicializa o motor de clustering.
        
        Args:
            n_clusters (int): Número de clusters desejado.
            random_state (int): Seed para reprodutibilidade.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        self.cluster_profiles = {}
        self.feature_names = []
        self.cluster_labels = {}
        self.is_trained = False
    
    def find_optimal_clusters(self, X: np.ndarray, max_k: int = 6) -> Dict:
        """
        Encontra número ótimo de clusters usando método do cotovelo.
        
        Args:
            X (np.ndarray): Dados normalizados.
            max_k (int): Número máximo de clusters a testar.
        
        Returns:
            dict: Métricas para cada k testado.
        """
        print("\nTestando diferentes números de clusters...")
        
        results = {
            'k_values': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': []
        }
        
        for k in range(2, min(max_k + 1, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            results['k_values'].append(k)
            results['inertia'].append(kmeans.inertia_)
            
            # Silhouette score (maior é melhor, -1 a 1)
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X, labels)
                results['silhouette'].append(sil_score)
            else:
                results['silhouette'].append(0)
            
            # Davies-Bouldin score (menor é melhor)
            if len(np.unique(labels)) > 1:
                db_score = davies_bouldin_score(X, labels)
                results['davies_bouldin'].append(db_score)
            else:
                results['davies_bouldin'].append(float('inf'))
            
            print(f"  k={k}: Inércia={kmeans.inertia_:.2f}, Silhouette={results['silhouette'][-1]:.3f}")
        
        # Recomendação baseada em silhouette
        best_k_idx = np.argmax(results['silhouette'])
        best_k = results['k_values'][best_k_idx]
        
        print(f"\n✓ Recomendação: k={best_k} (melhor silhouette score)")
        
        return results
    
    def train(self, X: np.ndarray, feature_names: List[str]) -> 'ClusteringEngine':
        """
        Treina o modelo de clustering.
        
        Args:
            X (np.ndarray): Dados normalizados.
            feature_names (list): Nomes das features.
        
        Returns:
            self: Para encadeamento de métodos.
        """
        print(f"\n⏳ Treinando modelo de clustering (k={self.n_clusters})...")
        
        self.feature_names = feature_names
        self.model.fit(X)
        self.is_trained = True
        
        # Calcula métricas
        labels = self.model.labels_
        inertia = self.model.inertia_
        
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            
            print(f"✓ Modelo treinado com sucesso!")
            print(f"  Inércia: {inertia:.2f}")
            print(f"  Silhouette Score: {sil_score:.3f}")
            print(f"  Davies-Bouldin Score: {db_score:.3f}")
        else:
            print(f"✓ Modelo treinado (métricas não disponíveis)")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediz clusters para novos dados.
        
        Args:
            X (np.ndarray): Dados normalizados.
        
        Returns:
            np.ndarray: Labels dos clusters.
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Execute train() primeiro.")
        
        return self.model.predict(X)
    
    def analyze_clusters(self, X: np.ndarray, original_data: pd.DataFrame) -> Dict:
        """
        Analisa e interpreta os clusters formados.
        
        Args:
            X (np.ndarray): Dados normalizados usados no clustering.
            original_data (pd.DataFrame): Dados originais (não normalizados).
        
        Returns:
            dict: Perfis de cada cluster.
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Execute train() primeiro.")
        
        print("\n⏳ Analisando perfis dos clusters...")
        
        labels = self.model.labels_
        profiles = {}
        
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_data = original_data[mask]
            
            profile = {
                'cluster_id': cluster_id,
                'size': int(mask.sum()),
                'percentage': float(mask.sum() / len(labels) * 100),
                'characteristics': {}
            }
            
            # Analisa cada feature
            for col in original_data.columns:
                if col in cluster_data.columns:
                    if cluster_data[col].dtype in ['int64', 'float64']:
                        # Numérica: média
                        profile['characteristics'][col] = {
                            'mean': float(cluster_data[col].mean()),
                            'median': float(cluster_data[col].median()),
                            'std_dev': float(cluster_data[col].std())
                        }
                    else:
                        # Categórica: moda
                        if len(cluster_data[col]) > 0:
                            mode_value = cluster_data[col].mode()
                            if len(mode_value) > 0:
                                profile['characteristics'][col] = {
                                    'most_common': str(mode_value[0]),
                                    'frequency': float((cluster_data[col] == mode_value[0]).sum() / len(cluster_data) * 100)
                                }
            
            profiles[cluster_id] = profile
            
            print(f"\n📊 Cluster {cluster_id}:")
            print(f"  Tamanho: {profile['size']} ({profile['percentage']:.1f}%)")
            
            # Mostra características principais
            if 'idade' in profile['characteristics']:
                print(f"  Idade média: {profile['characteristics']['idade']['mean']:.1f} anos")
            if 'remuneracao' in profile['characteristics']:
                print(f"  Renda média: R$ {profile['characteristics']['remuneracao']['mean']:.2f}")
            if 'situacao_trabalho' in profile['characteristics']:
                print(f"  Sit. Trabalho média: {profile['characteristics']['situacao_trabalho']['mean']:.2f}")
        
        self.cluster_profiles = profiles
        return profiles
    
    def interpret_clusters(self, profiles: Dict) -> Dict[int, str]:
        """
        Interpreta clusters e atribui labels semânticos.
        
        Args:
            profiles (dict): Perfis dos clusters.
        
        Returns:
            dict: Mapeamento cluster_id -> label.
        """
        print("\n⏳ Interpretando clusters...")
        
        # Calcula scores para cada cluster
        cluster_scores = {}

        for cluster_id, profile in profiles.items():
            score = 0.0
            chars = profile['characteristics']

            # Nível educacional (0=sem, 1=fundamental, 2=médio, 3=superior)
            if 'education_level' in chars:
                education = chars['education_level']['mean']
                if education >= 2.5:
                    score += 1.5
                elif education >= 1.5:
                    score += 1.0
                elif education > 0:
                    score += 0.5

            # Renda declarada
            if 'income' in chars:
                income_mean = chars['income']['mean']
                if income_mean >= 1200:
                    score += 2.0
                elif income_mean > 600:
                    score += 1.0

            # Busca ativa por trabalho
            if 'looking_for_job' in chars:
                looking_mean = chars['looking_for_job']['mean']
                if looking_mean >= 0.5:
                    score += 2.0
                elif looking_mean >= 0.2:
                    score += 1.0

            # Estudo em andamento
            if 'studying' in chars:
                studying_mean = chars['studying']['mean']
                if studying_mean >= 0.2:
                    score += 1.0

            # Sinal de emprego (0 desempregado, 0.6 informal, 1 formal)
            if 'employment_signal' in chars:
                employment_mean = chars['employment_signal']['mean']
                if employment_mean >= 0.8:
                    score += 1.0
                elif employment_mean >= 0.4:
                    score += 0.5

            # Tempo de vínculo com o programa (meses)
            if 'program_duration' in chars:
                duration_mean = chars['program_duration']['mean']
                if duration_mean >= 18:
                    score += 1.0
                elif duration_mean >= 12:
                    score += 0.5

            # Benefício social elevado pode indicar maior vulnerabilidade
            if 'family_benefit' in chars:
                benefit_mean = chars['family_benefit']['mean']
                if benefit_mean >= 0.7:
                    score -= 0.5

            # Dependência química (quanto menor, melhor)
            if 'substance_dependency' in chars:
                substance_mean = chars['substance_dependency']['mean']
                if substance_mean <= 0.4:
                    score += 1.0
                elif substance_mean >= 0.9:
                    score -= 1.0

            cluster_scores[cluster_id] = score
        
        # Ordena clusters por score
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Atribui labels com interpretação gradual
        labels = {}
        if len(sorted_clusters) >= 3:
            labels[sorted_clusters[0][0]] = "Possível Liberação"
            labels[sorted_clusters[1][0]] = "Monitoramento Ativo"
            labels[sorted_clusters[2][0]] = "Apoio Intensivo"
            for cluster_id, _ in sorted_clusters[3:]:
                labels[cluster_id] = "Apoio Intensivo"
        elif len(sorted_clusters) == 2:
            labels[sorted_clusters[0][0]] = "Possível Liberação"
            labels[sorted_clusters[1][0]] = "Apoio Intensivo"
        else:
            labels[sorted_clusters[0][0]] = "Perfil Único"
        
        self.cluster_labels = labels
        
        print("\n✓ Interpretação dos clusters:")
        for cluster_id, label in labels.items():
            score = cluster_scores[cluster_id]
            size = profiles[cluster_id]['size']
            pct = profiles[cluster_id]['percentage']
            chars = profiles[cluster_id]['characteristics']

            highlights = []
            education_mean = chars.get('education_level', {}).get('mean')
            if education_mean is not None:
                highlights.append(f"educação média={education_mean:.2f}")

            income_mean = chars.get('income', {}).get('mean')
            if income_mean is not None:
                highlights.append(f"renda média=R$ {income_mean:.2f}")

            looking_mean = chars.get('looking_for_job', {}).get('mean')
            if looking_mean is not None:
                highlights.append(f"busca emprego={looking_mean*100:.0f}%")

            employment_mean = chars.get('employment_signal', {}).get('mean')
            if employment_mean is not None:
                highlights.append(f"emprego sinal={employment_mean:.2f}")

            substance_mean = chars.get('substance_dependency', {}).get('mean')
            if substance_mean is not None:
                highlights.append(f"dependência={substance_mean*100:.0f}%")

            duration_mean = chars.get('program_duration', {}).get('mean')
            if duration_mean is not None:
                highlights.append(f"tempo programa={duration_mean:.1f}m")

            highlight_text = ", ".join(highlights[:4])
            print(f"  Cluster {cluster_id} → {label} (score: {score:.1f}, n={size}, {pct:.1f}%) | {highlight_text}")
        
        return labels
    
    def get_cluster_label(self, cluster_id: int) -> str:
        """
        Retorna o label semântico de um cluster.
        
        Args:
            cluster_id (int): ID do cluster.
        
        Returns:
            str: Label interpretado.
        """
        return self.cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
    
    def calculate_confidence(self, X: np.ndarray, cluster_id: int) -> float:
        """
        Calcula confiança da classificação baseado na distância ao centróide.
        
        Args:
            X (np.ndarray): Dados do registro (1 linha).
            cluster_id (int): ID do cluster atribuído.
        
        Returns:
            float: Confiança (0-100%).
        """
        if not self.is_trained:
            return 50.0
        
        # Distância ao centróide do cluster atribuído
        centroid = self.model.cluster_centers_[cluster_id]
        dist_to_assigned = np.linalg.norm(X - centroid)
        
        # Distâncias a todos os centróides
        all_distances = [np.linalg.norm(X - c) for c in self.model.cluster_centers_]
        
        # Confiança: quanto mais próximo do seu cluster e longe dos outros, maior
        min_dist = min(all_distances)
        max_dist = max(all_distances)
        
        if max_dist == min_dist:
            return 50.0
        
        # Normaliza: distância pequena = confiança alta
        confidence = (1 - (dist_to_assigned - min_dist) / (max_dist - min_dist)) * 100
        
        # Garante entre 30% e 95%
        confidence = max(30.0, min(95.0, confidence))
        
        return round(confidence, 1)
    
    def save_model(self, output_path: str = 'model/clustering_model.joblib') -> str:
        """
        Salva o modelo treinado.
        
        Args:
            output_path (str): Caminho de saída.
        
        Returns:
            str: Caminho do arquivo salvo.
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não treinado. Execute train() primeiro.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'n_clusters': self.n_clusters,
            'cluster_profiles': self.cluster_profiles,
            'cluster_labels': self.cluster_labels,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, output_path)
        print(f"✓ Modelo salvo em: {output_path}")
        
        return output_path
    
    @staticmethod
    def load_model(model_path: str, verbose: bool = True) -> 'ClusteringEngine':
        """
        Carrega modelo salvo.

        Args:
            model_path (str): Caminho do modelo.
            verbose (bool): Define se mensagens informativas devem ser exibidas.

        Returns:
            ClusteringEngine: Instância com modelo carregado.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        model_data = joblib.load(model_path)
        
        engine = ClusteringEngine(n_clusters=model_data['n_clusters'])
        engine.model = model_data['model']
        engine.cluster_profiles = model_data['cluster_profiles']
        engine.cluster_labels = model_data['cluster_labels']
        engine.feature_names = model_data['feature_names']
        engine.is_trained = True

        if verbose:
            print(f"✓ Modelo carregado de: {model_path}")
            print(f"  Clusters: {engine.n_clusters}")
            print(f"  Labels: {list(engine.cluster_labels.values())}")
        
        return engine


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
