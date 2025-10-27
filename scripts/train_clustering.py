"""
Script para treinar modelo de clustering com dados históricos.

Uso:
    python scripts/train_clustering.py
"""

import json
import os
import sys
from datetime import datetime

import numpy as np

# Adiciona src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processor_clustering import DataProcessorClustering
from src.clustering_engine import ClusteringEngine
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


def format_percentage(value: float) -> float:
    """
    Converte valor proporcional em percentual com uma casa decimal.
    """
    return round(value * 100, 1)


def describe_education_level(mean_value: float) -> str:
    """
    Retorna interpretação aproximada do nível educacional médio.
    """
    if mean_value >= 2.5:
        return "Predomínio de Ensino Médio ou Superior"
    if mean_value >= 1.5:
        return "Predomínio de Ensino Fundamental completo"
    if mean_value > 0:
        return "Predomínio de Ensino Fundamental inicial"
    return "Sem escolaridade informada"


def main():
    """
    Fluxo principal de treinamento.
    """
    print("="*70)
    print("TREINAMENTO DE MODELO DE CLUSTERING")
    print("="*70)
    
    # 1. Configuração
    default_data_path = os.path.join('data')
    data_path = os.getenv('DATABASE_PATH', default_data_path)
    model_output = 'model/clustering_model.joblib'

    print(f"\n📁 Arquivo de dados: {data_path}")

    if not os.path.exists(data_path):
        print(f"❌ Erro: Arquivo não encontrado: {data_path}")
        print("ℹ Ajuste a variável DATABASE_PATH no .env ou coloque o arquivo em data/<nome_do_arquivo>.xlsx")
        return
    
    try:
        # 2. Carrega e processa dados
        print("\n" + "="*70)
        print("ETAPA 1: PROCESSAMENTO DE DADOS")
        print("="*70)
        
        processor = DataProcessorClustering(data_path)
        processor.load_data()
        
        # Valida colunas (não strict para permitir continuar)
        processor.validate_columns(strict=False)
        
        # Prepara features
        features_df = processor.prepare_features_for_clustering()
        
        # Normaliza
        features_normalized = processor.normalize_features(fit=True)
        
        print(f"\n✓ Dados processados: {len(features_df)} registros, {len(features_df.columns)} features")
        
        # 3. Encontra número ótimo de clusters
        print("\n" + "="*70)
        print("ETAPA 2: ANÁLISE DE CLUSTERS")
        print("="*70)

        engine = ClusteringEngine(n_clusters=3)

        print("\nTestando diferentes números de clusters...")
        results = engine.find_optimal_clusters(features_normalized, max_k=5)

        best_k = 2
        best_score = 0.0
        if results['silhouette']:
            best_index = int(np.argmax(results['silhouette']))
            best_k = results['k_values'][best_index]
            best_score = results['silhouette'][best_index]

            if best_score < 0.15:
                print("⚠ Silhouette muito baixo; definindo k=2 para garantir estabilidade.")
                best_k = 2
        else:
            print("⚠ Não foi possível calcular silhouette; usando k=2.")

        print(f"\n✓ k selecionado automaticamente: {best_k} (silhouette={best_score:.3f})")

        engine = ClusteringEngine(n_clusters=best_k)
        
        # 4. Treina modelo
        print("\n" + "="*70)
        print("ETAPA 3: TREINAMENTO")
        print("="*70)
        
        engine.train(features_normalized, processor.feature_names)
        
        # 5. Analisa e interpreta clusters
        print("\n" + "="*70)
        print("ETAPA 4: INTERPRETAÇÃO")
        print("="*70)
        
        profiles = engine.analyze_clusters(features_normalized, features_df)
        labels = engine.interpret_clusters(profiles)

        # Resumo analítico inspirado no fluxo de desempregados
        print("\n🔍 Resumo analítico por cluster:")
        for cluster_id, profile in profiles.items():
            stats = profile.get('characteristics', {})
            size = profile.get('size', 0)
            pct = profile.get('percentage', 0)
            age_mean = stats.get('age', {}).get('mean', 0)
            education_mean = stats.get('education_level', {}).get('mean', 0)
            benefit_mean = stats.get('family_benefit', {}).get('mean', 0)
            studying_mean = stats.get('studying', {}).get('mean', 0)
            looking_mean = stats.get('looking_for_job', {}).get('mean', 0)
            substance_mean = stats.get('substance_dependency', {}).get('mean', 0)
            income_mean = stats.get('income', {}).get('mean', 0)
            duration_mean = stats.get('program_duration', {}).get('mean', 0)

            print(f"\n[Cluster {cluster_id}] {labels.get(cluster_id, f'Cluster {cluster_id}')}" )
            print(f"   Tamanho: {size} ({pct:.1f}%)")
            print(f"   Idade média: {age_mean:.1f} anos")
            print(f"   Escolaridade média: {education_mean:.2f} → {describe_education_level(education_mean)}")
            print(f"   Benefício social: {format_percentage(benefit_mean)}% recebem Bolsa Família")
            print(f"   Estudando atualmente: {format_percentage(studying_mean)}%")
            print(f"   Buscando emprego: {format_percentage(looking_mean)}%")
            print(f"   Dependência química: {format_percentage(substance_mean)}% informaram uso")
            print(f"   Renda média declarada: R$ {income_mean:.2f}")
            print(f"   Tempo médio no programa: {duration_mean:.1f} meses")
        
        # 6. Salva modelo
        print("\n" + "="*70)
        print("ETAPA 5: SALVAMENTO")
        print("="*70)
        
        import joblib
        scaler_path = 'model/scaler.joblib'
        os.makedirs('model', exist_ok=True)
        joblib.dump(processor.scaler, scaler_path)
        print(f"✓ Scaler salvo em: {scaler_path}")

        engine.save_model(model_output)

        metadata = {
            "training_datetime": datetime.now().isoformat(),
            "source_file": data_path,
            "total_records": len(features_df),
            "selected_k": int(best_k),
            "silhouette_score": float(best_score),
            "feature_names": processor.feature_names
        }
        with open('model/clustering_metadata.json', 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        print("✓ Metadados salvos em: model/clustering_metadata.json")
        
        # 7. Resumo final
        print("\n" + "="*70)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
        print(f"\n📊 Resumo:")
        print(f"  Registros treinados: {len(features_df)}")
        print(f"  Features utilizadas: {len(processor.feature_names)}")
        print(f"  Número de clusters: {engine.n_clusters}")
        print(f"  Modelo salvo em: {model_output}")
        print(f"  Scaler salvo em: {scaler_path}")
        
        print(f"\n🎯 Perfis descobertos:")
        for cluster_id, label in labels.items():
            profile = profiles[cluster_id]
            size = profile.get('size', profile.get('tamanho'))
            percentage = profile.get('percentage', profile.get('percentual'))
            if size is None or percentage is None:
                print(f"  {label}: dados de tamanho indisponíveis")
            else:
                print(f"  {label}: {size} pessoas ({percentage:.1f}%)")

        liberation_cluster_id = None
        for cluster_id, label in labels.items():
            if label == "Possível Liberação":
                liberation_cluster_id = cluster_id
                break

        if liberation_cluster_id is not None:
            liberation_profile = profiles[liberation_cluster_id]
            stats = liberation_profile.get('characteristics', {})

            income_mean = stats.get('income', {}).get('mean', 0)
            looking_pct = format_percentage(stats.get('looking_for_job', {}).get('mean', 0))
            studying_pct = format_percentage(stats.get('studying', {}).get('mean', 0))
            employment_signal = stats.get('employment_signal', {}).get('mean', 0)
            duration_mean = stats.get('program_duration', {}).get('mean', 0)
            education_mean = stats.get('education_level', {}).get('mean', 0)

            print("\n📝 Motivos da classificação 'Possível Liberação':")
            print(f"  • Renda média declarada de R$ {income_mean:.2f}, indicando alguma ocupação")
            print(f"  • {looking_pct}% buscando emprego e {studying_pct}% estudando (sinais de engajamento)")
            print(f"  • Sinal de emprego médio {employment_signal:.2f} (0=desempregado, 0.6=informal, 1=formal)")
            print(f"  • Tempo médio no programa de {duration_mean:.1f} meses")
            print(f"  • Escolaridade média {education_mean:.2f}, acima do grupo de Apoio Intensivo")

            print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌❌❌ Erro durante treinamento: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()


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
