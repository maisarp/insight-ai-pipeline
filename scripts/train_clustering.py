"""
Script para treinar modelo de clustering com dados históricos.

Uso:
    python scripts/train_clustering.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
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


def is_numeric(value: object) -> bool:
    """
    Indica se o valor informado representa um número válido.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return not np.isnan(numeric)


def build_release_notes(stats: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Cria lista com os principais motivos para o cluster de possível liberação.
    """
    notes: List[str] = []

    income_mean = stats.get('income', {}).get('mean')
    if is_numeric(income_mean):
        notes.append(
            f"Renda média declarada de R$ {float(income_mean):.2f}, indicando alguma ocupação"
        )

    looking_mean = stats.get('looking_for_job', {}).get('mean')
    studying_mean = stats.get('studying', {}).get('mean')
    if is_numeric(looking_mean) and is_numeric(studying_mean):
        notes.append(
            f"{format_percentage(float(looking_mean))}% buscando emprego e {format_percentage(float(studying_mean))}% estudando (sinais de engajamento)"
        )

    employment_signal = stats.get('employment_signal', {}).get('mean')
    if is_numeric(employment_signal):
        notes.append(
            f"Sinal de emprego médio {float(employment_signal):.2f} (0=desempregado, 0.6=informal, 1=formal)"
        )

    duration_mean = stats.get('program_duration', {}).get('mean')
    if is_numeric(duration_mean):
        notes.append(
            f"Tempo médio no programa de {float(duration_mean):.1f} meses"
        )

    education_mean = stats.get('education_level', {}).get('mean')
    if is_numeric(education_mean):
        notes.append(
            f"Escolaridade média {float(education_mean):.2f}, acima do grupo de Apoio Intensivo"
        )

    if not notes:
        notes.append("Indicadores específicos desse cluster não foram identificados durante o treinamento.")

    return notes


def save_training_report(
    metadata: Dict[str, object],
    cluster_summaries: List[Dict[str, object]],
    release_notes: List[str],
) -> None:
    """
    Gera arquivo com resumo do treinamento utilizado em tempo de análise.
    """
    report_path = Path("data") / "training_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "  TRAINING RESULTS ABOVE - USE FOR REFERENCE TO UNDERSTAND THE MODEL",
        "=" * 70,
        "",
        "📊 Resumo:",
        f"  Registros treinados: {metadata.get('total_records', 'N/D')}",
        f"  Features utilizadas: {len(metadata.get('feature_names', []))}",
        f"  Número de clusters: {metadata.get('selected_k', 'N/D')}",
        f"  Modelo salvo em: {metadata.get('model_path', 'N/D')}",
        f"  Scaler salvo em: {metadata.get('scaler_path', 'N/D')}",
        "",
        "🎯 Perfis descobertos:",
    ]

    for summary in cluster_summaries:
        label = summary.get('label', 'Cluster')
        size_value = summary.get('size')
        pct_value = summary.get('percentage')

        if is_numeric(size_value):
            size_text = f"{int(float(size_value))}"
        else:
            size_text = "N/D"

        if is_numeric(pct_value):
            pct_text = f"{float(pct_value):.1f}%"
        else:
            pct_text = "N/D"

        lines.append(f"  {label}: {size_text} pessoas ({pct_text})")

    if release_notes:
        lines.append("")
        lines.append("📝 Motivos da classificação 'Possível Liberação':")
        for note in release_notes:
            lines.append(f"  • {note}")

    lines.append("")
    lines.append("=" * 70)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Relatório de treinamento salvo em: {report_path}")


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
        
        # Valida coluna identificadora
        processor.validate_identifier_columns()
        
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
        cluster_summaries: List[Dict[str, object]] = []
        release_notes: List[str] = []

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

            size_value = int(size) if is_numeric(size) else None
            pct_value = float(pct) if is_numeric(pct) else None
            cluster_summaries.append(
                {
                    "label": labels.get(cluster_id, f"Cluster {cluster_id}"),
                    "size": size_value,
                    "percentage": pct_value,
                }
            )

            if labels.get(cluster_id) == "Possível Liberação":
                release_notes = build_release_notes(stats)
        
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
            "feature_names": processor.feature_names,
            "model_path": model_output,
            "scaler_path": scaler_path,
        }
        with open('model/clustering_metadata.json', 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        print("✓ Metadados salvos em: model/clustering_metadata.json")

        print("\n🎯 Perfis descobertos:")
        for summary in cluster_summaries:
            label = summary.get("label", "Cluster")
            size_value = summary.get("size")
            pct_value = summary.get("percentage")
            if size_value is None or pct_value is None:
                print(f"  {label}: dados de tamanho indisponíveis")
            else:
                print(f"  {label}: {size_value} pessoas ({pct_value:.1f}%)")

        if release_notes:
            print("\n📝 Motivos da classificação 'Possível Liberação':")
            for note in release_notes:
                print(f"  • {note}")

        save_training_report(metadata, cluster_summaries, release_notes)

        print("\n" + "="*70)
        
        # 7. Resumo final
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        
        print(f"\n� Resumo:")
        print(f"  Registros treinados: {len(features_df)}")
        print(f"  Features utilizadas: {len(processor.feature_names)}")
        print(f"  Número de clusters: {engine.n_clusters}")
        print(f"  Modelo salvo em: {model_output}")
        print(f"  Scaler salvo em: {scaler_path}")

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
