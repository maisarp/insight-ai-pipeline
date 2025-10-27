from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TYPE_CHECKING

from src.services.insight_service import ClusterInsightService

if TYPE_CHECKING:  # pragma: no cover
    from src.predictor_clustering import PredictorClustering


class ClusterReportBuilder:
    """Orquestra a apresentação e persistência dos resultados de clustering."""

    def __init__(self, insight_service: ClusterInsightService, output_dir: Path | None = None) -> None:
        """Define dependências e garante estrutura de saída."""
        self.insight_service = insight_service
        self.output_dir = output_dir or Path.cwd() / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_identifier_value(column: str, value: object) -> str:
        """Formata identificadores para visualização preservando confidencialidade."""
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return ""

        if column.strip().lower() == "cpf":
            digits = re.sub(r"\D", "", text)
            if len(digits) == 11:
                text = f"{digits[:3]}.***.***-{digits[-2:]}"

        return f"{column}: {text}"

    def print_overview(self, predictor: 'PredictorClustering') -> None:
        """Exibe resumo agregado dos resultados no console."""
        stats = predictor.get_summary_stats()
        total = stats["total"]

        print("\n=== Visão Geral da Análise ===")
        print(f"Casos analisados: {total}")
        for label, payload in stats["clusters"].items():
            percentage = payload["percentage"]
            count = payload["count"]
            print(f"- {label}: {count} registros ({percentage:.1f}%)")

        print("\nDistribuição por risco:")
        for risk_label, payload in stats["risks"].items():
            percentage = payload["percentage"]
            count = payload["count"]
            print(f"- {risk_label}: {count} registros ({percentage:.1f}%)")

        confidence = stats["average_confidence"]
        print(f"\nConfiança média estimada: {confidence:.1f}%\n")

    def show_cluster_insights(self, labels: Iterable[str]) -> None:
        """Apresenta insights associados aos clusters observados."""
        printed = set()
        print("=== Insights por Cluster ===")
        for label in labels:
            if label in printed:
                continue
            printed.add(label)
            print(f"\n{label}")
            print(self.insight_service.get_summary(label))
            for bullet in self.insight_service.get_details(label):
                print(f"  • {bullet}")
        print()

    def print_individual_overview(
        self,
        predictor: 'PredictorClustering',
        preferred_identifiers: Sequence[str],
        limit: int = 25
    ) -> None:
        """Lista os registros classificados, respeitando limite para o console."""
        report = predictor.generate_report()
        identifier_columns = [col for col in preferred_identifiers if col in report.columns]
        if not identifier_columns:
            identifier_columns = [col for col in report.columns if col not in {"Classificação", "Confiança", "Nível_de_Risco", "Cluster_ID"}]
            if "ID" not in identifier_columns and "ID" in report.columns:
                identifier_columns.insert(0, "ID")

        print("=== Classificação por Registro ===")
        if report.empty:
            print("Nenhum registro disponível para exibir.")
            return

        for index, row in report.head(limit).iterrows():
            identifiers: List[str] = []
            for column in identifier_columns:
                if column in report.columns:
                    formatted = self._format_identifier_value(column, row[column])
                    if formatted:
                        identifiers.append(formatted)

            identifier_text = " | ".join(identifiers) if identifiers else f"Registro {row['ID']}"
            print(
                f"- {identifier_text} -> {row['Classificação']} | Risco: {row['Nível_de_Risco']} | Confiança: {row['Confiança']}"
            )

            cluster_details = self.insight_service.get_details(row['Classificação'])
            highlight = cluster_details[0] if cluster_details else self.insight_service.get_summary(row['Classificação'])
            print(f"  Observação: {highlight}")

        if len(report) > limit:
            print(f"... {len(report) - limit} registros adicionais foram omitidos na visualização.")
        print()

    def save_outputs(self, predictor: 'PredictorClustering') -> List[Path]:
        """Persiste CSV e resumo textual com timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"cluster_predictions_{timestamp}.csv"
        text_path = self.output_dir / f"cluster_summary_{timestamp}.txt"

        predictor.save_predictions_to_csv(str(csv_path))
        self._write_text_summary(predictor, text_path)

        return [csv_path, text_path]

    def _write_text_summary(self, predictor: 'PredictorClustering', path: Path) -> None:
        """Gera arquivo de texto com visão geral e insights."""
        stats = predictor.get_summary_stats()
        if stats["total"] == 1:
            self._write_single_record_summary(predictor, path)
        else:
            self._write_batch_summary(predictor, path)

    def _write_single_record_summary(self, predictor: 'PredictorClustering', path: Path) -> None:
        """Gera um relatório de texto focado em um único registro."""
        report_row = predictor.generate_report().iloc[0]
        report_dict = self._safe_to_dict(report_row)

        original_series = None
        if getattr(predictor, "processor", None) and getattr(predictor.processor, "dataframe", None) is not None:
            original_series = predictor.processor.dataframe.iloc[0]
        record_dict = self._safe_to_dict(original_series)

        dependency_flag, dependency_text = self._extract_dependency_info(record_dict)
        observed_factors = self._build_observed_factors(record_dict, dependency_flag, dependency_text)
        action_plan = self._build_action_plan(report_dict, dependency_flag, dependency_text, record_dict)
        confidence_text = self._format_confidence(report_dict.get("Confiança"))
        classification_text = self._clean_text(report_dict.get("Classificação"), "Não informado")
        risk_text = self._clean_text(report_dict.get("Nível_de_Risco"), "Não informado")

        with path.open("w", encoding="utf-8") as handler:
            handler.write("Relatório de Análise de Perfil\n")
            handler.write("=" * 60 + "\n\n")

            handler.write("Identificação do Registro:\n")
            if predictor.id_data is not None:
                for column in predictor.id_data.columns:
                    value = report_dict.get(column)
                    text = self._clean_text(value)
                    if text:
                        handler.write(f"  • {column}: {text}\n")
            else:
                identifier = self._clean_text(report_dict.get("ID"), "Não informado")
                handler.write(f"  • ID: {identifier}\n")
            handler.write("\n")

            handler.write("Resultado da Análise:\n")
            handler.write(f"  • Classificação: {classification_text}\n")
            handler.write(f"  • Nível de Risco: {risk_text}\n")
            handler.write(f"  • Confiança da Análise: {confidence_text}\n\n")

            summary = self.insight_service.get_summary(classification_text)
            handler.write(f"O que esta classificação significa?\n  {summary}\n\n")

            details = self.insight_service.get_details(classification_text)
            if details:
                handler.write("Recomendações e Pontos de Atenção:\n")
                for bullet in details:
                    handler.write(f"  • {bullet}\n")
                handler.write("\n")

            if observed_factors:
                handler.write("Fatores observados nos dados fornecidos:\n")
                for item in observed_factors:
                    handler.write(f"  • {item}\n")
                handler.write("\n")

            if action_plan:
                handler.write("Sugestões para a equipe da ONG:\n")
                for item in action_plan:
                    handler.write(f"  • {item}\n")
                handler.write("\n")

            handler.write("---\n")
            handler.write(
                "Nota sobre a análise: Esta análise foi gerada por um modelo de inteligência artificial treinado com dados históricos anonimizados de programas sociais. Os resultados servem como apoio à decisão e precisam ser combinados com a avaliação profissional.\n"
            )
            training_context = self.insight_service.get_training_context(classification_text)
            if training_context:
                handler.write(f"Contexto do modelo: {training_context}\n")
            training_notes = self.insight_service.get_training_notes(classification_text)
            if training_notes:
                handler.write("Indicadores observados no treinamento original:\n")
                for note in training_notes:
                    handler.write(f"  • {note}\n")


    def _write_batch_summary(self, predictor: 'PredictorClustering', path: Path) -> None:
        """Gera arquivo de texto com visão geral para um lote de registros."""
        stats = predictor.get_summary_stats()
        report = predictor.generate_report()
        original_data = None
        if getattr(predictor, "processor", None) and getattr(predictor.processor, "dataframe", None) is not None:
            original_data = predictor.processor.dataframe.reset_index(drop=True)

        with path.open("w", encoding="utf-8") as handler:
            handler.write("Relatório de Clustering\n")
            handler.write("=" * 60 + "\n\n")
            handler.write(f"Total analisado: {stats['total']}\n")
            handler.write(f"Confiança média: {stats['average_confidence']:.1f}%\n\n")

            handler.write("Distribuição por cluster:\n")
            for label, payload in stats["clusters"].items():
                handler.write(
                    f"- {label}: {payload['count']} registros ({payload['percentage']:.1f}%)\n"
                )
            handler.write("\nDistribuição por risco:\n")
            for label, payload in stats["risks"].items():
                handler.write(
                    f"- {label}: {payload['count']} registros ({payload['percentage']:.1f}%)\n"
                )

            handler.write("\nInsights principais:\n")
            for label in stats["clusters"].keys():
                handler.write(f"\n{label}\n")
                handler.write(self.insight_service.get_summary(label) + "\n")
                for bullet in self.insight_service.get_details(label):
                    handler.write(f"  • {bullet}\n")

            handler.write("\nDetalhamento individual (até 10 primeiros registros):\n")
            sample = report.head(10).reset_index(drop=True)
            for idx, row in sample.iterrows():
                handler.write("\n")
                handler.write(f"Registro {row['ID']}\n")
                if predictor.id_data is not None:
                    for column in predictor.id_data.columns:
                        if column in row and str(row[column]).strip().lower() not in {"", "nan"}:
                            handler.write(f"  {column}: {row[column]}\n")

                handler.write(f"  Classificação: {row['Classificação']}\n")
                handler.write(f"  Nível de risco: {row['Nível_de_Risco']}\n")
                handler.write(f"  Confiança estimada: {row['Confiança']}\n")

                if original_data is not None and idx < len(original_data):
                    record = original_data.iloc[idx]
                    handler.write("  Informações observadas no cadastro:\n")
                    for source, label in [
                        ("Idade", "Idade"),
                        ("Situação de Trabalho", "Situação de trabalho"),
                        ("Remuneração", "Renda declarada"),
                        ("Procurando trabalho", "Busca por trabalho"),
                        ("Situação escolar", "Situação escolar"),
                        ("Bolsa Família", "Recebe Bolsa Família"),
                    ]:
                        if source in record:
                            value = record[source]
                            text = str(value).strip()
                            if text and text.lower() != "nan":
                                handler.write(f"    • {label}: {text}\n")

                summary = self.insight_service.get_summary(row['Classificação'])
                handler.write(f"  O que significa essa classificação: {summary}\n")
                details = self.insight_service.get_details(row['Classificação'])
                if details:
                    handler.write("  Recomendações associadas:\n")
                    for bullet in details[:2]:
                        handler.write(f"    • {bullet}\n")

            handler.write("\nDados completos exportados em CSV no mesmo diretório deste relatório.\n")

    def _safe_to_dict(self, data: object) -> Dict[str, Any]:
        """Converte estruturas diversas em dicionário simples quando possível."""
        if data is None:
            return {}
        if isinstance(data, dict):
            return data
        if hasattr(data, "to_dict"):
            try:
                converted = data.to_dict()
                if isinstance(converted, dict):
                    return converted
            except Exception:
                return {}
        if hasattr(data, "items"):
            try:
                return dict(data.items())
            except Exception:
                return {}
        return {}

    def _clean_text(self, value: object, default: str = "") -> str:
        """Normaliza texto removendo valores vazios ou inválidos."""
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        lowered = text.lower()
        if lowered in {"nan", "none"}:
            return default
        return text

    def _format_confidence(self, value: object) -> str:
        """Formata valor de confiança em percentual legível."""
        text = self._clean_text(value)
        if not text:
            return "Não informado"
        normalized = text.replace("%", "").replace(" ", "").replace(",", ".")
        try:
            numeric = float(normalized)
            if numeric <= 1:
                numeric *= 100
            return f"{numeric:.1f}%"
        except (TypeError, ValueError):
            return text

    def _extract_dependency_info(self, record: Dict[str, Any]) -> tuple[bool, str]:
        """Avalia presença de dependência química informada no cadastro."""
        raw_text = self._clean_text(record.get("Faz ou fez uso de drogas"))
        if not raw_text:
            return False, ""

        normalized = raw_text.lower()
        simplified = (
            normalized.replace("ã", "a")
            .replace("õ", "o")
            .replace("â", "a")
            .replace("ê", "e")
            .replace("é", "e")
        )

        if "nao" in simplified and "sim" not in simplified:
            return False, raw_text
        if any(token in simplified for token in {"ausente", "negado", "sem", "não", "nao"}) and "sim" not in simplified:
            return False, raw_text
        return True, raw_text

    def _format_currency(self, value: object) -> str:
        """Formata valores monetários no padrão brasileiro quando possível."""
        text = self._clean_text(value)
        if not text:
            return ""
        cleaned = text.replace("R$", "").replace("r$", "").replace(" ", "")
        normalized = cleaned.replace(".", "").replace(",", ".")
        try:
            amount = float(normalized)
        except ValueError:
            try:
                amount = float(cleaned)
            except ValueError:
                return text
        formatted = f"R$ {amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return formatted

    def _normalize_boolean(self, value: object) -> str:
        """Traduz valores variados em 'Sim', 'Não' ou mantém texto original."""
        text = self._clean_text(value)
        if not text:
            return "Não informado"
        normalized = text.lower()
        simplified = normalized.replace("ã", "a").replace("õ", "o")
        if "sim" in simplified:
            return "Sim"
        if "nao" in simplified or "não" in normalized:
            return "Não"
        return text

    def _build_observed_factors(
        self,
        record: Dict[str, Any],
        dependency_flag: bool,
        dependency_text: str
    ) -> List[str]:
        """Gera lista de fatores relevantes encontrados no cadastro original."""
        if not record:
            return ["Sem informações adicionais do cadastro disponíveis para este registro."]

        factors: List[str] = []

        age_text = self._clean_text(record.get("Idade"))
        if age_text:
            factors.append(f"Idade informada: {age_text}")

        employment_text = self._clean_text(record.get("Situação de Trabalho"))
        if employment_text:
            factors.append(f"Situação de trabalho declarada: {employment_text}")

        income_text = self._format_currency(record.get("Remuneração"))
        if income_text:
            factors.append(f"Renda declarada: {income_text}")

        study_text = self._clean_text(record.get("Situação escolar"))
        if study_text:
            factors.append(f"Situação escolar atual: {study_text}")

        job_search = self._normalize_boolean(record.get("Procurando trabalho"))
        if job_search != "Não informado":
            factors.append(f"Busca por trabalho: {job_search}")

        family_benefit = self._normalize_boolean(record.get("Bolsa Família"))
        if family_benefit != "Não informado":
            factors.append(f"Recebe Bolsa Família: {family_benefit}")

        if dependency_flag:
            text = dependency_text or "presença informada"
            factors.append(f"Dependência química relatada: {text}")
        else:
            if dependency_text:
                factors.append(f"Dependência química: {dependency_text}")
            else:
                factors.append("Dependência química: não há registro atualizado no cadastro.")

        return factors

    def _build_action_plan(
        self,
        report: Dict[str, Any],
        dependency_flag: bool,
        dependency_text: str,
        record: Dict[str, Any]
    ) -> List[str]:
        """Elabora sugestões práticas para a equipe com base na classificação."""
        classification = self._clean_text(report.get("Classificação"))
        risk = self._clean_text(report.get("Nível_de_Risco"))

        actions: List[str] = []

        if classification == "Apoio Intensivo":
            actions.append("Realizar reunião interdisciplinar para montar plano intensivo nos próximos 30 dias.")
            if dependency_flag:
                actions.append(
                    "Agendar avaliação especializada de dependência química e garantir adesão ao tratamento indicado."
                )
            else:
                actions.append(
                    "Manter rastreio periódico de dependência química e saúde mental, mesmo sem relato atual."
                )
            actions.append(
                "Utilizar os dados socioeconômicos para priorizar benefícios, oficinas e acompanhamento familiar."
            )
            actions.append("Registrar evolução semanal em prontuário e revisar metas curtas com a equipe técnica.")
        elif classification == "Possível Liberação":
            actions.append("Construir plano de transição assistida com check-ins quinzenais até o desligamento.")
            if dependency_flag:
                actions.append(
                    "Confirmar estabilidade no tratamento de dependência química antes da liberação definitiva."
                )
            else:
                actions.append(
                    "Estabelecer rede de apoio para prevenir recaídas em dependência química após o desligamento."
                )
            actions.append(
                "Aproveitar os dados para alinhar encaminhamentos de trabalho, educação e suporte financeiro."
            )
            actions.append("Documentar fatores que sustentam a autonomia para futuras reavaliações da ONG.")
        elif classification == "Monitoramento Ativo":
            actions.append("Programar acompanhamento mensal com indicadores claros de avanço e alerta.")
            if dependency_flag:
                actions.append(
                    "Monitorar possíveis recaídas em dependência química e acionar rede especializada quando necessário."
                )
            else:
                actions.append(
                    "Oferecer orientações preventivas sobre dependência química, mesmo sem relato atual."
                )
            actions.append(
                "Direcionar os dados coletados para mentorias, oficinas e intervenções específicas conforme evolução."
            )
        else:
            actions.append(
                "Utilizar os dados disponíveis para construir plano individual com metas e monitoramento periódico."
            )

        job_search_status = self._normalize_boolean(record.get("Procurando trabalho"))
        if job_search_status == "Não":
            actions.append(
                "Promover mentoria para retomada da busca ativa por trabalho e articular com empregadores parceiros."
            )
        elif job_search_status == "Sim":
            actions.append(
                "Acompanhar semanalmente as candidaturas em andamento para garantir continuidade do engajamento."
            )

        study_status = self._clean_text(record.get("Situação escolar")).lower()
        if study_status:
            simplified_study = study_status.replace("ç", "c").replace("ã", "a")
            if "interromp" in simplified_study:
                actions.append("Oferecer apoio para retomada educacional (EJA, cursos rápidos ou reforço).")
            elif "cursando" in simplified_study:
                actions.append("Garantir suporte para permanência na trajetória educacional atual.")

        employment_status = self._clean_text(record.get("Situação de Trabalho")).lower()
        if employment_status:
            employment_simple = employment_status.replace("ç", "c").replace("ã", "a")
            if any(token in employment_simple for token in {"desempregado", "sem emprego", "dona de casa"}):
                actions.append(
                    "Avaliar alternativas de geração de renda emergencial enquanto o emprego formal não acontece."
                )

        if risk:
            actions.append(f"Registrar no prontuário que o risco identificado está classificado como {risk}.")

        actions.append(
            "Atualize o prontuário com estes indicadores para fortalecer a tomada de decisão baseada em evidências."
        )

        return actions
