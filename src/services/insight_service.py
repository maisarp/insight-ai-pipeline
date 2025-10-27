from __future__ import annotations

from pathlib import Path
from typing import Dict, List


class ClusterInsightService:
    """Carrega descrições de clusters a partir do relatório de treinamento."""

    def __init__(self, training_report_path: Path | None = None) -> None:
        """Inicializa o serviço e prepara o mapa de insights."""
        self.training_report_path = training_report_path
        self._disclaimer = (
            "OBSERVAÇÃO IMPORTANTE: Este modelo inicial foi treinado com uma amostra reduzida,\n"
            "servindo apenas como apoio às equipes técnicas. Sempre confirme as sugestões com\n"
            "avaliação humana especializada."
        )
        self.insight_map = self._build_insight_map()

    def _build_insight_map(self) -> Dict[str, Dict[str, List[str] | str]]:
        """Monta o dicionário de insights mesclando padrões e relatório."""
        insights = self._default_insights()

        if self.training_report_path and self.training_report_path.exists():
            text = self.training_report_path.read_text(encoding='utf-8')
            parsed = self._parse_training_report(text)

            for label, payload in parsed.items():
                bucket = insights.setdefault(label, {})
                training_summary = payload.get("training_summary")
                if training_summary:
                    bucket["training_summary"] = training_summary
                training_details = payload.get("training_details")
                if training_details:
                    bucket["training_details"] = training_details

        return insights

    def _default_insights(self) -> Dict[str, Dict[str, List[str] | str]]:
        """Fornece descrições padrão baseadas na última rodada de treinamento."""
        return {
            "Possível Liberação": {
                "summary": "Perfil com autonomia crescente e prontidão para transição assistida.",
                "details": [
                    "Indicadores de renda e ocupação apontam capacidade crescente de sustentação própria.",
                    "Busca ativa por trabalho e continuidade educacional sustentam a transição.",
                    "Investigue se permanecem barreiras ocultas (como dependência química) antes do desligamento.",
                    "Planeje roteiro de acompanhamento pós-saída para evitar rupturas abruptas.",
                    "Utilize os dados para conectar o participante a vagas, cursos e monitoramento remoto.",
                    self._disclaimer,
                ]
            },
            "Apoio Intensivo": {
                "summary": "Perfil com vulnerabilidades acumuladas que exige apoio contínuo e intensivo.",
                "details": [
                    "Demanda articulação multidisciplinar entre assistência social, saúde e geração de renda.",
                    "Avalie sinais de dependência química, saúde mental e física para priorizar intervenções.",
                    "Fortaleça vínculos com redes de proteção social e serviços especializados.",
                    "Priorize atividades práticas que desenvolvam habilidades profissionais e educacionais.",
                    "Registre metas de curto prazo e indicadores monitorados semanalmente pela equipe.",
                    self._disclaimer,
                ]
            },
            "Monitoramento Ativo": {
                "summary": "Perfil intermediário com avanços parciais e necessidade de supervisão próxima.",
                "details": [
                    "Oscilações em renda, estudo ou vínculo laboral requerem acompanhamento estruturado.",
                    "Mapeie gatilhos de regressão (incluindo recaídas em uso de substâncias) para respostas rápidas.",
                    "Mantenha reuniões periódicas para recalibrar o plano individual conforme evolução.",
                    "Aproveite os dados para direcionar mentorias, oficinas e intervenções específicas.",
                    self._disclaimer,
                ]
            }
        }

    def _parse_training_report(self, text: str) -> Dict[str, Dict[str, List[str] | str]]:
        """Extrai resumos e bullets diretamente do relatório."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        parsed: Dict[str, Dict[str, List[str] | str]] = {}

        for line in lines:
            lowered = line.lower()
            if lowered.startswith("possível liberação:") or lowered.startswith("possivel liberacao:"):
                parsed.setdefault("Possível Liberação", {})["training_summary"] = line
            if lowered.startswith("apoio intensivo:"):
                parsed.setdefault("Apoio Intensivo", {})["training_summary"] = line

        capture_details = False
        detail_bucket: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if "Motivos da classificação 'Possível Liberação'" in stripped:
                capture_details = True
                detail_bucket = []
                continue
            if capture_details:
                if stripped.startswith("•"):
                    detail_bucket.append(stripped.lstrip("• "))
                    continue
                if stripped and not stripped.startswith("•"):
                    if detail_bucket:
                        parsed.setdefault("Possível Liberação", {})["training_details"] = detail_bucket
                    capture_details = False
        if capture_details and detail_bucket:
            parsed.setdefault("Possível Liberação", {})["training_details"] = detail_bucket

        return parsed

    def get_summary(self, label: str) -> str:
        """Retorna resumo do cluster informado."""
        data = self.insight_map.get(label)
        if not data:
            return "Sem resumo disponível para este cluster."
        return str(data.get("summary", "Sem resumo disponível para este cluster."))

    def get_details(self, label: str) -> List[str]:
        """Lista os principais pontos associados ao cluster."""
        data = self.insight_map.get(label)
        if not data:
            return []
        details = data.get("details", [])
        return [str(item) for item in details]

    def get_training_context(self, label: str) -> str | None:
        """Retorna descrição do treinamento utilizada como referência adicional."""
        data = self.insight_map.get(label)
        if not data:
            return None
        context = data.get("training_summary")
        if context:
            return str(context)
        return None

    def get_training_notes(self, label: str) -> List[str]:
        """Lista indicadores capturados no relatório de treinamento original."""
        data = self.insight_map.get(label)
        if not data:
            return []
        notes = data.get("training_details", [])
        return [str(item) for item in notes]
