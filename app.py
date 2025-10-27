"""
Aplicação console para análise de prontidão com modelo de clustering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

from src.predictor_clustering import PredictorClustering
from src.services.file_service import InputFileService
from src.services.insight_service import ClusterInsightService
from src.services.report_service import ClusterReportBuilder

def resolve_asset_path(relative_path: str) -> Path:
    """Resolve caminho de assets tanto no desenvolvimento quanto no executável."""
    base_path = getattr(sys, "_MEIPASS", None)
    if base_path:
        return Path(base_path) / relative_path
    return Path(__file__).resolve().parent / relative_path


MODEL_PATH = resolve_asset_path("model/clustering_model.joblib")
SCALER_PATH = resolve_asset_path("model/scaler.joblib")
TRAINING_REPORT_PATH = resolve_asset_path("data/training_report.md")
IDENTIFIER_PRIORITY = ["ID_Atendido", "Nome", "CPF", "Matrícula", "ID"]

custom_theme = Theme(
    {
        "header": "bold white on blue",
        "title": "bold cyan",
        "info": "white",
        "warning": "bold yellow",
        "success": "bold green",
        "error": "bold red",
        "highlight": "bold magenta",
        "prompt": "bold cyan",
    }
)
console = Console(theme=custom_theme)

def print_header() -> None:
    """Mostra mensagem inicial para contextualizar o usuário."""
    console.print()
    console.rule("[header] Sistema de Análise de Prontidão (Clustering) ")
    console.print()


def print_usage_overview() -> None:
    """Explica o propósito do assistente antes de solicitar ações."""
    overview_message = (
        "[info]Este assistente ajuda a identificar o nível de prontidão das pessoas atendidas.\n"
        "Nenhuma informação sai do seu computador: tudo é processado e salvo localmente.\n\n"
        "[highlight]Como funciona:[/highlight]\n"
        "[info]  1. Você escolhe a planilha que deseja analisar.\n"
        "  2. O sistema pode gerar uma cópia em CSV na mesma pasta do arquivo original.\n"
        "  3. Relatórios com os resultados ficam salvos na sua máquina, em uma nova pasta.\n\n"
        "[info]Dica: tenha a planilha em mãos antes de prosseguir."
    )

    console.print(
        Panel.fit(overview_message, title="[title]Visão Geral", border_style="highlight")
    )


def ensure_model_assets() -> None:
    """Garante que modelo e scaler estejam disponíveis antes da execução."""
    missing_assets = []
    if not MODEL_PATH.exists():
        missing_assets.append(str(MODEL_PATH))
    if not SCALER_PATH.exists():
        missing_assets.append(str(SCALER_PATH))

    if missing_assets:
        console.print("Arquivos essenciais não encontrados:", style="error")
        for asset in missing_assets:
            console.print(f"- {asset}", style="error")
        console.print(
            "\nExecute o script de treinamento antes de continuar (scripts/train_clustering.py).",
            style="warning",
        )
        sys.exit(1)


def open_file_selection_dialog() -> Optional[str]:
    """Abre uma janela de seleção de arquivo e retorna o caminho escolhido."""
    try:
        from tkinter import Tk, filedialog

    except ImportError:
        return None

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.after(0, root.focus_force)
    try:
        root.update()
        selected_path = filedialog.askopenfilename(
            title="Selecione a planilha para análise",
            filetypes=[
                ("Planilhas", "*.xlsx *.xls *.csv"),
                ("Todos os arquivos", "*.*"),
            ],
            parent=root,
        )
    except Exception:
        selected_path = None
    finally:
        root.destroy()

    return selected_path or None


def request_file_path(file_service: InputFileService) -> Path:
    """Solicita ao usuário o arquivo a ser analisado de forma guiada."""
    while True:
        choice = open_file_selection_dialog()

        if not choice:
            console.print(
                "\nNenhum arquivo foi selecionado ou a janela não pôde ser aberta.",
                style="warning",
            )
            manual_choice = console.input(
                "[prompt]Informe o caminho completo do arquivo ou pressione Enter para tentar novamente: "
            ).strip()

            if not manual_choice:
                continue

            choice = manual_choice

        path = file_service.normalize_path(choice)
        is_valid, message = file_service.validate_input(path)
        if is_valid:
            return path

        console.print(f"\n{message}\n", style="warning")
        console.print("Vamos tentar novamente.", style="info")

def detect_identifier_columns(dataset_path: Path) -> List[str]:
    """Identifica colunas com dados válidos respeitando a ordem de prioridade."""
    try:
        if dataset_path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(dataset_path, encoding="utf-8-sig")
        else:
            dataframe = pd.read_excel(dataset_path)
    except Exception as exc:
        console.print(
            "\nNão foi possível ler a planilha para identificar colunas. Vamos seguir com o padrão.",
            style="warning",
        )
        console.print(str(exc), style="warning")
        return []

    if dataframe.empty:
        return []

    def has_meaningful_value(cell: object) -> bool:
        text = str(cell).strip()
        return bool(text) and text.lower() not in {"nan", "none"}

    detected: List[str] = []
    for column in IDENTIFIER_PRIORITY:
        if column not in dataframe.columns:
            continue
        if dataframe[column].apply(has_meaningful_value).any():
            detected.append(column)

    return detected


def run_console_flow() -> None:
    """Executa o fluxo completo de coleta, análise e apresentação."""
    ensure_model_assets()

    file_service = InputFileService()
    print_usage_overview()
    console.input("\n[prompt]Pressione Enter quando estiver pronta(o) para escolher o arquivo...")
    source_path = request_file_path(file_service)
    prepared_path, converted = file_service.ensure_csv(source_path)

    if converted:
        console.print(
            "\n✓ Criamos uma cópia em CSV para facilitar a análise:",
            style="success",
        )
        console.print(f"  {prepared_path}", style="info")
    else:
        console.print(
            "\nArquivo já está em formato CSV. Vamos seguir para a análise.",
            style="info",
        )

    console.print(
        "\nAs próximas informações ajudam apenas na organização dos relatórios gerados no seu computador."
        " Nenhum dado é enviado para outros sistemas.",
        style="info",
    )

    identifier_columns = detect_identifier_columns(prepared_path)
    if identifier_columns:
        console.print(
            f"Colunas identificadoras detectadas: {', '.join(identifier_columns)}",
            style="highlight",
        )
    else:
        console.print(
            "\nNão encontramos dados identificadores confiáveis. Tentaremos usar ID_Atendido como padrão.",
            style="warning",
        )
        identifier_columns = [IDENTIFIER_PRIORITY[0]]

    predictor = PredictorClustering(str(MODEL_PATH), str(SCALER_PATH), identifier_columns)
    predictor.load_and_process_data(str(prepared_path)).predict()

    insight_service = ClusterInsightService(TRAINING_REPORT_PATH)
    user_output_dir = prepared_path.parent / "ia_analise_relatorio"
    report_builder = ClusterReportBuilder(insight_service, user_output_dir)

    report_builder.print_overview(predictor)
    stats = predictor.get_summary_stats()
    report_builder.show_cluster_insights(stats["clusters"].keys())

    fallback_identifiers = identifier_columns or IDENTIFIER_PRIORITY
    report_builder.print_individual_overview(predictor, fallback_identifiers)

    saved_files = report_builder.save_outputs(predictor)
    console.print("Arquivos gerados no seu computador:", style="title")
    for generated in saved_files:
        console.print(f"- {generated.resolve()}", style="info")

    console.print(
        f"\nVocê encontra todos os arquivos em: {user_output_dir.resolve()}",
        style="success",
    )

    if converted and prepared_path.exists():
        console.print(
            "\nO CSV convertido permanece disponível caso queira reutilizá-lo.",
            style="info",
        )

    console.print(
        "\nProcesso concluído com sucesso. Obrigado por utilizar o sistema!\n",
        style="success",
    )


def main() -> None:
    """Ponto de entrada principal com tratamento de exceções."""
    try:
        print_header()
        run_console_flow()
    except KeyboardInterrupt:
        console.print("\nOperação interrompida pelo usuário.", style="warning")
        sys.exit(0)
    except Exception as exc:
        console.print("\nErro inesperado durante a execução:", style="error")
        console.print(str(exc), style="error")
        sys.exit(1)


if __name__ == "__main__":
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