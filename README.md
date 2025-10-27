# Insight AI Pipeline

Ferramenta local de análise inteligente que ajuda programas sociais a tomarem decisões baseadas em dados sobre a prontidão dos acolhidos. O objetivo central é permitir que as equipes sejam **data driven**, transformando planilhas e registros em indicadores acionáveis, relatórios claros e sugestões de acompanhamento.

## Visão Geral

- **Propósito:** apoiar organizações sociais na priorização de casos, planejamento de desligamentos assistidos e direcionamento de recursos para quem mais precisa.
- **Abordagem:** o aplicativo processa planilhas, gera clusters (perfis) e entrega relatórios com explicações, recomendações e boas práticas de acompanhamento humano.
- **Valor agregado:** com dados organizados e padronizados, a instituição consegue escalar atendimentos, reduzir retrabalho e ampliar o repertório de ações baseadas em evidências.

## Por que Governança de Dados Importa

Tomar decisões confiáveis depende de dados completos, consistentes e atualizados. Este projeto reforça que:

- **Coletar tudo no mesmo padrão** (nomes de colunas, formatos de data, codificações) evita perdas no processamento e melhora a qualidade das análises.
- **Registrar cada atendimento** gera histórico para medir evolução, testar hipóteses e ajustar políticas internas.
- **Manter documentação das fontes** (quem preencheu, quando, como) aumenta a credibilidade dos insights e facilita auditorias.
- **Garantir segurança e anonimização** protege os acolhidos e cumpre a LGPD, permitindo uso ético da inteligência de dados.

Quanto maior a aderência às boas práticas de governança, mais robustos serão os insights e as ações estratégicas sugeridas pela ferramenta.

## Fluxo do Projeto

1. **Entrada de dados:** planilhas `.xlsx`, `.xls` ou `.csv` contendo informações socioeconômicas dos acolhidos.
2. **Processamento local:** limpeza, transformação e normalização das features relevantes.
3. **Análise por cluster:** o modelo agrupa perfis semelhantes e calcula indicadores como confiança e nível de risco.
4. **Geração de relatórios:** arquivos `.txt` com visão geral, recomendações e melhores práticas, preservando o processamento no computador da organização.
5. **Apoio a decisões humanas:** os resultados são um ponto de partida para reuniões técnicas, construção de planos individualizados e priorização de iniciativas.

## Recursos Principais

- **Relatórios interpretáveis:** cada cluster vem acompanhado de resumos em linguagem acessível, recomendações de acompanhamento e orientações para uso responsável de IA aberta em brainstorms.
- **Metadados de treinamento:** o aplicativo informa quando e como o modelo foi treinado, além de lembrar que o processo roda 100% offline.
- **Console guiado:** fluxo interativo para selecionar arquivos, visualizar insights e localizar os relatórios gerados.

## Segurança e LGPD

- Todos os dados permanecem no computador da ONG; não há envio para serviços externos.
- Identificadores são utilizados apenas nos relatórios locais (se existirem nas planilhas) e não são incorporados ao modelo.
- O relatório reforça práticas de anonimização antes de qualquer uso adicional de IA aberta.

## Como Executar (Operação)

1. Coloque `AI_insights.exe` em uma pasta local (ex.: `Documentos/AI_insights`).
2. Dê dois cliques no executável. Caso o SmartScreen apareça, selecione **Mais informações** → **Executar assim mesmo**.
3. Siga as instruções do terminal: escolha a planilha (`.xlsx`, `.xls` ou `.csv`) e aguarde a geração dos resultados.
4. Consulte a nova pasta `IA_insights_arquivos` criada ao lado do arquivo analisado; ali estará o relatório `cluster_summary_*.txt` (referência principal) e, se necessário, uma versão `.csv` convertida da planilha original.

## Como Executar (Desenvolvimento)

```bash
# 1. Criar e ativar um ambiente virtual (opcional, mas recomendado)
python -m venv .venv
.venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar a aplicação em modo console
py -3 app.py
```

### Treinar/Re-treinar o modelo

```bash
# Certifique-se de ter o arquivo de dados configurado em data/
py -3 scripts/train_clustering.py
```

- Saída esperada: `model/clustering_model.joblib`, `model/scaler.joblib` e `model/clustering_metadata.json` atualizados.
- Ajuste variáveis no `.env` (ex.: `DATABASE_PATH`) se quiser apontar para uma planilha específica.

### Empacotar como executável Windows (PyInstaller)

```bash
pyinstaller AI_insights.spec
```

- O executável resultante ficará em `dist/AI_insights/AI_insights.exe`.
- Após gerar o `.exe`, copie-o juntamente com eventuais recursos adicionais para a pasta de distribuição da ONG.

## Sobre o Uso de IA no Desenvolvimento

Ferramentas de IA generativa foram usadas como suporte de produtividade (ex.: sugestões de código, brainstorming). Toda entrega resultante foi revisada, testada e validada manualmente para garantir aderência às metas do projeto, ética e segurança dos dados.