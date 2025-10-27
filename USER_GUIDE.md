
## Como Executar

Este passo a passo considera que você está utilizando o aplicativo preparado para Windows (`AI_insights.exe`). Tudo acontece localmente no seu computador.

### 1. Preparar a pasta do aplicativo

- Copie o arquivo `AI_insights.exe` para uma pasta de sua preferência (ex.: `Documentos/AI_insights`).
- O executável já carrega o modelo, o scaler e o relatório de treinamento incorporados. Não é preciso baixar dependências extras.

### 2. Iniciar a análise

- Clique duas vezes em `AI_insights.exe`. Se o Windows alertar via SmartScreen, selecione **Mais informações** e depois **Executar assim mesmo**.
- O terminal exibirá instruções guiadas em português. Pressione **Enter** quando solicitado para selecionar a planilha (`.xlsx`, `.xls` ou `.csv`).
- Escolha o arquivo com os dados dos acolhidos e aguarde o processamento. O sistema pode converter automaticamente para `.csv` quando necessário.

### 3. Conferir os resultados

- No final do processamento, o aplicativo indica a pasta onde os arquivos foram gerados. Por padrão, cria-se `IA_insights_arquivos` ao lado da planilha analisada.
- Arquivos entregues:
	- `cluster_summary_YYYYMMDD_HHMMSS.txt`: relatório principal com visão geral, recomendações, boas práticas e metadados do modelo.
	- (Opcional) `arquivo_original_convertido.csv`: somente quando a planilha inicial não estava em `.csv`.

### 4. Como interpretar o relatório `.txt`

O relatório é dividido em seções. Segue um guia rápido:

- **Aviso inicial:** reforça que o modelo é experimental, treinado com poucos registros e serve como apoio à decisão humana.
- **Resumo geral:** mostra total de casos, confiança média e explicações sobre cada indicador.
	- *Cluster:* agrupamento de perfis semelhantes identificado pelo modelo.
	- *Confiança da análise:* quão próximo o registro está do centro do cluster (quanto maior, mais aderente ao padrão do grupo).
	- *Nível de risco:* grau de atenção recomendado (Baixo, Médio ou Alto) para orientar prioridade de acompanhamento.
	- *Silhouette:* métrica que mede a qualidade da separação entre clusters (valores maiores indicam grupos mais bem definidos).
- **Distribuições:** porcentagem de acolhidos por cluster e por nível de risco.
- **Insights principais:** resumo em linguagem acessível com recomendações para cada cluster.
- **Detalhamento individual (até 10 primeiros registros):** apresenta classificações, indicações do cadastro original e recomendações prioritárias.
- **Boas práticas para usar IA aberta:** seção separada por `======` com orientações de anonimização, brainstorm responsável e exemplos de perguntas seguras.
- **Informações sobre o modelo:** após outra linha de separação, mostra metadados de treinamento e relembra que todo o processamento é local.
- **Avisos e responsabilidade:** fechamento demarcado por `------` reforçando que o modelo é experimental e que decisões continuam com a equipe humana.

### 5. Boas práticas ao usar os resultados

- Combine a análise com avaliação profissional (psicossocial, pedagógica, jurídica etc.).
- Atualize a planilha original sempre que houver novas informações ou evoluções do acolhido.
- Padronize nomes de colunas e formatos de dados para preservar a qualidade das próximas análises.
- Antes de consultar qualquer IA aberta, remova dados pessoais e compartilhe apenas descrições analíticas.

### 6. Dúvidas e suporte

- Registre mensagens de erro exibidas no terminal e compartilhe com a equipe técnica responsável pelo projeto.
- Se antivírus ou SmartScreen bloquearem a execução, sinalize o arquivo como confiável ou mova-o para uma pasta com permissões adequadas.
