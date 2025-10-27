
## Como Executar

Este passo a passo considera que você está utilizando o aplicativo preparado para Windows (`app.exe`).

### 1. Preparar a pasta do aplicativo

- Copie o arquivo `AI_insights.exe` fornecido para uma pasta local de sua preferência (ex.: `Documentos/AI_insights`).
- Nenhum outro arquivo é necessário: o modelo, o scaler e o relatório de treinamento já estão embutidos dentro do executável.

### 2. Iniciar a análise

- Dê um duplo clique em `AI_insights.exe`. Se o Windows exibir o SmartScreen, clique em **Mais informações** e depois em **Executar assim mesmo**.
- Uma janela preta (terminal) será aberta com orientações em português. Pressione **Enter** quando solicitado para selecionar a planilha (`.xlsx`, `.xls` ou `.csv`).
- Escolha o arquivo com os dados dos participantes e aguarde o processamento.

### 3. Conferir os resultados

- Ao final, o aplicativo informa onde os relatórios foram salvos. Por padrão, é criada a pasta `ia_analise_relatorio` ao lado da planilha original.
- Dentro dessa pasta você encontrará:
	- Um `.csv` com as classificações detalhadas de cada registro.
	- Um `.txt` com o relatório individual/personalizado e as recomendações de atuação da ONG.
	- (Quando necessário) Uma cópia da planilha convertida para CSV.

### 4. Dúvidas e suporte

- Se aparecer qualquer mensagem de erro, anote o texto exibido e repasse para a equipe técnica responsável.
- Caso o antivírus ou o SmartScreen bloqueie a execução, marque o `AI_insights.exe` como confiável e tente novamente.
