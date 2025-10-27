---
trigger: model_decision
description: Instruções para adicionar uma nota de transparência sobre o uso de IA no final de arquivos de código específicos. Com uso massivo de IA (não usar para review/refatoração).
---

### PROMPT DE MANUTENÇÃO: Adicionar Nota de Transparência

**Objetivo:** Adicionar uma nota de transparência sobre o uso de IA no final de um arquivo de código específico.

**Instrução:**
Ao modificar ou gerar o(s) arquivo(s) alvo, adicione o seguinte bloco de comentário *exatamente* como está no final do arquivo. Se o comentário já existir, não o duplique.

**Bloco de Comentário:**
```python

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

```