---
trigger: always_on
---
# Coding Style - Real Practices and Readability

## Philosophy

This project follows a style that prioritizes:
1. **Readability** > Sophistication
2. **Real market practices** > Inventions
3. **Clarity** > Excessive conciseness
4. **Technical honesty** > Impressing

---

## Naming Rules

### Variables (snake_case in English)

**GOOD - Descriptive and clear:**
```python
average_age = cluster_data['idade'].mean()
percentage_approved = data['is_approved'].mean() * 100
looking_for_work = data['procurando_trabalho'].mean() * 100
is_studying = data['estudando'].mean() * 100
avg_program_months = data['tempo_programa_meses'].mean()
```

**AVOID - Too short or ambiguous:**
```python
avg_age = ...  # Too abbreviated
substance = ...  # Too generic
work = ...  # Too generic
```

**AVOID - Portuguese in code:**
```python
idade_media = ...  # Code should be in English
use_ai_frequencia = ...  # Mixes Portuguese + English
```

### Functions and Methods (snake_case in English)

**GOOD - Descriptive verbs:**
```python
def calculate_cluster_statistics(cluster_data):
    """Calcula estatísticas do cluster."""
    
def interpret_vulnerability_barriers(profiles):
    """Interpreta barreiras de vulnerabilidade."""
    
def generate_aggregated_report(results):
    """Gera relatório com dados agregados."""

# Note: Docstrings remain in PT-BR as per project rules
```

**AVOID - Vague names:**
```python
def process():  # Process what?
def get_data():  # Which data?
def calc():  # Calculate what?
```

### Classes (PascalCase in English)

**GOOD - Clear nouns:**
```python
class DataProcessor:
class ClusteringEngine:
class VulnerabilityAnalyzer:
```

---

## Documentation

### Docstrings (ALWAYS in Portuguese PT-BR)

**GOOD - Complete and clear:**
```python
def calculate_cluster_statistics(cluster_data):
    """
    Calcula estatísticas descritivas de um cluster.
    
    Args:
        cluster_data (pd.DataFrame): Dados do cluster a analisar.
    
    Returns:
        dict: Dicionário com estatísticas (média, mediana, etc).
    """
```

**AVOID - English or incomplete:**
```python
def calculate_cluster_statistics(cluster_data):
    """Calculate cluster statistics."""  # WRONG: English
    
def calculate_cluster_statistics(cluster_data):
    """Calcula estatísticas."""  # WRONG: Too vague
```

### Comments (Portuguese PT-BR, only when necessary)

**GOOD - Explains the "why":**
```python
# Normaliza para 0-100 para facilitar interpretação em relatórios
percentage = (value / total) * 100

# Filter unemployed for barrier analysis
unemployed_mask = data['employment_status'] == 0
```

**AVOID - Obvious or redundant:**
```python
# Increment counter
counter += 1  # WRONG: Obvious

# Loop through clusters
for cluster in clusters:  # WRONG: Redundant
```

---

## Code Structure

### Simplicity > Complexity

**GOOD - Direct and clear:**
```python
def analyze_cluster(cluster_data):
    """Analyze cluster characteristics."""
    average_age = cluster_data['age'].mean()
    average_education = cluster_data['education_level'].mean()
    
    return {
        'average_age': average_age,
        'average_education': average_education
    }
```

**AVOID - Over-engineering:**
```python
def analyze_cluster(cluster_data):
    """Analisa características de um cluster."""
    # WRONG: Doesn't need a class for this
    analyzer = ClusterStatisticsAnalyzer()
    analyzer.set_data(cluster_data)
    analyzer.configure_metrics(['age', 'education'])
    return analyzer.compute()
```

### No Forced Encapsulation (For Now)

```python
class DataProcessor:
    def __init__(self):
        self.dataframe = None  # Public, OK
        self.processed_data = None  # Public, OK
```

**DEFER - Complex encapsulation:**
```python
class DataProcessor:
    def __init__(self):
        self._dataframe = None  # Private - defer until better understood
        
    @property
    def dataframe(self):  # Property - defer
        return self._dataframe
```

---

## Practices to AVOID

### 1. Inventing Non-Existent Patterns

**DO NOT:**
- Create files like "SECURITY_MAPPING.py" if it's not a common practice
- Invent nomenclatures that aren't market standard
- Add unnecessary complexity

**DO:**
- Follow standard Python conventions (PEP 8)
- Use documented and recognized practices
- Ask if unsure whether something is standard

### 2. Exaggerating Concepts

**DO NOT:**
- Call simple automation "AI"
- Add unnecessary abstraction layers
- Use technical jargon unnecessarily

**DO:**
- Be honest about what the code does
- Use correct technical terms
- Keep it simple when possible

### 3. Non-Idiomatic Code

**AVOID:**
```python
# Too verbose
if condition == True:
    return True
else:
    return False

# Better:
return condition
```

---

## Quality Checklist

Before considering code ready, validate:

- [ ] Variable names are descriptive and in English
- [ ] Docstrings are in Portuguese PT-BR
- [ ] No unnecessary complexity
- [ ] Code follows real market practices
- [ ] No "invented" patterns
- [ ] Readable for a junior developer
- [ ] Comments explain "why", not "what"

---

## Learning Principles

### For the Developer (Junior)

This project is a learning opportunity. Code should:

1. **Teach real best practices** - Not invented ones
2. **Be understandable** - Not impressive
3. **Follow market standards** - Not experiments
4. **Allow gradual evolution** - Not force advanced concepts

### Technical Progression

**Now (Junior):**
- Clear naming
- Complete docstrings
- Clean and readable code
- Basic market practices

---

## References

- [PEP 8 - Style Guide for Python Code](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Clean Code (Robert C. Martin)](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)

---

## Document Evolution

This document evolves as the project and developer evolve.

**Last update:** 26/10/2024  
**Version:** 1.0  
**Status:** Active - Apply to all new/refactored code