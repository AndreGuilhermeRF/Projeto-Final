# Implementações desta versão

- NER clínico com `BrazilAI/openmed-br-clinical-pt` via Transformers (CPU).
- Extração de relações estruturadas via Ollama, validando a presença literal das entidades e evidências.
- Embeddings reais por `nomic-embed-text` via Ollama; vetor de 768 dimensões.
- Fulltext para chunks e entidades, ambos consultados.
- Busca híbrida Vector + Fulltext com Reciprocal Rank Fusion.
- Geração de Cypher pelo Ollama com allowlist read-only e limite de saltos.
- Caminhos completos incluídos no contexto da LLM e exibidos com IDs [P#].
- PageRank, Louvain e Leiden via Neo4j Graph Data Science.
- Interface de avaliação com conjunto-ouro JSON para NER/RE e métricas Precision/Recall/F1.
- Testes unitários em `tests/test_core.py`.
- Correções de compatibilidade de anotações de tipo para Python 3.8.

## Pré-requisitos operacionais

1. `ollama pull qwen2.5:3b`
2. `ollama pull nomic-embed-text`
3. Instalar o plugin Neo4j Graph Data Science no DBMS.
4. Configurar `.env` com as credenciais Neo4j.
5. Se existir o índice antigo de vetores com 512 dimensões, seguir a migração no README.

## Validação feita nesta entrega

- Compilação de todos os módulos Python: aprovada.
- Sintaxe compatível com Python 3.8: aprovada por análise AST.
- Testes unitários locais: 6/6 aprovados.

## Ainda depende do ambiente do usuário

Não foi possível executar integração real contra o Neo4j Desktop, Ollama e download/inferência do modelo NER neste ambiente de preparação. A disponibilidade do GDS e os índices devem ser confirmados na máquina de execução. O F1 publicado do NER não substitui avaliação no ChatBulário.
