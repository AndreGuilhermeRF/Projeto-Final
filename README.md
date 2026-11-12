# GraphRAG Med — ChatBulário + Neo4j

Versão ampliada com Neo4j como armazenamento persistente, NER clínico OpenMed-BR, extração de relações assistida por Ollama, embeddings semânticos reais, recuperação híbrida, caminhos explícitos e geração controlada de Cypher.

> **Uso acadêmico e experimental.** Não é dispositivo médico nem substitui bula oficial, farmacêutico ou profissional de saúde. As extrações automáticas podem conter erros e exigem revisão humana.

## Requisitos

- Windows 10/11, Python 3.8 (compatibilidade das dependências fixada) ou Python mais recente.
- Neo4j 5.11+; para índice vetorial nativo, prefira Neo4j 5.18+.
- Neo4j Graph Data Science (GDS) instalado no servidor para PageRank, Louvain e Leiden.
- Ollama instalado e em execução.
- 8 GB RAM: comece com `CHATBULARIO_LIMIT=100`, NER CPU e modelo Ollama pequeno.

## Instalação no Windows

Na pasta que contém `app.py`:

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se o PyTorch ainda não estiver instalado, instale a versão CPU adequada ao seu Python pelo seletor oficial: https://pytorch.org/get-started/locally/

Copie o arquivo de exemplo e configure sua senha local:

```powershell
Copy-Item .env.example .env
notepad .env
```

Não compartilhe o `.env` e não o envie para repositórios.
## Reconstrução do dataset ChatBulário

O arquivo `graphrag_med/data/chatbulario.json` não é armazenado diretamente neste repositório devido ao seu tamanho. Ele pode ser reconstruído automaticamente a partir do dataset original disponível no Hugging Face.

O projeto inclui o script `gerar_chatbulario.py`, responsável por baixar e processar os registros em modo streaming, sem carregar todo o dataset na memória.

**1. Instale a dependência necessária:**

```powershell
pip install datasets
```
**2. Gere o arquivo ChatBulário:**

Na pasta raiz do projeto, execute:

```powershell
python gerar_chatbulario.py
```

Por padrão, o script:
- Utiliza o dataset `walmeidadf/ChatBulario`, disponível no Hugging Face.
- Processa a divisão `train` em modo streaming.
- Ignora registros sem pergunta ou resposta.
- Aplica o filtro de qualidade com `fuzzy_score` mínimo de 80.
- Salva os registros progressivamente.
- Valida o JSON antes de substituir o arquivo de destino.

**3. Configure a quantidade de registros (opcional):**

Para gerar uma base reduzida, adequada a computadores com recursos limitados:

```powershell
python gerar_chatbulario.py --limit 100
```

Para gerar todos os registros:

```powershell
python gerar_chatbulario.py
```

Para alterar o limite mínimo de qualidade:

```powershell
python gerar_chatbulario.py --min-fuzzy-score 85
```

**4. Confira o arquivo gerado:**

Após a conclusão, o arquivo estará disponível em:

```text
graphrag_med/data/chatbulario.json
```

O script apresenta a quantidade de registros lidos, salvos e ignorados, além do tamanho final do arquivo.

**Importante:** a geração depende de conexão com a internet e da disponibilidade do dataset no Hugging Face. A quantidade de registros e o conteúdo podem variar conforme a versão disponibilizada na origem.

O arquivo gerado é local e está excluído do controle de versão pelo `.gitignore`. Portanto, cada pessoa que clonar o projeto deverá executar esse procedimento antes de iniciar a ingestão do ChatBulário.


## Modelos locais

Baixe o modelo de geração e o modelo de embeddings:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

O NER baixa automaticamente na primeira execução o modelo `BrazilAI/openmed-br-clinical-pt` do Hugging Face. Ele é um modelo de classificação de tokens em português brasileiro, com 110M parâmetros, executado em CPU. O cartão do modelo reporta F1 geral de 0,598 no conjunto held-out utilizado pelos autores; o desempenho pode variar nos textos do ChatBulário. Atribuição/licença CC BY 4.0: https://huggingface.co/BrazilAI/openmed-br-clinical-pt

## Configuração do Neo4j

1. Inicie o DBMS no Neo4j Desktop.
2. Confira `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` e `NEO4J_DATABASE` no `.env`.
3. O aplicativo verifica a conexão e aplica `graphrag_med/schema.cypher`.
4. Ative o plugin **Neo4j Graph Data Science** no DBMS para executar PageRank, Louvain e Leiden.

### Migração do índice vetorial antigo

A versão anterior usava um vetor de hashing de 512 dimensões. A nova usa `nomic-embed-text` com 768 dimensões e cria o índice `chunk_embedding_768`, sem reaproveitar o índice antigo. O índice antigo pode ser removido uma única vez no Neo4j Browser:

```cypher
DROP INDEX chunk_embedding IF EXISTS;
```

Se o banco já tiver um índice `chunk_embedding_768` com dimensão diferente, remova-o antes de iniciar:

```cypher
DROP INDEX chunk_embedding_768 IF EXISTS;
```

O schema recria o índice com 768 dimensões. Aguarde o estado `ONLINE` antes da primeira busca vetorial. A busca Fulltext é usada em paralelo e serve de fallback caso o índice vetorial não esteja disponível.

## Executar

```powershell
streamlit run app.py
```

Na primeira inicialização, o sistema:
- carrega o ChatBulário;
- extrai entidades com OpenMed-BR;
- usa o Ollama para extração assistida de relações;
- cria chunks e embeddings semânticos via Ollama;
- persiste fontes, chunks, entidades e relações no Neo4j.

A ingestão inicial pode demorar. O limite é controlado por `CHATBULARIO_LIMIT`.

## O que foi implementado

- **NER:** `BrazilAI/openmed-br-clinical-pt` (doenças, sinais/sintomas, medicamentos, procedimentos e anatomia), mais resolução pelo léxico.
- **RE:** extração estruturada com Ollama, validação de menções e evidência literal; regras e seções do ChatBulário continuam como fallback.
- **Entity resolution:** IDs canônicos e sinônimos preservados.
- **Fulltext:** índice de chunks e índice de entidades utilizados na recuperação.
- **Vector Search:** embeddings reais do `nomic-embed-text` armazenados nos nós `Chunk` e consultados pelo índice vetorial do Neo4j.
- **Busca híbrida:** ranking por Reciprocal Rank Fusion entre Vector Search e Fulltext, seguido de travessia Cypher.
- **Multi-hop:** consultas limitadas a cinco saltos.
- **Graph Analytics:** PageRank, Louvain e Leiden via Neo4j GDS. Se GDS não estiver instalado, a interface informa explicitamente que as métricas não foram calculadas.
- **LLM-to-Cypher:** geração assistida pelo Ollama; apenas consultas de leitura passam pela validação e execução.
- **Caminhos na resposta:** os caminhos completos são apresentados e incluídos no contexto da LLM com referências `[P#]`.
- **Avaliação:** métricas de recuperação e função Precision/Recall/F1 para conjuntos-ouro anotados.

## Variáveis importantes do `.env`

| Variável | Padrão | Finalidade |
|---|---|---|
| `CHATBULARIO_LIMIT` | `100` | Limite inicial de documentos |
| `OLLAMA_MODEL` | `qwen2.5:3b` | LLM para RE e Cypher |
| `ENABLE_MODEL_NER` | `true` | Ativa OpenMed-BR NER |
| `NER_MODEL` | `BrazilAI/openmed-br-clinical-pt` | Modelo NER |
| `NER_THRESHOLD` | `0.40` | Limiar de confiança do modelo |
| `ENABLE_LLM_EXTRACTION` | `true` | Ativa extração de relações |
| `ENABLE_LLM_CYPHER` | `true` | Ativa Cypher gerado pela LLM |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Modelo de embeddings |
| `VECTOR_DIMENSIONS` | `768` | Dimensão do vetor |
| `VECTOR_DIMENSIONS` | `768` | Dimensão do vetor |

## Testes

```powershell
python -m unittest discover -s tests -v
```

Os testes unitários cobrem validações locais, extração por regras e bloqueio de Cypher de escrita. Para comprovar precisão/recall clínicos de NER e RE, é necessário construir e anotar um conjunto-ouro representativo do ChatBulário. Não confunda testes de software com validação clínica.

## Limitações e validação necessária

1. O F1 publicado do OpenMed-BR não é uma garantia de desempenho no ChatBulário.
2. A extração de relações por LLM é assistida e sujeita a alucinações; a implementação rejeita relações sem menções e evidência textual literal, mas ainda exige revisão.
3. A pontuação exibida é um indicador heurístico de suporte, não uma probabilidade clínica calibrada.
4. A geração de Cypher usa allowlist read-only; deve ser testada contra a versão concreta do Neo4j.
5. PageRank/Louvain/Leiden dependem do plugin GDS no servidor Neo4j.
6. Os resultados precisam de avaliação com anotações humanas antes de qualquer uso operacional ou clínico.

## Referências dos modelos

- OpenMed-BR Clinical NER Efficient: https://huggingface.co/BrazilAI/openmed-br-clinical-pt
- Ollama: https://ollama.com/
- Neo4j Graph Data Science: https://neo4j.com/docs/graph-data-science/current/
