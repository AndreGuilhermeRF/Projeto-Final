
"""
Gera graphrag_med/data/chatbulario.json diretamente do Hugging Face.

Características:
- Baixa o dataset em modo streaming.
- Não carrega o dataset inteiro na memória.
- Ignora registros sem resposta.
- Filtra registros por fuzzy_score.
- Salva o JSON progressivamente.
- Usa arquivo temporário para evitar JSON final incompleto.
"""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from datetime import date, datetime

DATASET_ID = "walmeidadf/ChatBulario"

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    PROJECT_DIR / "graphrag_med" / "data" / "chatbulario.json"
)


def json_safe(value):
    """Converte valores para tipos compatíveis com JSON."""
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if hasattr(value, "item"):
        return json_safe(value.item())

    return str(value)


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def validate_record(row, min_score):
    """Valida os campos necessários ao GraphRAG Med."""

    question = clean_text(row.get("pergunta"))
    answer = clean_text(row.get("resposta"))

    if not answer:
        return False

    if not question:
        return False

    score = row.get("fuzzy_score")

    if score is not None:
        try:
            if float(score) < min_score:
                return False
        except (TypeError, ValueError):
            pass

    return True


def generate(output, split, limit, min_score):
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERRO: biblioteca datasets não instalada.\n"
            "Execute: pip install datasets"
        )
        sys.exit(1)

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GERAÇÃO DO CHATBULÁRIO")
    print("=" * 60)
    print("Dataset:", DATASET_ID)
    print("Split:", split)
    print("Destino:", output)
    print("Limite:", limit if limit else "Todos os registros")
    print("Fuzzy score mínimo:", min_score)
    print()
    print("[1/3] Conectando ao Hugging Face...")

    try:
        dataset = load_dataset(
            DATASET_ID,
            split=split,
            streaming=True,
        )
    except Exception as exc:
        print("Erro ao carregar dataset:", exc)
        sys.exit(1)

    print("[2/3] Gerando arquivo JSON...")
    print("O processamento pode demorar dependendo da conexão.")
    print()

    total_read = 0
    total_saved = 0
    total_skipped = 0

    # O arquivo temporário fica no mesmo diretório do destino.
    # Isso permite substituição atômica no mesmo sistema de arquivos.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".tmp",
            prefix="chatbulario_",
            dir=str(output.parent),
            delete=False,
        ) as temp_file:

            temp_path = Path(temp_file.name)

            temp_file.write("[\n")
            first_record = True

            for row in dataset:
                total_read += 1

                if not validate_record(row, min_score):
                    total_skipped += 1
                    continue

                record = json_safe(dict(row))

                # Preserva a origem do split.
                record.setdefault("split", split)

                if not first_record:
                    temp_file.write(",\n")

                json.dump(
                    record,
                    temp_file,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )

                first_record = False
                total_saved += 1

                if total_saved % 500 == 0:
                    print(
                        "Lidos: {} | Salvos: {} | Ignorados: {}".format(
                            total_read,
                            total_saved,
                            total_skipped,
                        )
                    )

                if limit and total_saved >= limit:
                    break

            temp_file.write("\n]\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())

        print()
        print("[3/3] Validando JSON...")

        # Confere a sintaxe antes de substituir o arquivo final.
        with temp_path.open("r", encoding="utf-8") as check_file:
            json.load(check_file)

        # Substituição atômica: o arquivo anterior permanece intacto
        # se ocorrer falha durante a geração ou validação.
        os.replace(str(temp_path), str(output))
        temp_path = None

        size_mb = output.stat().st_size / (1024 * 1024)

        print()
        print("=" * 60)
        print("GERAÇÃO CONCLUÍDA")
        print("=" * 60)
        print("Registros lidos:", total_read)
        print("Registros salvos:", total_saved)
        print("Registros ignorados:", total_skipped)
        print("Tamanho: {:.2f} MB".format(size_mb))
        print("Arquivo:", output)
        print()
        print("JSON validado com sucesso!")

    except KeyboardInterrupt:
        print("\nGeração interrompida pelo usuário.")
        print("O arquivo JSON anterior foi preservado.")

    except Exception as exc:
        print("\nERRO durante a geração:", exc)
        print("O arquivo JSON anterior foi preservado.")
        raise

    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Gera o chatbulario.json do GraphRAG Med."
    )

    parser.add_argument(
        "--split",
        default="train",
        help="Divisão do dataset. Padrão: train",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Caminho do arquivo JSON de saída.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Quantidade máxima de registros. 0 significa todos.",
    )

    parser.add_argument(
        "--min-fuzzy-score",
        type=float,
        default=80.0,
        help="Pontuação mínima de qualidade. Padrão: 80.",
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit não pode ser negativo.")

    generate(
        output=args.output,
        split=args.split,
        limit=args.limit,
        min_score=args.min_fuzzy_score,
    )


if __name__ == "__main__":
    main()
