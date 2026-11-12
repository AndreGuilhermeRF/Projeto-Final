"""Renderização interativa do grafo completo usando vis-network."""

import json
import logging
import uuid

logger = logging.getLogger(__name__)

COLORS = {
    "Drug": "#4C78A8",
    "Condition": "#F58518",
    "AdverseEffect": "#E45756",
    "DrugClass": "#72B7B2",
    "Procedure": "#54A24B",
    "Anatomy": "#B279A2",
}


def graph_html(
    store,
    edges,
    highlight=(),
    height=650,
    include_analytics=False,
):
    """Gera a visualização completa do grafo."""

    edges = edges or []
    highlight = {str(value) for value in (highlight or [])}

    pagerank = {}
    community = {}

    if include_analytics:
        try:
            analytics = store.analytics() or {}
            pagerank = analytics.get("pagerank") or {}
            community = analytics.get("community") or {}
        except Exception:
            logger.exception("Falha ao carregar analytics.")

    # Recupera todos os nós em uma única consulta.
    try:
        raw_nodes = store.all_nodes() or []
    except Exception:
        logger.exception("Falha ao recuperar todos os nós.")
        raw_nodes = []

    # Normaliza os IDs dos nós.
    node_map = {}

    for node in raw_nodes:
        if not isinstance(node, dict):
            continue

        node_id = node.get("id")

        if node_id is None:
            continue

        visual_id = str(node_id)
        node_map[visual_id] = node

    nodes = []

    for visual_id, node in node_map.items():
        node_type = node.get("type") or "Entity"
        node_name = node.get("name") or visual_id

        highlighted = visual_id in highlight

        nodes.append({
            "id": visual_id,
            "label": str(node_name),
            "title": (
                "{}\nID: {}\nPageRank: {:.4f}\nComunidade: {}"
            ).format(
                node_type,
                visual_id,
                float(pagerank.get(visual_id, 0) or 0),
                community.get(visual_id, "-"),
            ),
            "group": str(node_type),
            # O tamanho do nó acompanha o PageRank (vis-network scaling).
            "value": float(pagerank.get(visual_id, 0) or 0) or 0.0001,
            "color": {
                "background": COLORS.get(node_type, "#999999"),
                "border": "#111111" if highlighted else "#ffffff",
            },
            "borderWidth": 4 if highlighted else 1,
            "font": {
                "color": "#111111",
                "size": 12,
            },
        })

    # Recupera as arestas cujos nós existem no grafo.
    graph_edges = []

    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue

        source = edge.get("src")
        target = edge.get("dst")

        if source is None or target is None:
            continue

        source_id = str(source)
        target_id = str(target)

        if source_id not in node_map or target_id not in node_map:
            continue

        relation = str(edge.get("rel") or "RELATED_TO")

        evidence = edge.get("evidence") or []
        evidence_text = ""

        if isinstance(evidence, list) and evidence:
            first = evidence[0]

            if isinstance(first, dict):
                evidence_text = first.get("text", "")
            else:
                evidence_text = str(first)

        graph_edges.append({
            "id": index,
            "from": source_id,
            "to": target_id,
            "label": relation.lower(),
            "arrows": "" if relation == "INTERACTS_WITH" else "to",
            "dashes": edge.get("certainty") == "hypothetical",
            "color": (
                "#d62728"
                if edge.get("negated", False)
                else "#888888"
            ),
            "title": str(evidence_text),
            "font": {
                "size": 9,
                "align": "middle",
                "background": "#ffffff",
            },
        })

    data = json.dumps(
        {
            "nodes": nodes,
            "edges": graph_edges,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")

    container_id = "graph_" + uuid.uuid4().hex
    status_id = container_id + "_status"

    try:
        height = max(300, int(height))
    except (TypeError, ValueError):
        height = 650

    return """
<div id="{container_id}"
     style="
        width:100%;
        height:{height}px;
        border:1px solid #ddd;
        border-radius:8px;
        background:#ffffff;
        position:relative;
        overflow:hidden;
     ">

    <div id="{status_id}"
         style="
            position:absolute;
            top:10px;
            left:10px;
            z-index:2;
            color:#555;
            font:13px Arial,sans-serif;
            background:rgba(255,255,255,0.9);
            padding:6px 9px;
            border-radius:5px;
         ">
        Preparando grafo...
    </div>
</div>

<script>
(function() {{
    const container = document.getElementById("{container_id}");
    const status = document.getElementById("{status_id}");
    const data = {data};

    let network = null;

    function error(message) {{
        status.style.color = "#b00020";
        status.textContent = message;
    }}

    function render() {{
        try {{
            if (!window.vis || !window.vis.Network) {{
                error("Biblioteca vis-network indisponível.");
                return;
            }}

            if (!data.nodes.length) {{
                error("Nenhum nó foi recuperado do Neo4j.");
                return;
            }}

            const nodeData = new vis.DataSet(data.nodes);
            const edgeData = new vis.DataSet(data.edges);

            status.textContent =
                "Renderizando " + data.nodes.length +
                " nós e " + data.edges.length +
                " relacionamentos...";

            network = new vis.Network(
                container,
                {{
                    nodes: nodeData,
                    edges: edgeData
                }},
                {{
                    autoResize: true,

                    physics: {{
                        enabled: true,
                        solver: "barnesHut",
                        stabilization: {{
                            enabled: true,
                            iterations: 80,
                            updateInterval: 20,
                            fit: true
                        }},
                        barnesHut: {{
                            gravitationalConstant: -3000,
                            centralGravity: 0.15,
                            springLength: 100,
                            springConstant: 0.025,
                            damping: 0.12,
                            avoidOverlap: 0.2
                        }},
                        minVelocity: 0.75
                    }},

                    interaction: {{
                        hover: true,
                        navigationButtons: true,
                        keyboard: true,
                        zoomView: true,
                        dragView: true,
                        dragNodes: true,
                        multiselect: true
                    }},

                    nodes: {{
                        shape: "dot",
                        size: 10,
                        scaling: {{
                            min: 5,
                            max: 18
                        }}
                    }},

                    edges: {{
                        smooth: {{
                            enabled: false
                        }},
                        width: 1,
                        selectionWidth: 2
                    }},

                    groups: {{
                        Drug: {{
                            color: "#4C78A8"
                        }},
                        Condition: {{
                            color: "#F58518"
                        }},
                        AdverseEffect: {{
                            color: "#E45756"
                        }},
                        DrugClass: {{
                            color: "#72B7B2"
                        }},
                        Procedure: {{
                            color: "#54A24B"
                        }},
                        Anatomy: {{
                            color: "#B279A2"
                        }}
                    }}
                }}
            );

            network.on("stabilizationProgress", function(params) {{
                const percent = Math.round(
                    100 * params.iterations / params.total
                );

                status.textContent =
                    "Estabilizando grafo: " + percent + "%";
            }});

            network.once(
                "stabilizationIterationsDone",
                function() {{
                    network.setOptions({{
                        physics: {{
                            enabled: false
                        }}
                    }});

                    network.fit({{
                        animation: {{
                            duration: 500,
                            easingFunction: "easeInOutQuad"
                        }}
                    }});

                    status.textContent =
                        data.nodes.length + " nós e " +
                        data.edges.length + " relacionamentos";
                }}
            );

            network.on("doubleClick", function(params) {{
                if (params.nodes.length > 0) {{
                    network.focus(params.nodes[0], {{
                        scale: 1.5,
                        animation: true
                    }});
                }}
            }});

            network.on("selectNode", function(params) {{
                const selected = params.nodes[0];

                if (selected) {{
                    const node = nodeData.get(selected);

                    status.textContent =
                        node.label + " (" + node.group + ")";
                }}
            }});

            network.on("deselectNode", function() {{
                status.textContent =
                    data.nodes.length + " nós e " +
                    data.edges.length + " relacionamentos";
            }});

        }} catch (exception) {{
            console.error(exception);
            error("Erro na visualização: " + exception.message);
        }}
    }}

    function loadLibrary() {{
        if (window.vis && window.vis.Network) {{
            render();
            return;
        }}

        const script = document.createElement("script");

        script.src =
            "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js";

        script.onload = render;

        script.onerror = function() {{
            error("Falha ao carregar vis-network. Verifique a internet.");
        }};

        document.head.appendChild(script);
    }}

    loadLibrary();
}})();
</script>
""".format(
        container_id=container_id,
        status_id=status_id,
        height=height,
        data=data,
    )