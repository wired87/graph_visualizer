import numpy as np
import networkx as nx
import random
import io
import zipfile
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ray
import typing

# --- JAX/NumPy Kompatibilität (Da Sie JAX verwenden, aber Matplotlib NumPy erwartet)
# Wir verwenden np anstelle von jnp in der Rendering-Logik.
# Die in graph_visualization.py definierten Konstanten werden hier übernommen.

NODE_SIZE = 150
BACKGROUND_COLOR = "black"

if not ray.is_initialized():
    # Nur initialisieren, wenn es noch nicht geschehen ist
    ray.init()


# --- Die Hilfsfunktionen aus graph_visualization.py (mit np statt jnp) ---

def _compute_node_activity(graph):
    """Calculates the activity of each node as the sum of its edge weights."""
    # Verwende np.zeros
    node_activity = np.zeros(len(graph.nodes()))

    # Sicherstellen, dass Knoten IDs korrekt als Indizes verwendet werden können
    # Wir nehmen an, Knoten sind 0 bis N-1 (wie bei der Positions-Matrix)
    node_map = {node_id: i for i, node_id in enumerate(graph.nodes())}

    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 0)
        u_idx = node_map.get(u, -1)
        v_idx = node_map.get(v, -1)

        if u_idx != -1:
            node_activity[u_idx] += weight
        if v_idx != -1:
            node_activity[v_idx] += weight

    return node_activity


def _get_node_colors(node_activity, max_activity=1.0):
    """Calculates node colors from white (0 activity) to blue (max activity)."""
    if len(node_activity) == 0 or np.all(node_activity == 0):
        return [(1.0, 1.0, 1.0)] * len(node_activity)

    # Vermeide Division durch Null, falls Max Activity 0 ist
    max_activity = max_activity if max_activity > 1e-6 else 1.0

    normalized_activity = node_activity / max_activity
    colors = []
    for n in normalized_activity:
        # Interpolation von Weiß zu Blau: (1-n, 1-n, 1.0)
        colors.append((1.0 - n, 1.0 - n, 1.0))
    return colors


@ray.remote
def render_field_graph_3d_remote(
        G_data: typing.Dict[str, typing.Any],
        positions: np.ndarray,
        frames: int = 50,
        interval: int = 200,
        weight_change_range: float = 0.2,
        file_name: str = "animation.zip"
) -> bytes:
    """
    Kapselt den 3D-Graph-Renderer in einen Ray Task.
    Gibt die Animation als gezippte MP4-Bytes zurück.
    """

    # Rekonstruiere den Graphen aus dem serialisierten Dictionary
    G = nx.from_dict_of_dicts(G_data)
    num_nodes = len(G.nodes())
    if num_nodes == 0:
        print("Graph ist leer.")
        return b''

    # Initiale Gewichts-Ranges für konsistente Skalierung
    initial_edge_weights = [d.get('weight', 0) for _, _, d in G.edges(data=True)]
    edge_weight_min = min(initial_edge_weights) if initial_edge_weights else 0.0
    edge_weight_max = max(initial_edge_weights) if initial_edge_weights else 1.0
    if edge_weight_max == edge_weight_min: edge_weight_max += 1e-6

    initial_node_activity = _compute_node_activity(G)
    max_node_activity = np.max(initial_node_activity) if len(initial_node_activity) > 0 else 1.0
    if max_node_activity == 0: max_node_activity = 1.0

    # Figure und Achsen Setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_axis_off()

    # (Hier würde der Rest des Matplotlib/NetworkX Render-Codes folgen)
    # (Der Einfachheit halber simulieren wir den Update-Prozess kurz und konzentrieren uns auf die Ausgabe.)

    # --- SIMULIERTE RENDERING-SCHRITTE ---

    # Nodes initial rendern (nur notwendig für blit=True)
    node_colors = _get_node_colors(initial_node_activity, max_node_activity)
    sc = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        s=NODE_SIZE, c=node_colors,
    )

    # Edge Referenzen initialisieren
    edge_refs = {}
    for i, j, data in G.edges(data=True):
        # Nur Platzhalter, da wir den vollen Edge-Style-Code ausgelassen haben
        line, = ax.plot([0, 0], [0, 0], [0, 0], alpha=0)
        edge_refs[(i, j)] = line

    # --- Update-Funktion (SIMULIERTE LIVE-LOGIK) ---
    def update(frame):
        # 1. Daten-Änderungslogik (Hier ersetzen Sie mit Ihrer echten Graphen-Aktualisierung)
        for u, v, data in G.edges(data=True):
            if 'weight' in data:
                # Simuliere die Änderung (wie im Original-Code)
                delta = random.uniform(-weight_change_range, weight_change_range)
                data['weight'] = np.clip(data['weight'] + delta, edge_weight_min, edge_weight_max)

        # 2. Update der visuellen Elemente (Nodes und Edges)
        current_node_activity = _compute_node_activity(G)
        new_node_colors = _get_node_colors(current_node_activity, max_node_activity)
        sc.set_facecolor(new_node_colors)

        # (Edge-Updates müssten hier auch erfolgen)

        # Matplotlib erfordert, dass alle geänderten Objekte zurückgegeben werden
        return sc, *list(edge_refs.values())

    # --- Animation und Speicherung ---
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

    # Speichern des Videos in einem BytesIO-Puffer
    mp4_buffer = io.BytesIO()

    # WARNUNG: FFMpegWriter muss auf dem System installiert sein!
    try:
        writer = animation.FFMpegWriter(fps=1000 / interval)
        ani.save(mp4_buffer, writer=writer, codec='h264')
        mp4_buffer.seek(0)

        # Gezippte MP4-Datei erstellen (um binäre Daten als Bytes sicher zurückzugeben)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(file_name.replace(".zip", ".mp4"), mp4_buffer.read())
        zip_buffer.seek(0)

        # Bereinigte Bytes als Rückgabe
        return zip_buffer.getvalue()

    except Exception as e:
        print(f"FATAL ERROR beim Speichern der Animation im Ray Task: {e}")
        return b''
    finally:
        plt.close(fig)  # Wichtig, um Matplotlib-Speicherlecks zu vermeiden


# --- 3. Testdaten Erzeugung und Live-Simulation ---

def run_live_view_simulation():
    """Erzeugt Graphen-Testdaten und ruft den Ray Task auf."""

    N = 10  # Anzahl der Knoten (Modular Dimension)
    frames = 20  # Anzahl der Zeitschritte

    # 1. Hartecodierte Knoten-Positionen (3D-Feld)
    positions = np.random.rand(N, 3) * 10

    # 2. Graph erstellen (nx.Graph ist pickelbar/serialisierbar)
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)

    # 3. Kanten mit initialen Gewichten hinzufügen
    for i in range(N):
        for j in range(i + 1, N):
            # Füge zufällige Startgewichte hinzu
            G.add_edge(i, j, weight=random.uniform(0.1, 2.0))

    # Konvertiere den Graphen in ein serialisierbares Dictionary für Ray
    G_data = nx.to_dict_of_dicts(G)

    print(f"Sende Graph mit {N} Knoten und {G.number_of_edges()} Kanten an Ray Task...")

    # 4. Ray Remote Task starten
    remote_ref = render_field_graph_3d_remote.remote(
        G_data=G_data,
        positions=positions,
        frames=frames,
        interval=100,
        file_name="simulated_workflow_animation.zip"
    )

    # 5. Blockiere und erhalte die Bytes der Animation
    # Dies simuliert den Abschluss der "Live-View"-Berechnung
    try:
        animation_zip_bytes = ray.get(remote_ref)

        if animation_zip_bytes:
            # Speichere die Bytes lokal ab (Simulation der Übergabe an das Frontend)
            output_file = "simulated_workflow_animation.zip"
            with open(output_file, "wb") as f:
                f.write(animation_zip_bytes)
            print(f"\n✅ Simulation erfolgreich. Animation gespeichert als: {output_file}")
            print("Sie müssen die ZIP-Datei entpacken, um die MP4-Datei zu sehen.")
        else:
            print("\n❌ Fehler: Ray Task konnte keine Animations-Bytes zurückgeben.")

    except ray.exceptions.RayTaskError as e:
        print(f"\n❌ Fehler im Ray Task: {e}")


# Führe die Simulation aus
run_live_view_simulation()