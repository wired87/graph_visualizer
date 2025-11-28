import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from typing import Dict, Any, List
try:
    plt.switch_backend('Agg')
except ImportError:
    pass


# 3. Animation Creator Class
class AnimationCreator:
    """// Generates and saves a Matplotlib animation simulating field cluster evolution."""

    def __init__(self, ):
        # DO USE A GENERAL PYTHON CLASS AND NOT A RAY REMMOTE
        self.pixel_nodes_init = self.g.get_nodes(filter_key="type", filter_value=["PIXEL"], data=True)
        self.output_path = os.path.join(OUTPUT_DIR, "field_animation.mp4")

        # Prepare output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"🖼️ Animation Creator initialized. Target MP4: {self.output_path}")

    def generate_data_cluster(self, frames: int = 20) -> List[List[Dict[str, Any]]]:
        """
        // generate list of data entries (representing a cluster) in n dim format (e.g. 4 array, float or 4,4 matrice) ->
        // Simulates the time evolution of field data across multiple PIXELs (frames).
        """
        print(f"📊 Generating {frames} frames of simulated field data...")
        time_series_data = []

        # Deep copy the initial nodes to track evolution without modifying the initial state
        current_state_nodes = {nid: self.g.get_node(nid).copy() for nid in self.g._nodes.keys()}

        for frame in range(frames):
            current_frame_pixels = []

            for p_node in self.pixel_nodes_init:
                p_nid = p_node['nid']
                total_frame_energy = 0

                # Access the current state of fields linked to this PIXEL
                for f_nid in p_node["field_nids"]:
                    field_state = current_state_nodes[f_nid]

                    # 1. Simulate field evolution (n dim format update)
                    old_value = field_state[self.field_key]

                    # Simple sinusoidal/random time evolution
                    noise = np.random.normal(0, 0.05, size=np.shape(old_value))
                    new_value = old_value + noise * np.sin(frame * 0.5)

                    # 2. Calculate new energy (norm of the N-dim data)
                    new_energy = np.linalg.norm(new_value.flat)
                    total_frame_energy += new_energy

                    # Update node state for the next frame
                    field_state[self.field_key] = new_value
                    field_state["current_energy"] = new_energy

                # Store the PIXEL's aggregated state for this frame
                current_frame_pixels.append({
                    "nid": p_nid,
                    "pos": p_node["pos"],
                    "total_energy": total_frame_energy,
                    "field_ids": p_node["field_nids"]
                })

            time_series_data.append(current_frame_pixels)

        print("✅ Data generation complete.")
        return time_series_data

    def create_and_save_animation(self, field_data_cluster: List[List[Dict[str, Any]]]):
        """
        // create matplot animation by looping though all list entries -> save as mp4 file
        // Renders PIXEL clusters colored by energy strength.
        """
        print(f"🎥 Creating 3D animation (Frames: {len(field_data_cluster)})...")

        if not field_data_cluster:
            print("❌ Cannot create animation: No data frames generated.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Calculate maximum energy for color normalization (fixed across all frames)
        max_energy = max(
            p['total_energy']
            for frame in field_data_cluster
            for p in frame
        ) or 1.0

        # Initial data for plot setup
        initial_frame = field_data_cluster[0]
        xs = [p['pos'][0] for p in initial_frame]
        ys = [p['pos'][1] for p in initial_frame]
        zs = [p['pos'][2] for p in initial_frame]
        colors = [p['total_energy'] / max_energy for p in initial_frame]

        # Plot PIXEL points (representing the sub cluster)
        scat = ax.scatter(xs, ys, zs,
                          c=colors,
                          s=100,
                          cmap='jet',  # Use 'jet'. Blue is low energy, red is high.
                          alpha=0.8)

        # Set fixed limits based on mock data range
        ax.set_xlabel('X (Space)')
        ax.set_ylabel('Y (Space)')
        ax.set_zlabel('Z (Space)')
        ax.set_title("Quantenfeld-Cluster Evolution")
        ax.set_xlim([-12, 12])
        ax.set_ylim([-12, 12])
        ax.set_zlim([-12, 12])

        # Color bar logic: The strength of the blue field color you will render by its "energy" value.
        cbar = fig.colorbar(scat, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Total Field Energy (Intensity)')

        def update(frame_idx):
            current_frame = field_data_cluster[frame_idx]

            xs_f = [p['pos'][0] for p in current_frame]
            ys_f = [p['pos'][1] for p in current_frame]
            zs_f = [p['pos'][2] for p in current_frame]

            # Map energy to color (0 to 1 normalized)
            colors_f = [p['total_energy'] / max_energy for p in current_frame]

            # Update Scatter Data
            scat._offsets3d = (xs_f, ys_f, zs_f)
            scat.set_array(np.array(colors_f))
            ax.set_title(f"Quantenfeld-Cluster Evolution (Frame: {frame_idx + 1})")

            return scat,

        # Create the Animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(field_data_cluster),
            interval=100,
            blit=False
        )

        # Save the animation
        writer = animation.FFMpegWriter(fps=10)
        try:
            # save as mp4 file under "data/"
            ani.save(self.output_path, writer=writer, dpi=150)
            print(f"💾 Animation saved successfully to: {self.output_path}")
        except Exception as e:
            print(f"❌ Failed to save MP4. Ensure FFMpeg is installed and in PATH. Error: {e}")
        finally:
            plt.close(fig)

    def run_test_workflow(self, frames: int = 20):
        """// Executes the entire workflow."""
        print("\n--- STARTING ANIMATION CREATOR TEST ---")

        # 1. Generate N-dim data entries
        data_cluster = self.generate_data_cluster(frames=frames)

        # 2. Create and save animation (Matplotlib loop)
        self.create_and_save_animation(data_cluster)

        print("--- ANIMATION CREATOR TEST FINISHED ---")


if __name__ == '__main__':
    # 1. Initialize Mock GUtils

    # 2. Initialize the Animation Creator Test Class (General Python Class)
    creator_test = AnimationCreator(
    )

    # 3. Run the complete test workflow
    creator_test.run_test_workflow(frames=20)