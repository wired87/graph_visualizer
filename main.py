import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from typing import List, Dict, Any


class PlotAnimationWorker:
    """
    A worker class to generate a time-series plot animation from a list of
    dictionaries, based on specific keys, and save it to a local file.
    """

    def __init__(
            self,
            parent,
            data: List[Dict[str, Any]],

    ):
        """
        Initializes the worker.

        Args:
            data: The list of dictionaries, where each dict is one frame/step.
            keys_to_plot: A list of keys from the dicts to include in the plot.
            output_dir: The directory to save the output video file.
            filename: The name of the output video file.
        """
        self.data = data
        self.keys_to_plot = keys_to_plot
        #self.output_path = os.path.join(output_dir, filename)
        self.keys_to_pplot= {
            "FERMION": [
                "psi", "energy", "dmu", "dirac"
            ],
            "GAUGE": ["field_value", "dmu", "j_nu"],
            "HIGGS": ["h", "dmu"]
        }
        self.plot_keys = self.keys_to_pplot[parent]
        # Prepare data structure for plotting
        self.prepared_data: Dict[str, List[float]] = {k: [] for k in keys_to_plot}
        self.max_steps = len(data)
        self._prepare_data()

    def _prepare_data(self):
        """
        Extracts and organizes the data points for the specified keys.
        """
        for entry in self.data:
            for key in self.keys_to_plot:
                # Use .get() to handle cases where a key might be missing in a step
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    self.prepared_data[key].append(value)
                else:
                    # Append NaN or 0 if value is missing or invalid
                    self.prepared_data[key].append(np.nan)

    def create_and_save_animation(self):
        """
        Generates the Matplotlib animation and saves the video file.
        """
        if self.max_steps == 0 or not self.prepared_data:
            print("Error: No data or keys to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Initialize lines and text for the plot
        lines = {
            key: ax.plot([], [], label=key)[0] for key in self.keys_to_plot
        }

        # Create a text object to display the current step/frame values
        val_text = ax.text(0.9, 0.9, '', transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

        # Setup plot limits and labels
        ax.set_xlim(0, self.max_steps)

        # Calculate overall Y limits for consistent scale
        all_values = [v for k in self.keys_to_plot for v in self.prepared_data[k] if not np.isnan(v)]
        if all_values:
            y_min = min(all_values) - 0.1 * abs(min(all_values))
            y_max = max(all_values) + 0.1 * abs(max(all_values))
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 1)  # Default scale

        ax.set_xlabel("Time Step (Index)")
        ax.set_ylabel("Value")
        ax.set_title("Animation of Key Values Over Time")
        ax.legend(loc='lower left')

        def animate_frame(i):
            """Updates the plot for frame i."""

            # Update line data up to the current frame i
            for key in self.keys_to_plot:
                x_data = np.arange(i + 1)
                y_data = self.prepared_data[key][:i + 1]
                lines[key].set_data(x_data, y_data)

            # Update the text box with current frame values
            current_values = self.data[i]
            text_lines = ["Step: {}".format(i)]
            for key in self.keys_to_plot:
                text_lines.append(f"{key}: {current_values.get(key, 'N/A'):.4f}")
            val_text.set_text("\n".join(text_lines))

            # Return updated artists
            return list(lines.values()) + [val_text]

        # Create the animation
        ani = animation.FuncAnimation(
            fig,
            animate_frame,
            frames=self.max_steps,
            interval=200,  # Milliseconds per frame
            blit=True
        )

        # Save the animation
        try:
            # Requires 'ffmpeg' or 'imagemagick' installed on the system
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            ani.save(self.output_path, writer='ffmpeg', fps=10)
            print(f"Animation successfully saved to: {self.output_path}")
        except Exception as e:
            print(f"Error saving animation (requires ffmpeg): {e}")
            print("Falling back to GIF generation (slower, may still require imagemagick).")
            try:
                # Fallback to slower GIF writer
                ani.save(self.output_path.replace(".mp4", ".gif"), writer='pillow', fps=10)
                print(f"Animation saved as GIF to: {self.output_path.replace('.mp4', '.gif')}")
            except Exception as e_gif:
                print(f"Failed to save as GIF as well: {e_gif}")
                print("Showing plot instead (if running interactively).")
                plt.show()




if __name__ == '__main__':
    # ------------------- DEMO DATA (SOA to AOS Conversion) -------------------
    NUM_STEPS = 50
    t = np.arange(NUM_STEPS)

    # Structure of Arrays (SOA) approach: Each key has its own full array.
    soa_data = {
        "Key_A": 10 * np.sin(t * 0.2) + 20,
        "Key_B": 0.5 * t ** 1.2 + 5,
        "Key_C": np.random.rand(NUM_STEPS) * 5,
        "Step_Index": t
    }

    # Convert SOA to Array of Structures (AOS) for the worker input
    demo_data = []
    keys_list = list(soa_data.keys())

    for i in range(NUM_STEPS):
        entry = {}
        for key in keys_list:
            # Cast to float for consistency, using 'Step_Index' for the step value
            entry[key] = float(soa_data[key][i])
        demo_data.append(entry)

    # ------------------- WORKER EXECUTION -------------------

    # 1. Define which keys from the dicts to plot
    keys_to_plot = ["Key_A", "Key_B", "Key_C"]  # Plotting all three keys

    # 2. Define output location
    output_directory = "./animation_output"

    worker = PlotAnimationWorker(
        data=demo_data,
        keys_to_plot=keys_to_plot,
        output_dir=output_directory,
        filename="soa_key_value_plot.mp4"
    )

    worker.create_and_save_animation()