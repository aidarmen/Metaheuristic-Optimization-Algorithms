"""
Interactive PSO Visualization Module
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from typing import Callable, Optional, List, Dict, Any
import time

# Optional plotly import
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Create dummy objects to avoid AttributeError in type hints
    class DummyGo:
        class Figure:
            pass
        class Scatter:
            pass
        class Bar:
            pass
        class Surface:
            pass
    go = DummyGo()
    def make_subplots(*args, **kwargs):
        raise ImportError("Plotly is required. Install it with: pip install plotly")


class InteractivePSOVisualizer:
    """
    Interactive visualization for Particle Swarm Optimization.
    """
    
    def __init__(
        self,
        objective_func: Callable,
        bounds: np.ndarray,
        resolution: int = 100
    ):
        """
        Initialize interactive visualizer.
        
        Parameters:
        -----------
        objective_func : Callable
            Objective function to visualize
        bounds : np.ndarray
            Bounds for each dimension, shape (n_dim, 2)
        resolution : int
            Resolution for surface plot
        """
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.resolution = resolution
        self.n_dim = len(bounds)
        
        if self.n_dim != 2:
            raise ValueError("Visualization currently supports only 2D functions")
        
        # Create mesh for surface plot
        x_range = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
        y_range = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
        self.X, self.Y = np.meshgrid(x_range, y_range)
        self.Z = np.zeros_like(self.X)
        
        for i in range(resolution):
            for j in range(resolution):
                self.Z[i, j] = self.objective_func(np.array([self.X[i, j], self.Y[i, j]]))
        
        self.fig = None
        self.ax = None
        self.animation = None
        
    def create_surface_plot(self, show_plotly: bool = True) -> Any:
        """
        Create 3D surface plot using Plotly.
        
        Parameters:
        -----------
        show_plotly : bool
            Whether to show the plotly figure
        
        Returns:
        --------
        go.Figure : Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for this function. Install it with: pip install plotly")
        
        fig = go.Figure(data=[go.Surface(
            x=self.X,
            y=self.Y,
            z=self.Z,
            colorscale='Viridis',
            showscale=True
        )])
        
        fig.update_layout(
            title='Objective Function Surface',
            scene=dict(
                xaxis_title='X1',
                yaxis_title='X2',
                zaxis_title='Fitness',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        if show_plotly:
            fig.show()
        
        return fig
    
    def plot_convergence(
        self,
        history: Dict,
        methods: List[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot convergence curves for one or more methods.
        
        Parameters:
        -----------
        history : Dict
            History dictionary with 'global_best_fitness' or 'fitness' key
        methods : List[str]
            List of method names for comparison
        show_plot : bool
            Whether to display the plot
        
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(history, dict) and 'global_best_fitness' in history:
            # PSO history
            iterations = range(len(history['global_best_fitness']))
            ax.plot(iterations, history['global_best_fitness'], 
                   label='PSO', linewidth=2, marker='o', markersize=3)
        elif isinstance(history, dict) and 'fitness' in history:
            # Other methods
            iterations = range(len(history['fitness']))
            ax.plot(iterations, history['fitness'], 
                   label=methods[0] if methods else 'Method', 
                   linewidth=2, marker='o', markersize=3)
        elif isinstance(history, list):
            # Multiple methods
            for i, h in enumerate(history):
                method_name = methods[i] if methods and i < len(methods) else f'Method {i+1}'
                if 'global_best_fitness' in h:
                    iterations = range(len(h['global_best_fitness']))
                    ax.plot(iterations, h['global_best_fitness'], 
                           label=method_name, linewidth=2, marker='o', markersize=3)
                elif 'fitness' in h:
                    iterations = range(len(h['fitness']))
                    ax.plot(iterations, h['fitness'], 
                           label=method_name, linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Fitness', fontsize=12)
        ax.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Check if all values are positive before using log scale
        all_values_positive = True
        if isinstance(history, dict):
            if 'global_best_fitness' in history:
                if np.any(np.array(history['global_best_fitness']) <= 0):
                    all_values_positive = False
            elif 'fitness' in history:
                if np.any(np.array(history['fitness']) <= 0):
                    all_values_positive = False
        elif isinstance(history, list):
            for h in history:
                if 'global_best_fitness' in h:
                    if np.any(np.array(h['global_best_fitness']) <= 0):
                        all_values_positive = False
                        break
                elif 'fitness' in h:
                    if np.any(np.array(h['fitness']) <= 0):
                        all_values_positive = False
                        break
        
        if all_values_positive:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return fig
    
    def animate_particles(
        self,
        positions_history: List[np.ndarray],
        velocities_history: List[np.ndarray] = None,
        global_best_history: List[np.ndarray] = None,
        interval: int = 100,
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        Animate particle movement over iterations.
        
        Parameters:
        -----------
        positions_history : List[np.ndarray]
            List of particle positions for each iteration
        velocities_history : List[np.ndarray]
            List of particle velocities for each iteration
        global_best_history : List[np.ndarray]
            List of global best positions
        interval : int
            Animation interval in milliseconds
        save_path : Optional[str]
            Path to save animation (e.g., 'animation.gif')
        
        Returns:
        --------
        FuncAnimation : Animation object
        """
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Plot surface contour
        contour = self.ax.contour(self.X, self.Y, self.Z, levels=20, alpha=0.3, cmap='viridis')
        self.ax.clabel(contour, inline=True, fontsize=8)
        
        # Initialize scatter plots
        particles_scatter = self.ax.scatter([], [], c='blue', s=50, alpha=0.6, label='Particles')
        global_best_scatter = self.ax.scatter([], [], c='red', s=200, marker='*', 
                                             label='Global Best', zorder=5)
        
        # Initialize quiver with first frame data if available
        # Use a list to store quiver reference so we can modify it in animate function
        quiver_list = [None]
        if velocities_history and len(positions_history) > 0 and len(velocities_history) > 0:
            first_positions = positions_history[0]
            first_velocities = velocities_history[0]
            quiver_list[0] = self.ax.quiver(
                first_positions[:, 0], first_positions[:, 1],
                first_velocities[:, 0], first_velocities[:, 1],
                angles='xy', scale_units='xy', scale=1, 
                alpha=0.5, color='green', width=0.003
            )
        
        self.ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
        self.ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
        self.ax.set_xlabel('X1', fontsize=12)
        self.ax.set_ylabel('X2', fontsize=12)
        self.ax.set_title('Particle Swarm Optimization - Iteration 0', fontsize=14, fontweight='bold')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        def animate(frame):
            if frame >= len(positions_history):
                return
            
            positions = positions_history[frame]
            
            # Update particles
            particles_scatter.set_offsets(positions)
            
            # Update global best
            if global_best_history and frame < len(global_best_history):
                global_best_scatter.set_offsets([global_best_history[frame]])
            
            # Update velocities (quiver) - remove old and create new
            if quiver_list[0] is not None and velocities_history and frame < len(velocities_history):
                velocities = velocities_history[frame]
                # Remove old quiver
                quiver_list[0].remove()
                # Create new quiver with updated data
                quiver_list[0] = self.ax.quiver(
                    positions[:, 0], positions[:, 1],
                    velocities[:, 0], velocities[:, 1],
                    angles='xy', scale_units='xy', scale=1,
                    alpha=0.5, color='green', width=0.003
                )
            
            # Update title
            self.ax.set_title(f'Particle Swarm Optimization - Iteration {frame}', 
                            fontsize=14, fontweight='bold')
            
            artists = [particles_scatter, global_best_scatter]
            if quiver_list[0] is not None:
                artists.append(quiver_list[0])
            return artists
        
        self.animation = FuncAnimation(
            self.fig, animate, frames=len(positions_history),
            interval=interval, blit=False, repeat=True
        )
        
        if save_path:
            self.animation.save(save_path, writer='pillow', fps=10)
        
        plt.tight_layout()
        plt.show()
        
        return self.animation
    
    def create_3d_gif(
        self,
        positions_history: List[np.ndarray],
        global_best_history: List[np.ndarray] = None,
        save_path: str = 'pso_animation.gif',
        fps: int = 5,
        dpi: int = 100
    ) -> str:
        """
        Create an appealing 3D GIF visualization of PSO optimization.
        
        Parameters:
        -----------
        positions_history : List[np.ndarray]
            List of particle positions for each iteration
        global_best_history : List[np.ndarray]
            List of global best positions
        save_path : str
            Path to save the GIF file
        fps : int
            Frames per second for the GIF
        dpi : int
            Resolution (dots per inch)
        
        Returns:
        --------
        str : Path to saved GIF file
        """
        # Create 3D figure
        fig = plt.figure(figsize=(14, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        # Create appealing surface plot
        surf = ax.plot_surface(
            self.X, self.Y, self.Z,
            cmap='plasma',
            alpha=0.6,
            linewidth=0,
            antialiased=True,
            edgecolor='none'
        )
        
        # Add contour lines on the surface
        contour = ax.contour(
            self.X, self.Y, self.Z,
            zdir='z', offset=self.Z.min(),
            cmap='viridis',
            alpha=0.3,
            linewidths=1
        )
        
        # Set axis properties
        ax.set_xlabel('X1', fontsize=12, color='white', fontweight='bold')
        ax.set_ylabel('X2', fontsize=12, color='white', fontweight='bold')
        ax.set_zlabel('Fitness', fontsize=12, color='white', fontweight='bold')
        ax.tick_params(colors='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        ax.grid(True, alpha=0.2, color='white')
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
        
        # Initialize scatter plots with appealing colors
        particles_scatter = ax.scatter([], [], [], c='cyan', s=80, alpha=0.8, 
                                       edgecolors='white', linewidths=1.5, label='Particles')
        global_best_scatter = ax.scatter([], [], [], c='yellow', s=300, marker='*', 
                                         edgecolors='orange', linewidths=2, 
                                         label='Global Best', zorder=10)
        
        # Store particle trails for visual appeal
        trail_length = 3
        particle_trails = []
        
        def animate(frame):
            if frame >= len(positions_history):
                return
            
            ax.clear()
            ax.set_facecolor('black')
            
            # Recreate surface
            surf = ax.plot_surface(
                self.X, self.Y, self.Z,
                cmap='plasma',
                alpha=0.6,
                linewidth=0,
                antialiased=True,
                edgecolor='none'
            )
            
            # Recreate contour
            ax.contour(
                self.X, self.Y, self.Z,
                zdir='z', offset=self.Z.min(),
                cmap='viridis',
                alpha=0.3,
                linewidths=1
            )
            
            positions = positions_history[frame]
            
            # Calculate fitness for each particle (for z-coordinate)
            fitnesses = np.array([self.objective_func(pos) for pos in positions])
            
            # Update particle trails
            if frame < trail_length:
                particle_trails.append(positions.copy())
            else:
                particle_trails.append(positions.copy())
                if len(particle_trails) > trail_length:
                    particle_trails.pop(0)
            
            # Draw particle trails
            for trail_positions in particle_trails[:-1]:
                trail_fitnesses = np.array([self.objective_func(pos) for pos in trail_positions])
                ax.scatter(trail_positions[:, 0], trail_positions[:, 1], trail_fitnesses,
                          c='cyan', s=20, alpha=0.2, edgecolors='none')
            
            # Draw current particles with gradient colors based on fitness
            colors = plt.cm.viridis((fitnesses - fitnesses.min()) / (fitnesses.max() - fitnesses.min() + 1e-10))
            ax.scatter(positions[:, 0], positions[:, 1], fitnesses,
                      c=colors, s=100, alpha=0.9, edgecolors='white', linewidths=1.5)
            
            # Draw global best
            gb_fitness = None
            if global_best_history and frame < len(global_best_history):
                gb_pos = global_best_history[frame]
                gb_fitness = self.objective_func(gb_pos)
                ax.scatter([gb_pos[0]], [gb_pos[1]], [gb_fitness],
                          c='yellow', s=400, marker='*', 
                          edgecolors='orange', linewidths=2, zorder=10)
            
            # Set axis properties again (cleared by ax.clear())
            ax.set_xlabel('X1', fontsize=12, color='white', fontweight='bold')
            ax.set_ylabel('X2', fontsize=12, color='white', fontweight='bold')
            ax.set_zlabel('Fitness', fontsize=12, color='white', fontweight='bold')
            ax.tick_params(colors='white')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.xaxis.pane.set_alpha(0.1)
            ax.yaxis.pane.set_alpha(0.1)
            ax.zaxis.pane.set_alpha(0.1)
            ax.grid(True, alpha=0.2, color='white')
            
            # Set viewing angle with slight rotation
            ax.view_init(elev=30, azim=45 + frame * 0.5)
            
            # Set limits
            ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
            ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
            z_min, z_max = self.Z.min(), self.Z.max()
            ax.set_zlim(z_min, z_max)
            
            # Title with iteration number
            if gb_fitness is not None:
                title_text = f'Particle Swarm Optimization - Iteration {frame}\nBest Fitness: {gb_fitness:.4f}'
            else:
                title_text = f'Particle Swarm Optimization - Iteration {frame}'
            ax.set_title(title_text, fontsize=14, fontweight='bold', color='white', pad=20)
            
            return []
        
        # Create animation
        anim = FuncAnimation(
            fig, animate, frames=len(positions_history),
            interval=1000/fps, blit=False, repeat=True
        )
        
        # Save as GIF
        print(f"Saving GIF to {save_path}...")
        anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
        print(f"GIF saved successfully!")
        
        plt.close(fig)
        
        return save_path
    
    def plot_comparison(
        self,
        pso_history: Dict,
        other_histories: Dict[str, Dict],
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Create side-by-side comparison visualization.
        
        Parameters:
        -----------
        pso_history : Dict
            PSO optimization history
        other_histories : Dict[str, Dict]
            Dictionary of other method histories
        show_plot : bool
            Whether to display the plot
        
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 6))
        
        # Convergence plot
        ax1 = plt.subplot(1, 2, 1)
        
        # Plot PSO
        if 'global_best_fitness' in pso_history:
            iterations = range(len(pso_history['global_best_fitness']))
            ax1.plot(iterations, pso_history['global_best_fitness'], 
                   label='PSO', linewidth=2, marker='o', markersize=3, color='blue')
        
        # Plot other methods
        colors = ['red', 'green', 'orange', 'purple']
        for i, (method_name, history) in enumerate(other_histories.items()):
            if 'fitness' in history:
                iterations = range(len(history['fitness']))
                color = colors[i % len(colors)]
                ax1.plot(iterations, history['fitness'], 
                       label=method_name, linewidth=2, marker='s', 
                       markersize=3, color=color)
        
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Best Fitness', fontsize=12)
        ax1.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Check if all values are positive before using log scale
        all_values_positive = True
        if 'global_best_fitness' in pso_history:
            if np.any(np.array(pso_history['global_best_fitness']) <= 0):
                all_values_positive = False
        for history in other_histories.values():
            if 'fitness' in history:
                if np.any(np.array(history['fitness']) <= 0):
                    all_values_positive = False
                    break
        
        if all_values_positive:
            ax1.set_yscale('log')
        
        # Final solution comparison
        ax2 = plt.subplot(1, 2, 2)
        
        methods = ['PSO'] + list(other_histories.keys())
        final_fitness = []
        
        if 'global_best_fitness' in pso_history:
            final_fitness.append(pso_history['global_best_fitness'][-1])
        
        for history in other_histories.values():
            if 'fitness' in history:
                final_fitness.append(history['fitness'][-1])
        
        bars = ax2.bar(methods, final_fitness, color=['blue'] + colors[:len(other_histories)])
        ax2.set_ylabel('Final Best Fitness', fontsize=12)
        ax2.set_title('Final Solution Quality', fontsize=14, fontweight='bold')
        
        # Check if all final fitness values are positive before using log scale
        if np.all(np.array(final_fitness) > 0):
            ax2.set_yscale('log')
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            # Adjust label position based on whether value is positive or negative
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2e}', ha='center', va='bottom', fontsize=9)
            else:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2e}', ha='center', va='top', fontsize=9)
        
        # Use subplots_adjust instead of tight_layout to avoid warnings
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1, wspace=0.3)
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_interactive_plotly_comparison(
        self,
        pso_history: Dict,
        other_histories: Dict[str, Dict]
    ) -> Any:
        """
        Create interactive Plotly comparison visualization.
        
        Parameters:
        -----------
        pso_history : Dict
            PSO optimization history
        other_histories : Dict[str, Dict]
            Dictionary of other method histories
        
        Returns:
        --------
        go.Figure : Plotly figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for this function. Install it with: pip install plotly")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Convergence Comparison', 'Final Solution Quality'),
            specs=[[{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # Convergence plot
        if 'global_best_fitness' in pso_history:
            iterations = list(range(len(pso_history['global_best_fitness'])))
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=pso_history['global_best_fitness'],
                    mode='lines+markers',
                    name='PSO',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        colors = ['red', 'green', 'orange', 'purple']
        for i, (method_name, history) in enumerate(other_histories.items()):
            if 'fitness' in history:
                iterations = list(range(len(history['fitness'])))
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=history['fitness'],
                        mode='lines+markers',
                        name=method_name,
                        line=dict(color=color, width=2)
                    ),
                    row=1, col=1
                )
        
        # Final solution comparison
        methods = ['PSO'] + list(other_histories.keys())
        final_fitness = []
        
        if 'global_best_fitness' in pso_history:
            final_fitness.append(pso_history['global_best_fitness'][-1])
        
        for history in other_histories.values():
            if 'fitness' in history:
                final_fitness.append(history['fitness'][-1])
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=final_fitness,
                marker_color=['blue'] + colors[:len(other_histories)],
                text=[f'{f:.2e}' for f in final_fitness],
                textposition='outside',
                name='Final Fitness'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_yaxes(title_text="Best Fitness", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Method", row=1, col=2)
        fig.update_yaxes(title_text="Final Best Fitness", type="log", row=1, col=2)
        
        fig.update_layout(
            title_text="PSO vs Traditional Methods Comparison",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_parameter_adaptation(
        self,
        parameter_history: Dict,
        show_plot: bool = True
    ) -> plt.Figure:
        """
        Plot how adaptive parameters change over iterations.
        
        Parameters:
        -----------
        parameter_history : Dict
            Dictionary containing 'w', 'c1', 'c2' lists
        show_plot : bool
            Whether to display the plot
        
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        iterations = range(len(parameter_history['w']))
        
        # Plot inertia weight
        axes[0].plot(iterations, parameter_history['w'], 
                    linewidth=2, color='blue', marker='o', markersize=3)
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('Inertia Weight (w)', fontsize=12)
        axes[0].set_title('Adaptive Inertia Weight Evolution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([min(parameter_history['w']) * 0.9, max(parameter_history['w']) * 1.1])
        
        # Plot cognitive coefficient
        axes[1].plot(iterations, parameter_history['c1'], 
                    linewidth=2, color='green', marker='s', markersize=3)
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Cognitive Coefficient (c1)', fontsize=12)
        axes[1].set_title('Adaptive Cognitive Coefficient Evolution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([min(parameter_history['c1']) * 0.9, max(parameter_history['c1']) * 1.1])
        
        # Plot social coefficient
        axes[2].plot(iterations, parameter_history['c2'], 
                    linewidth=2, color='red', marker='^', markersize=3)
        axes[2].set_xlabel('Iteration', fontsize=12)
        axes[2].set_ylabel('Social Coefficient (c2)', fontsize=12)
        axes[2].set_title('Adaptive Social Coefficient Evolution', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([min(parameter_history['c2']) * 0.9, max(parameter_history['c2']) * 1.1])
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return fig

