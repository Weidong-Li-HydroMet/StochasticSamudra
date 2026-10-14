"""
Standardized Scientific Plotting Configuration for Publication-Quality Figures

Usage:
    from plot_config import params, contourLevels, colormap, savefig_format
    plt.rcParams.update(params)
"""

import matplotlib.pyplot as plt
import subprocess
import warnings

# ============================================================================
# Font Size Configuration
# ============================================================================

# Standard scientific figure font sizes
SMALL_SIZE = 10    # for tick labels, annotations
MEDIUM_SIZE = 12   # for axis labels, legends  
LARGE_SIZE = 14    # for titles

# ============================================================================
# Plot Parameters Configuration
# ============================================================================

# Base parameters that work without LaTeX
base_params = {
    'figure.dpi': 100,               # Display DPI
    'savefig.dpi': 600,              # High DPI for saving
    'figure.facecolor': 'white',     # White figure background
    'axes.facecolor': 'white',       # White axes background
    
    # Font sizes
    'font.size': MEDIUM_SIZE,        # Default font size
    'axes.titlesize': LARGE_SIZE,    # Axes title size
    'axes.labelsize': MEDIUM_SIZE,   # Axes label size
    'xtick.labelsize': SMALL_SIZE,   # X-axis tick label size
    'ytick.labelsize': SMALL_SIZE,   # Y-axis tick label size
    'legend.fontsize': MEDIUM_SIZE,  # Legend font size
    'figure.titlesize': LARGE_SIZE,  # Figure title size
    
    # Line and marker properties
    'lines.linewidth': 1.5,          # Default line width
    'lines.markersize': 5,           # Default marker size
    'patch.linewidth': 0.5,          # Default patch line width
    
    # Tick properties
    'xtick.direction': 'in',         # Ticks point inward
    'ytick.direction': 'in',         # Ticks point inward
    'xtick.top': True,               # Show top ticks
    'xtick.bottom': True,            # Show bottom ticks
    'ytick.left': True,              # Show left ticks
    'ytick.right': True,             # Show right ticks
    'xtick.minor.visible': True,     # Show minor ticks
    'ytick.minor.visible': True,     # Show minor ticks
    
    # Grid
    'axes.grid': False,              # No grid by default
    'grid.alpha': 0.3,              # Grid transparency
    
    # Spines
    'axes.spines.top': True,         # Show top spine
    'axes.spines.bottom': True,      # Show bottom spine
    'axes.spines.left': True,        # Show left spine
    'axes.spines.right': True,       # Show right spine
    

}

# LaTeX-specific parameters (only applied if LaTeX is available)
latex_params = {
    'text.usetex': True,                                   # Use LaTeX for text rendering
    # 'pgf.texsystem': 'pdflatex',                         # Enable pgf backend using pdflatex
    # 'pgf.preamble': r'\usepackage{amsmath}',             # Enable LaTeX math for pgf backend
    'text.latex.preamble': r'\usepackage{amsmath}',        # Math symbols package
    'font.family': 'serif',                                # Serif font family (Helvetica and Arial are other common fonts in sans-serif font family)
    'font.serif': ['Computer Modern Roman'],               # Specific serif font - closest to LaTeX default font
    'mathtext.fontset': 'cm',                              # Computer Modern math
}


# ============================================================================
# Plotting Constants and Presets
# ============================================================================

# Contour plot settings
contourLevels = 100                    # High-quality contour levels (vs. default 20)
colormap = 'bwr'                       # Blue-white-red colormap (white = zero)

# File format settings
savefig_format = 'png'                 # Default save format (pdf for vector graphics)

# Default save directory (empty string = current directory)
SAVE_DIR = ''

# ============================================================================
# LaTeX Availability Check
# ============================================================================

def check_latex_available():
    """
    Check if LaTeX is available on the system using 'which latex'.
    
    Returns:
        bool: True if LaTeX is available, False otherwise
    """
    try:
        # Use 'which latex' to find LaTeX installation
        result = subprocess.run(['which', 'latex'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            latex_path = result.stdout.strip()
            print(f"✓ LaTeX found at: {latex_path}")
            return True
        else:
            return False
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False

# Check LaTeX availability
LATEX_AVAILABLE = check_latex_available()

if not LATEX_AVAILABLE:
    print("✗ LaTeX not found in PATH")
    warnings.warn(
        "LaTeX not found. Falling back to standard matplotlib fonts. "
        "For better typography, install LaTeX (see README for instructions).",
        UserWarning
    )

# Combine parameters based on LaTeX availability
if LATEX_AVAILABLE:
    params = {**base_params, **latex_params}
else:
    params = {**base_params, 
              'font.family': 'serif',           # Fallback serif font
              'mathtext.fontset': 'dejavuserif' # Fallback math font
              }
    
# ============================================================================
# Auto-apply settings when module is imported
# ============================================================================


# Print configuration info
print(f"🎨 Scientific plotting configuration loaded")
print(f"   LaTeX support: {'✓ Enabled' if LATEX_AVAILABLE else '✗ Disabled'}")
print(f"   Default save format: {savefig_format}")
print(f"   Contour levels: {contourLevels}")
print(f"   Colormap: {colormap}")