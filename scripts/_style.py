"""Shared matplotlib style for the MethodsX figures (serif, STIX maths)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def apply():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.labelsize": 12, "legend.fontsize": 9,
    })
