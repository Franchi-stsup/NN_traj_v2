
# --- Minimal Config for BART NUFFT Rosette Pipeline ---
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrajectoryConfig:
    fov: float = 224.0
    res: int = 50
    smp_freq: float = 2000.0
    dwell_time: float = 10.0
    over_smp_on: bool = True
    no_paddles: int = 79
    spec_res: int = 325
    os_factor: int = 2
    o1: float = 1.0
    o2: float = 1.0
    beta: float = 2 * 3.141592653589793

@dataclass
class SystemConfig:
    gamma: float = 42.575575

@dataclass
class ReconstructionConfig:
    oprPath: str = 'bart_files'
    cs_reg: float = 0.0027
    iterations: int = 30
    cg_iterations: int = 30
    use_norm: bool = False
    apply_filter: str = ''
    recon_type: str = ''
    k_factor: float = 1.0
    do_k0_correction: bool = False
    k0_zero_fill: int = 4
    do_b0_correction: bool = False
    b0_ref_channel: int = 27
    b0_cor_mode: str = 'lin'
    debug_on: bool = True
    is_water_ref: bool = False

@dataclass
class PipelineConfig:
    input_image_path: str = "img_src/T2_MRI_of_Human_Brain_1024.jpg"
    output_dir: str = "output"
    plots_dir: str = "plots"
    tmp_dir: str = "tmp"
    save_intermediate: bool = True
    show_plots: bool = True
    save_plots: bool = True
    trajectory: TrajectoryConfig = TrajectoryConfig()
    reconstruction: ReconstructionConfig = ReconstructionConfig()
    system: SystemConfig = SystemConfig()

def create_default_config() -> PipelineConfig:
    return PipelineConfig()
