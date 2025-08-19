import torchkbnufft as tkbn 
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_kb_nufft(mrData, kSpaceTrj, para):
    """
    Wrapper to run kbnufft directly in python with given parameters.
    Args:
        mrData: numpy array, k-space data
        kSpaceTrj: dict with keys 'kxx', 'kyy'
        para: dict of parameters
    Returns:
        reconMrsi: reconstructed image (pytorch tensor)
    """
    # TODO:
    #  o change mSize depending of FoV and maximum k-space radius
    #  o do image scaling to reference image to compensate missmatch
    #  o if input parameters are already pytorch gpu tensors, keep them as such
    #  o check image scaling to reference image

    # --- Prepare trajectory ---
    kxx = kSpaceTrj['kxx'] / torch.max(torch.abs(kSpaceTrj['kxx'])) * para['mSize']/2 * para['kFac']
    kyy = kSpaceTrj['kyy'] / torch.max(torch.abs(kSpaceTrj['kyy'])) * para['mSize']/2 * para['kFac']

    # print(f"Normalized k-space trajectory shapes: kxx={kxx.shape}, kyy={kyy.shape}")
    kxx = kxx.reshape(1, kxx.shape[0], kxx.shape[1])
    kyy = kyy.reshape(1, kyy.shape[0], kyy.shape[1])

    # print(f"Added a 3rd dimension to kxx and kyy: kxx={kxx.shape}, kyy={kyy.shape}")
    kspaceRSI = torch.concatenate([kxx, kyy], axis=0)
    # print(f"kspace RSI shape: {kspaceRSI.shape}")
    kspaceRSI = torch.concatenate([kspaceRSI, torch.zeros((1, kxx.shape[1], kxx.shape[2])).to(kspaceRSI.device) ], axis=0)

    trjNew = kspaceRSI[0,:,:] + 1j*kspaceRSI[1,:,:]
    trjNew = trjNew / torch.max(torch.abs(trjNew)) * torch.pi    

    trjNewReal = torch.zeros((2, trjNew.numel() )) 
    trjNewReal[0,:] = torch.real( torch.ravel( trjNew ) )
    trjNewReal[1,:] = torch.imag( torch.ravel( trjNew ) )

    # --- Prepare nufft operator ---
    nufft         = tkbn.KbNufft(im_size=(para['mSize'], para['mSize'])).to(kspaceRSI.device)
    nufft_adjoint = tkbn.KbNufftAdjoint(im_size=(para['mSize'], para['mSize'])).to(kspaceRSI.device)

    # para['mSize'] = 50
    # --- Calculate density compensation weights ---
    weights = tkbn.calc_density_compensation_function( trjNewReal.to(kspaceRSI.device), (para['mSize'], para['mSize']))

    # --- Reshape mrData ---
    mrData_reshaped = torch.ravel( mrData )
    mrData_reshaped = mrData_reshaped.reshape([1, 1, mrData_reshaped.numel() ])

    # --- Perform NUFFT adjoint operation ---
    reconImg = nufft_adjoint( mrData_reshaped*weights , trjNewReal.to( kspaceRSI.device ) )
    
    # plt.figure()
    # plt.imshow(np.abs( np.squeeze( reconImg.cpu().detach().numpy() ) )/np.sqrt(50)/2, cmap='jet')
    # plt.colorbar()

    # --- Directly return pytorch tensor for back propagation ---
    return reconImg