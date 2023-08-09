from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
import numpy as np

from utils import Unormalize
def compare_nie(target, input):
    NIE = np.abs(np.round(Unormalize(target) * 255.0) - np.round(Unormalize(input) * 255.0)).mean()
    return NIE

class Metrics():
    def __init__(self):
        self.mses = []
        self.psnrs = []
        self.ssims = []
        self.nies = []
        
    def update(self, target, predicted):
        assert len(target.shape) == 3 and len(predicted.shape) == 3
        self.mses.append(compare_mse(target, predicted))
        self.psnrs.append(compare_psnr(target, predicted))
        self.ssims.append(compare_ssim(target, predicted, data_range=1))
        self.nies.append(compare_nie(target, predicted))
        
    def average(self):
        return {"mse": np.mean(self.mses),
                "psnr": np.mean(self.psnrs),
                "ssim": np.mean(self.ssims),
                "nie": np.mean(self.nies)}