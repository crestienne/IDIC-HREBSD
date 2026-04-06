import os
import struct

from skimage import io
from scipy import ndimage
import numpy as np
from sympy import gamma
from skimage import exposure
import matplotlib.pyplot as plt


def _circular_mask(shape, radius=None, center=None):
    H, W = shape
    if center is None:
        cy, cx = H // 2, W // 2
    else:
        cy, cx = center
    if radius is None:
        radius = min(cy, cx, H - cy - 1, W - cx - 1)
    yy, xx = np.ogrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def _center_cross_mask(shape, half_width):
    H, W = shape
    cy, cx = H // 2, W // 2
    mask = np.ones((H, W), dtype=bool)
    mask[cy - half_width : cy + half_width, :] = False
    mask[:, cx - half_width : cx + half_width] = False
    return mask


class UP2:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.i = 0
        self.start_byte = np.int64(16) #start byte acutally defined in header (C editied on Jan 21, 2026)
        self.header()
        self.set_processing()

    def __len__(self):
        return self.nPatterns

    def __getitem__(self, i):
        return self.read_pattern(i, process=True)

    def __str__(self):
        return (
            f"UP2 file: {self.path}\n"
            f"Patterns: {self.nPatterns}\n"
            f"Pattern shape: {self.patshape}\n"
            f"File size: {self.filesize}\n"
            f"Bits per pixel: {self.bitsPerPixel}\n"
        )

    def __repr__(self):
        return self.__str__()

    def header(self):
        chunk_size = 4
        tmp = self.read(chunk_size)
        self.FirstEntryUpFile = struct.unpack("i", tmp)[0]
        tmp = self.read(chunk_size)
        sz1 = struct.unpack("i", tmp)[0] # pattern width, x 
        tmp = self.read(chunk_size)
        sz2 = struct.unpack("i", tmp)[0] # pattern height, y 
        tmp = self.read(chunk_size)
        self.bitsPerPixel = struct.unpack("i", tmp)[0]
        self.start_byte = self.bitsPerPixel #unclear where bits per pixel is being used so leaving for right now 
        sizeBytes = os.path.getsize(self.path) - 16
        self.filesize = str(round(sizeBytes / 1e6, 1)) + " MB"
        bytesPerPixel = 2
        self.nPatterns = int((sizeBytes / bytesPerPixel) / (sz1 * sz2))
        self.patshape = (sz1, sz2) #(width, height)


        self.pattern_bytes = np.int64(self.patshape[0] * self.patshape[1] * 2)

    def set_processing(
        self,
        low_pass_sigma: float = 0.0,
        high_pass_sigma: float = 30.0,
        truncate_std_scale: float = 3.0,
        mask_type: str = "circular",
        center_cross_half_width: int = 5,
        clahe_kernel: tuple = (5, 5),
        clahe_clip: float = 0.005,
        clahe_nbins: int = 256,
        flip_x: bool = False,
    ):
        """Set the parameters for processing the patterns.
        Values of 0.0 will skip the step.

        Args:
            low_pass_sigma (float): The sigma for the low pass filter. Roughly 1% of the image size works well.
            high_pass_sigma (float): The sigma for the high pass filter. Roughly 20% of the image size works well.
            truncate_std_scale (float): The number of standard deviations to truncate. 3.0 is a good value.
            mask_type (str): "circular", "center_cross", or None (no mask). Controls which mask is applied in process_pattern.
            center_cross_half_width (int): Half-width of the cross arms in pixels when mask_type="center_cross" (default 5 → 10 px total).
            clahe_kernel (tuple): Kernel size for CLAHE. Smaller = more local contrast. Default (5, 5).
            clahe_clip (float): Clip limit for CLAHE. Higher = more contrast. Default 0.005.
            clahe_nbins (int): Number of histogram bins for CLAHE. Default 256.
        """
        self.low_pass_sigma = low_pass_sigma
        self.high_pass_sigma = high_pass_sigma
        self.truncate_std_scale = truncate_std_scale
        self.mask_type = mask_type
        self.center_cross_half_width = center_cross_half_width
        self.clahe_kernel = clahe_kernel
        self.clahe_clip = clahe_clip
        self.clahe_nbins = clahe_nbins
        self.flip_x = flip_x
        print(f"Set UP2 pattern processing: low_pass_sigma={low_pass_sigma}, high_pass_sigma={high_pass_sigma}, truncate_std_scale={truncate_std_scale}, mask_type={mask_type}, clahe_kernel={clahe_kernel}, clahe_clip={clahe_clip}, flip_x={flip_x}")

    def read(self, chunks, i=None):
        """Read the next `chunks` bytes from the file. If `i` is not None, read from the current position."""
        if i is None:
            i = self.i
        with open(self.path, "rb") as upFile:
            upFile.seek(i)
            data = upFile.read(chunks)
        self.i += chunks
        return data

    def get_mask(self):
        """Return the boolean mask for the current mask_type, or None if no mask."""
        if self.mask_type == "circular":
            return _circular_mask(self.patshape)
        elif self.mask_type == "center_cross":
            return _center_cross_mask(self.patshape, self.center_cross_half_width)
        else:
            return None

    def read_pattern(self, i, process=False):
        # Read in the patterns
        seek_pos = np.int64(self.start_byte + np.int64(i) * self.pattern_bytes)
        buffer = self.read(chunks=self.pattern_bytes, i=seek_pos)
        pat = np.frombuffer(buffer, dtype=np.uint16).reshape(self.patshape) #order should be y,x
        if self.flip_x:
            pat = np.flipud(pat)  # flip about x axis (reverses rows)
        if process:
            pat = self.process_pattern(pat)
        return pat #pretty sure the patterns are in (x, y) format

    def read_patterns(self, idx=-1, process=False):
        if type(idx) == int:
            if idx != -1:
                return self.read_pattern(idx)
            else:
                idx = range(self.nPatterns)
        else:
            idx = np.asarray(idx)

        # Read in the patterns
        in_shape = idx.shape + self.patshape
        idx = idx.flatten()
        if process:
            pats = np.zeros(idx.shape + self.patshape, dtype=np.float32)
        else:
            pats = np.zeros(idx.shape + self.patshape, dtype=np.uint16)
        for i in range(idx.shape[0]):
            pats[i] = self.read_pattern(idx[i], process)
        return pats.reshape(in_shape)
    
    def process_pattern(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        """Cleans patterns by equalizing the histogram and normalizing.
        Applies a bandpass filter to the patterns and truncates the extreme values.
        Images will be in the range [0, 1].
        """

        circular_mask = _circular_mask
        center_cross_mask = _center_cross_mask

        def masked_normalize(image, mask):
            image = image.astype(np.float32).copy()
            vals = image[mask]
            vmin = vals.min()
            vmax = vals.max()
            if vmax > vmin:
                image[mask] = (image[mask] - vmin) / (vmax - vmin)
            else:
                image[mask] = 0.0
            image[~mask] = 0.0
            return image

        def to_uint8(image, mask):
            out = masked_normalize(image, mask)
            return np.around(255 * out).astype(np.uint8)

        # Correct dtype
        img = img.astype(np.float32)

        # Build mask (None = full image, no masking)
        if self.mask_type == "circular":
            mask = circular_mask(img.shape)
        elif self.mask_type == "center_cross":
            mask = center_cross_mask(img.shape, self.center_cross_half_width)
        else:  # None or anything else → no mask
            mask = np.ones(img.shape, dtype=bool)

        # Normalize inside mask
        img = masked_normalize(img, mask)

        img_noprocessing = to_uint8(img, mask)

        def masked_gaussian(image, mask, sigma):
            """Gaussian filter that treats masked pixels as non-existent (normalized convolution).
            Masked pixels contribute zero weight so they never bias the result."""
            float_mask = mask.astype(np.float32)
            weighted = ndimage.gaussian_filter(image * float_mask, sigma)
            weights  = ndimage.gaussian_filter(float_mask, sigma)
            return np.where(weights > 0, weighted / weights, 0.0)

        # Low pass filter
        if self.low_pass_sigma > 0:
            img = masked_gaussian(img, mask, self.low_pass_sigma)
            img[~mask] = 0.0

        img_lowpass = to_uint8(img, mask)

        # High pass filter
        if self.high_pass_sigma > 0:
            background = masked_gaussian(img, mask, self.high_pass_sigma)
            img = img - background
            img[~mask] = 0.0

        img_highpass = to_uint8(img, mask)

        # ---- adaptive histogram equalization ----
        # CLAHE has no native mask support, so inpaint the masked region with
        # Gaussian-interpolated values from valid neighbours before running it,
        # then zero it out again afterwards.
        img_clahe_in = masked_gaussian(img, mask, sigma=max(self.clahe_kernel))
        img_clahe_in[mask] = img[mask]  # leave valid pixels unchanged
        img = exposure.equalize_adapthist(
            img_clahe_in,
            kernel_size=self.clahe_kernel,
            clip_limit=self.clahe_clip,
            nbins=self.clahe_nbins,
        ).astype(np.float32)
        img[~mask] = 0.0

        img_CLAHE = to_uint8(img, mask)

        # Truncate step using only masked region
        if self.truncate_std_scale > 0:
            vals = img[mask]
            mean, std = vals.mean(), vals.std()
            img[mask] = np.clip(
                img[mask],
                mean - self.truncate_std_scale * std,
                mean + self.truncate_std_scale * std,
            )
            img[~mask] = 0.0

        img_truncated = to_uint8(img, mask)

        # Re-normalize inside mask
        img = masked_normalize(img, mask)
        img_renorm = to_uint8(img, mask)

        # Gamma correction
        gamma_val = 0.66
        img[mask] = img[mask] ** gamma_val
        img[~mask] = 0.0

        img_gamma = to_uint8(img, mask)

        # -------------------------
        # Save all steps as subplots
        # -------------------------
        # images = [
        #     img_noprocessing,
        #     img_lowpass,
        #     img_highpass,
        #     img_CLAHE,
        #     img_truncated,
        #     img_renorm,
        #     img_gamma,
        #     mask.astype(np.uint8) * 255,
        # ]

        # titles = [
        #     "Normalized Input",
        #     f"Low-pass\nsigma={self.low_pass_sigma}",
        #     f"High-pass\nsigma={self.high_pass_sigma}",
        #     f"CLAHE\nkernel={clahe_kernel}, clip={clahe_clip}",
        #     f"Truncate\nstd scale={self.truncate_std_scale}",
        #     "Re-normalized",
        #     f"Gamma Corrected\ngamma={gamma_val}",
        #     "Circular Mask",
        # ]

        # fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        # axes = axes.ravel()

        # for i, (image_step, title) in enumerate(zip(images, titles)):
        #     axes[i].imshow(image_step, cmap="gray")
        #     axes[i].set_title(title)
        #     axes[i].axis("off")

        # plt.tight_layout()
        # plt.savefig("debug/pattern_processing_steps.png", dpi=300, bbox_inches="tight")
        # plt.close(fig)

        return img

    def plot_parameter_sweep(
        self,
        pattern_idx: int = 0,
        high_pass_sigmas: list = None,
        clahe_kernels: list = None,
        save_dir: str = "debug",
    ):
        """Process one pattern across all combinations of high_pass_sigma and clahe_kernel
        and save a single grid JPG to save_dir.

        Layout: rows = clahe_kernel, columns = high_pass_sigma.
        Row labels on the left, column labels on top.

        Args:
            pattern_idx (int): Index of the pattern to use for the sweep.
            high_pass_sigmas (list): Sigma values (columns). Defaults to [5, 10, 20, 30, 50, 80].
            clahe_kernels (list): Kernel sizes (rows). Defaults to [(3,3),(5,5),(8,8),(12,12),(16,16),(24,24)].
            save_dir (str): Directory to save the output JPG. Default "debug".
        """
        if high_pass_sigmas is None:
            high_pass_sigmas = [5, 10, 20, 30, 50, 80]
        if clahe_kernels is None:
            clahe_kernels = [(3, 3), (5, 5), (8, 8), (12, 12), (16, 16), (24, 24)]

        os.makedirs(save_dir, exist_ok=True)
        raw = self.read_pattern(pattern_idx, process=False)

        def _process(img, high_pass_sigma, clahe_kernel):
            orig_hp = self.high_pass_sigma
            orig_ck = self.clahe_kernel
            self.high_pass_sigma = high_pass_sigma
            self.clahe_kernel = clahe_kernel
            result = self.process_pattern(img.copy())
            self.high_pass_sigma = orig_hp
            self.clahe_kernel = orig_ck
            return result

        nrows = len(clahe_kernels)
        ncols = len(high_pass_sigmas)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)

        for r, kernel in enumerate(clahe_kernels):
            for c, sigma in enumerate(high_pass_sigmas):
                result = _process(raw, sigma, kernel)
                axes[r, c].imshow(result, cmap="gray", vmin=0, vmax=1)
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(f"hp_sigma={sigma}", fontsize=8)
            axes[r, 0].text(
                -0.05, 0.5, f"clahe={kernel}",
                fontsize=8, transform=axes[r, 0].transAxes,
                ha="right", va="center", rotation=90,
            )

        fig.suptitle(f"Parameter sweep  |  pattern {pattern_idx}", fontsize=11)
        plt.tight_layout()
        out_path = f"{save_dir}/sweep_combined.jpg"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    def process_pattern_version2(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        """Cleans patterns by equalizing the histogram and normalizing.
        Applies a bandpass filter to the patterns and truncates the extreme values.
        Images will be in the range [0, 1].

        Args:
            img (np.ndarray): The patterns to clean. (H, W)
            low_pass_sigma (float): The sigma for the low pass filter.
            high_pass_sigma (float): The sigma for the high pass filter.
            truncate_std_scale (float): The number of standard deviations to truncate.

        Returns:
            np.ndarray: The cleaned patterns. (H, W)
        """

        # Correct dtype
        img = img.astype(np.float32)

        # Normalize
        img = (img - img.min()) / (img.max() - img.min())

        img_noprocessing = img.copy()
        img_noprocessing = np.around(
            255 * (img_noprocessing - img_noprocessing.min())
            / (img_noprocessing.max() - img_noprocessing.min())
        ).astype(np.uint8)

        # Low pass filter
        if self.low_pass_sigma > 0:
            img = ndimage.gaussian_filter(img, self.low_pass_sigma)

        img_lowpass = img.copy()
        img_lowpass = np.around(
            255 * (img_lowpass - img_lowpass.min())
            / (img_lowpass.max() - img_lowpass.min())
        ).astype(np.uint8)

        # High pass filter
        if self.high_pass_sigma > 0:
            background = ndimage.gaussian_filter(img, self.high_pass_sigma)
            img = img - background

        img_highpass = img.copy()
        img_highpass = np.around(
            255 * (img_highpass - img_highpass.min())
            / (img_highpass.max() - img_highpass.min())
        ).astype(np.uint8)

                # ---- adaptive histogram equalization ----
        clahe_kernel = (16, 16)
        clahe_clip = 0.01
        clahe_nbins = 256

        img = exposure.equalize_adapthist(
            img,
            kernel_size=clahe_kernel,
            clip_limit=clahe_clip,
            nbins=clahe_nbins,
        ).astype(np.float32)

        img_CLAHE = img.copy()
        img_CLAHE = np.around(
            255 * (img_CLAHE - img_CLAHE.min())
            / (img_CLAHE.max() - img_CLAHE.min())
        ).astype(np.uint8)

        # Truncate step
        if self.truncate_std_scale > 0:
            mean, std = img.mean(), img.std()
            img = np.clip(
                img,
                mean - self.truncate_std_scale * std,
                mean + self.truncate_std_scale * std,
            )

        img_truncated = img.copy()
        img_truncated = np.around(
            255 * (img_truncated - img_truncated.min())
            / (img_truncated.max() - img_truncated.min())
        ).astype(np.uint8)

        # Re-normalize
        img = (img - img.min()) / (img.max() - img.min())

        img_renorm = img.copy()
        img_renorm = np.around(
            255 * (img_renorm - img_renorm.min())
            / (img_renorm.max() - img_renorm.min())
        ).astype(np.uint8)

        # Gamma correction
        gamma_val = 1.33
        img = img**gamma_val

        img_gamma = img.copy()
        img_gamma = np.around(
            255 * (img_gamma - img_gamma.min())
            / (img_gamma.max() - img_gamma.min())
        ).astype(np.uint8)

        # -------------------------
        # Save all steps as subplots
        # -------------------------
        images = [
            img_noprocessing,
            img_CLAHE,
            img_lowpass,
            img_highpass,
            img_truncated,
            img_renorm,
            img_gamma,
        ]

        titles = [
            "Normalized Input",
            f"CLAHE\nkernel={clahe_kernel}, clip={clahe_clip}",
            f"Low-pass\nsigma={self.low_pass_sigma}",
            f"High-pass\nsigma={self.high_pass_sigma}",
            f"Truncate\nstd scale={self.truncate_std_scale}",
            "Re-normalized",
            f"Gamma Corrected\ngamma={gamma_val}",
        ]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        for i, (image_step, title) in enumerate(zip(images, titles)):
            axes[i].imshow(image_step, cmap="gray")
            axes[i].set_title(title)
            axes[i].axis("off")

        # Turn off any unused subplot
        for j in range(len(images), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig("debug/pattern_processing_steps.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        return img

    def process_pattern_old(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        """Cleans patterns by equalizing the histogram and normalizing.
        Applies a bandpass filter to the patterns and truncates the extreme values.
        Images will be in the range [0, 1].

        Args:
            img (np.ndarray): The patterns to clean. (H, W)
            low_pass_sigma (float): The sigma for the low pass filter.
            high_pass_sigma (float): The sigma for the high pass filter.
            truncate_std_scale (float): The number of standard deviations to truncate.
        Returns:
            np.ndarray: The cleaned patterns. (N, H, W)"""

        # Correct dtype
        img = img.astype(np.float32)

        # Normalize
        img = (img - img.min()) / (img.max() - img.min())

        img_noprocessing = img.copy()
        img_noprocessing = np.around(255 * (img_noprocessing - img_noprocessing.min()) / (img_noprocessing.max() - img_noprocessing.min())).astype(np.uint8)
        io.imsave("debug/preprocessing_image.png", img_noprocessing)

        # ---- adaptive histogram equilization ----

        # Adaptive histogram equalization
        img = exposure.equalize_adapthist(
            img,
            kernel_size=(16, 16),
            clip_limit=0.01,
            nbins=256,
        ).astype(np.float32)
        img_CLAHE = img.copy()
        img_CLAHE = np.around(255 * (img_CLAHE - img_CLAHE.min()) / (img_CLAHE.max() - img_CLAHE.min())).astype(np.uint8)
        io.imsave("debug/CLAHE_filtered.png", img_CLAHE)

        print('the val of low pass', self.low_pass_sigma)
        # Low pass filter
        if self.low_pass_sigma > 0:
            img = ndimage.gaussian_filter(img, self.low_pass_sigma)
        #copy img and conver to a usable format to save
        img_lowpass = img.copy()
        img_lowpass = np.around(255 * (img_lowpass - img_lowpass.min()) / (img_lowpass.max() - img_lowpass.min())).astype(np.uint8)
        # save intermediate result 
        io.imsave("debug/low_pass_filtered.png", img_lowpass)
        # High pass filter
        if self.high_pass_sigma > 0:
            background = ndimage.gaussian_filter(img, self.high_pass_sigma)
            img = img - background
        img_highpass = img.copy()
        img_highpass = np.around(255 * (img_highpass - img_highpass.min()) / (img_highpass.max() - img_highpass.min())).astype(np.uint8)
        io.imsave("debug/high_pass_filtered.png", img_highpass)

        # Truncate step
        if self.truncate_std_scale > 0:
            mean, std = img.mean(), img.std()
            img = np.clip(
                img,
                mean - self.truncate_std_scale * std,
                mean + self.truncate_std_scale * std,
            )
        

        # Re normalize
        img = (img - img.min()) / (img.max() - img.min())

        img = img**1.1  # gamma correction
        img_gamma = img.copy()
        img_gamma = np.around(255 * (img_gamma - img_gamma.min()) / (img_gamma.max() - img_gamma.min())).astype(np.uint8)

        io.imsave("debug/gamma_filtered.png", img_gamma)
        #also save the final 
      
    
            # -------------------------
        # Save all steps as subplots
        # -------------------------
        images = [
            img_noprocessing,
            img_CLAHE,
            img_lowpass,
            img_highpass,
            img_gamma,
        ]

        titles = [
            "Normalized Input",
            "CLAHE",
            "Low-pass",
            "High-pass",
            "Gamma Corrected",
        ]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        for i, (image_step, title) in enumerate(zip(images, titles)):
            axes[i].imshow(image_step, cmap="gray")
            axes[i].set_title(title)
            axes[i].axis("off")

        # Turn off any unused subplot
        for j in range(len(images), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig("debug/pattern_processing_steps.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        return img

def process_pattern_no_class(
    
        img: np.ndarray,
    ) -> np.ndarray:
        """Cleans patterns by equalizing the histogram and normalizing.
        Applies a bandpass filter to the patterns and truncates the extreme values.
        Images will be in the range [0, 1].

        Args:
            img (np.ndarray): The patterns to clean. (H, W)
            low_pass_sigma (float): The sigma for the low pass filter.
            high_pass_sigma (float): The sigma for the high pass filter.
            truncate_std_scale (float): The number of standard deviations to truncate.
        Returns:
            np.ndarray: The cleaned patterns. (N, H, W)"""
        
        low_pass_sigma = 2.0
        high_pass_sigma = 101.0
        truncate_std_scale = 3.0

        # Correct dtype
        img = img.astype(np.float32)

        # Normalize
        img = (img - img.min()) / (img.max() - img.min())

        # # Low pass filter
        # if low_pass_sigma > 0:
        #     img = ndimage.gaussian_filter(img, low_pass_sigma)

        # # High pass filter
        # if high_pass_sigma > 0:
        #     background = ndimage.gaussian_filter(img, high_pass_sigma)
        #     img = img - background

        # # Truncate step
        # if truncate_std_scale > 0:
        #     mean, std = img.mean(), img.std()
        #     img = np.clip(
        #         img,
        #         mean - truncate_std_scale * std,
        #         mean + truncate_std_scale * std,
        #     )

        # # Re normalize
        # img = (img - img.min()) / (img.max() - img.min())

        return img

#------ New clasee to read a region of interest from UP2 files -------


if __name__ == "__main__":
    up2_path = "E:/SiGe/a-C03-scan/ScanA_1024x1024.up2"
    up2 = UP2(up2_path)
    pat = up2.read_pattern(0, process=True)
    pat = np.around(255 * (pat - pat.min()) / (pat.max() - pat.min())).astype(np.uint8)
    io.imsave("pattern.png", pat)
