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
        low_pass_sigma: float = 1.0,
        high_pass_sigma: float = 10.0,
        high_pass_kernel_px: float = None,
        truncate_std_scale: float = 3.0,
        mask_type: str = "none",
        center_cross_half_width: int = 5,
        clahe_kernel: tuple = (5, 5),
        clahe_clip: float = 0.005,
        clahe_nbins: int = 256,
        flip_x: bool = False,
        use_clahe: bool = False,
        rescale_to_uint16: bool = False,
        unsharp_sigma: float = 0.0,
        unsharp_strength: float = 1.0,
        gamma: float = 0.33,
    ):
        """Set the parameters for processing the patterns.
        Values of 0.0 will skip the step.

        Args:
            low_pass_sigma (float): Sigma for a Gaussian noise-smoothing filter applied *after*
                background removal (Ernould et al. step 2). 1–2 pixels improves IC-GN convergence
                speed without loss of accuracy. 0.0 disables the step.
            high_pass_sigma (float): The sigma for the high pass filter. Roughly 20% of the image size works well.
            high_pass_kernel_px (float, optional): If given, overrides high_pass_sigma using
                the 3σ rule (sigma = kernel_px / 6).  Convenience for matching tools that
                specify a kernel size in pixels (CrossCourt, OpenXY, Wilkinson papers).
            truncate_std_scale (float): The number of standard deviations to truncate. 3.0 is a good value.
            mask_type (str): "circular", "center_cross", or None (no mask). Controls which mask is applied in process_pattern.
            center_cross_half_width (int): Half-width of the cross arms in pixels when mask_type="center_cross" (default 5 → 10 px total).
            clahe_kernel (tuple): Kernel size for CLAHE. Smaller = more local contrast. Default (5, 5).
            clahe_clip (float): Clip limit for CLAHE. Higher = more contrast. Default 0.005.
            clahe_nbins (int): Number of histogram bins for CLAHE. Default 256.
            use_clahe (bool): Set False to skip CLAHE entirely. Default True.
            rescale_to_uint16 (bool): If True, linearly rescale each raw pattern so
                its minimum maps to 0 and its maximum maps to 65535 before any other
                processing. Useful when the detector does not fill the full uint16
                dynamic range. Default False.
            unsharp_sigma (float): Sigma for unsharp masking applied after CLAHE.
                Sharpens band edges by subtracting a blurred copy.
                0.0 disables the step. Good starting range: 1.0–3.0 pixels. Default 0.0.
            unsharp_strength (float): Weight of the sharpening term. Higher = sharper
                but more noise amplification. Typical range: 0.5–3.0. Default 1.0.
        """
        self.low_pass_sigma = low_pass_sigma
        if high_pass_kernel_px is not None and high_pass_kernel_px > 0:
            high_pass_sigma = float(high_pass_kernel_px) / 6.0
            print(f"  [high-pass] kernel_px={high_pass_kernel_px} → sigma={high_pass_sigma:.3f} (3σ rule)")
        self.high_pass_sigma = high_pass_sigma
        self.truncate_std_scale = truncate_std_scale
        self.mask_type = mask_type
        self.center_cross_half_width = center_cross_half_width
        self.clahe_kernel = clahe_kernel
        self.clahe_clip = clahe_clip
        self.clahe_nbins = clahe_nbins
        self.flip_x = flip_x
        self.use_clahe = use_clahe
        self.rescale_to_uint16 = rescale_to_uint16
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_strength = unsharp_strength
        self.gamma = gamma
        print(f"Set UP2 pattern processing: low_pass_sigma={low_pass_sigma}, high_pass_sigma={high_pass_sigma}, truncate_std_scale={truncate_std_scale}, mask_type={mask_type}, clahe_kernel={clahe_kernel}, clahe_clip={clahe_clip}, use_clahe={use_clahe}, flip_x={flip_x}, rescale_to_uint16={rescale_to_uint16}, unsharp_sigma={unsharp_sigma}, unsharp_strength={unsharp_strength}, gamma={gamma}")

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
        if self.rescale_to_uint16:
            pmin, pmax = pat.min(), pat.max()
            if pmax > pmin:
                pat = np.round((pat.astype(np.float32) - pmin) / (pmax - pmin) * 65535).astype(np.uint16)
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
            # mode='constant', cval=0 ensures out-of-bounds pixels contribute
            # zero weight; 'reflect' (the default) mirrors real values at the
            # image edge, inflating weights near corners and washing out contrast.
            weighted = ndimage.gaussian_filter(image * float_mask, sigma, mode='constant', cval=0.0)
            weights  = ndimage.gaussian_filter(float_mask, sigma, mode='constant', cval=0.0)
            # Use np.divide with `where=` so the division ONLY runs where weights
            # are positive; everywhere else the output keeps its initial 0.0.
            # This eliminates the spurious "invalid value encountered in divide"
            # RuntimeWarning that `np.where` would trigger (np.where evaluates
            # both branches in full, so the bad division still happens).
            out = np.zeros_like(weighted)
            np.divide(weighted, weights, out=out, where=(weights > 0))
            return out

        # High-pass filter in log domain (Ernould et al.)
        # EBSD background is multiplicative, so subtract in log space:
        #   log(I) - gaussian(log(I))  ≡  log(I / background)
        # This correctly removes the smooth inelastic-scattering envelope without
        # biasing the result the way linear subtraction does.
        # A percentile floor (1st percentile of valid pixels) replaces the naive eps
        # to prevent log(~0) in dark corners from biasing the background estimate and
        # creating a bright ring / ramp artifact that would dominate the gradients.
        if self.high_pass_sigma > 0:
            img_floor = float(np.percentile(img[mask], 1))
            img_floor = max(img_floor, 1e-4)          # absolute safety minimum
            log_img = np.log(np.maximum(img, img_floor))
            log_img[~mask] = 0.0
            log_background = masked_gaussian(log_img, mask, self.high_pass_sigma)
            img = np.exp(log_img - log_background)
            img[~mask] = 0.0
            img = masked_normalize(img, mask)

        img_highpass = to_uint8(img, mask)

        # ---- adaptive histogram equalization ----
        # Applied immediately after background removal so CLAHE sees the sharpest
        # possible image. Running it after the low-pass would mean enhancing already-
        # blurred data, which wastes the contrast gain on smoothed band edges.
        # CLAHE has no native mask support, so inpaint the masked region with
        # Gaussian-interpolated values from valid neighbours before running it,
        # then zero it out again afterwards.
        if self.use_clahe:
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

        # Low-pass filter (noise smoothing) — applied *after* background removal and
        # CLAHE, per Ernould et al. A small radius (1–2 px) reduces high-frequency
        # noise and improves IC-GN convergence speed without loss of accuracy.
        if self.low_pass_sigma > 0:
            img = masked_gaussian(img, mask, self.low_pass_sigma)
            img[~mask] = 0.0

        img_lowpass = to_uint8(img, mask)

        # Unsharp masking — enhances band edges by amplifying fine detail
        # sharpened = img + strength * (img - gaussian(img, sigma))
        # Applied after CLAHE so contrast is already equalised before sharpening.
        if self.unsharp_sigma > 0:
            blurred = masked_gaussian(img, mask, self.unsharp_sigma)
            img[mask] = img[mask] + self.unsharp_strength * (img[mask] - blurred[mask])
            img[~mask] = 0.0
            img = masked_normalize(img, mask)   # re-clip to [0, 1] after boosting

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
        if self.gamma != 1.0:
            img[mask] = img[mask] ** self.gamma
        img[~mask] = 0.0

        img_gamma = to_uint8(img, mask)

        # Final zero-mean, unit-std normalisation over the valid region
        vals = img[mask]
        mean_val = vals.mean()
        std_val  = vals.std()
        if std_val > 0:
            img[mask] = (img[mask] - mean_val) / std_val
        else:
            img[mask] = 0.0
        img[~mask] = 0.0

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
                    axes[r, c].set_title(f"hp_sigma={sigma}", fontsize=20, fontweight="bold")
            axes[r, 0].text(
                -0.05, 0.5, f"clahe={kernel}",
                fontsize=20, fontweight="bold", transform=axes[r, 0].transAxes,
                ha="right", va="center", rotation=90,
            )

        fig.suptitle(f"Parameter sweep  |  pattern {pattern_idx}", fontsize=20, fontweight="bold")
        plt.tight_layout()
        out_path = f"{save_dir}/sweep_combined.jpg"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

#------ New clasee to read a region of interest from UP2 files -------


if __name__ == "__main__":
    up2_path = "E:/SiGe/a-C03-scan/ScanA_1024x1024.up2"
    up2 = UP2(up2_path)
    pat = up2.read_pattern(0, process=True)
    pat = np.around(255 * (pat - pat.min()) / (pat.max() - pat.min())).astype(np.uint8)
    io.imsave("pattern.png", pat)
