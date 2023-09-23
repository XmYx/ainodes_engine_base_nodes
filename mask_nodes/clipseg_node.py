
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np

from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import cv2

from scipy.ndimage import gaussian_filter

from typing import Optional, Tuple

import warnings


from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.settings import handle_ainodes_exception
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs



"""Helper methods for CLIPSeg nodes"""

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy array and scale its values to 0-255."""
    array = tensor.numpy().squeeze()
    return (array * 255).astype(np.uint8)

def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a tensor and scale its values from 0-255 to 0-1."""
    array = array.astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]

def apply_colormap(mask: torch.Tensor, colormap) -> np.ndarray:
    """Apply a colormap to a tensor and convert it to a numpy array."""
    colored_mask = colormap(mask.numpy())[:, :, :3]
    return (colored_mask * 255).astype(np.uint8)

def resize_image(image: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    """Resize an image to the given dimensions using linear interpolation."""
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)

def overlay_image(background: np.ndarray, foreground: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay the foreground image onto the background with a given opacity (alpha)."""
    return cv2.addWeighted(background, 1 - alpha, foreground, alpha, 0)

def dilate_mask(mask: torch.Tensor, dilation_factor: float) -> torch.Tensor:
    """Dilate a mask using a square kernel with a given dilation factor."""
    kernel_size = int(dilation_factor * 2) + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask.numpy(), kernel, iterations=1)
    return torch.from_numpy(mask_dilated)

OP_NODE_CLIPSEG = get_next_opcode()
class ClipSegWidget(QDMNodeContentWidget):
    def initUI(self):
        self.prompt = self.create_text_edit("Prompt")

        self.blur = self.create_double_spin_box("Blur", default_val=7.0, min_val=0.0, max_val=100.0)
        self.threshold = self.create_double_spin_box("Threshold", default_val=0.4, min_val=0.0)
        self.dilation = self.create_spin_box("Dilation", min_val=0, max_val=100, default_val=4)


        self.create_main_layout(grid=1)

@register_node(OP_NODE_CLIPSEG)
class DataMergeNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Data objects in aiNodes are simple dictionaries,\n" \
                "that can hold any values under any name.\n" \
                "In most cases, you'll find them drive parameters,\n" \
                "or hold sequences of images. For an example, the\n" \
                "OpenAI node emits it's prompt in a data line,\n" \
                "but you'll find this info in all relevant places."
    op_code = OP_NODE_CLIPSEG
    op_title = "ClipSeg"
    content_label_objname = "clipseg_node"
    category = "aiNodes Base/WIP Experimental"
    NodeContent_class = ClipSegWidget
    dim = (340, 180)
    output_data_ports = [0,1,2]
    exec_port = 3

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,1], outputs=[5,5,5,1])

    def evalImplementation_thread(self, index=0):
        images = self.getInputData(0)

        image = images[0]
        prompt = self.content.prompt.toPlainText()
        blur = self.content.blur.value()
        threshold = self.content.threshold.value()
        dilation = self.content.dilation.value()

        tensor_bw, mask_norm_image, image_out_binary = self.segment_image(image, prompt, blur, threshold, dilation)


        out_img = image - image_out_binary

        return [[mask_norm_image], [out_img], [image_out_binary]]



    def segment_image(self, image: torch.Tensor, text: str, blur: float, threshold: float, dilation_factor: int) -> \
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a segmentation mask from an image and a text prompt using CLIPSeg.

        Args:
            image (torch.Tensor): The image to segment.
            text (str): The text prompt to use for segmentation.
            blur (float): How much to blur the segmentation mask.
            threshold (float): The threshold to use for binarizing the segmentation mask.
            dilation_factor (int): How much to dilate the segmentation mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The segmentation mask, the heatmap mask, and the binarized mask.
        """

        # Convert the Tensor to a PIL image
        image_np = image.numpy().squeeze()  # Remove the first dimension (batch size of 1)
        # Convert the numpy array back to the original range (0-255) and data type (uint8)
        image_np = (image_np * 255).astype(np.uint8)
        # Create a PIL image from the numpy array
        i = Image.fromarray(image_np, mode="RGB")

        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        prompt = text

        input_prc = processor(text=prompt, images=i, padding="max_length", return_tensors="pt")

        # Predict the segemntation mask
        with torch.no_grad():
            outputs = model(**input_prc)

        tensor = torch.sigmoid(outputs[0])  # get the mask

        # Apply a threshold to the original tensor to cut off low values
        thresh = threshold
        tensor_thresholded = torch.where(tensor > thresh, tensor, torch.tensor(0, dtype=torch.float))

        # Apply Gaussian blur to the thresholded tensor
        sigma = blur
        tensor_smoothed = gaussian_filter(tensor_thresholded.numpy(), sigma=sigma)
        tensor_smoothed = torch.from_numpy(tensor_smoothed)

        # Normalize the smoothed tensor to [0, 1]
        mask_normalized = (tensor_smoothed - tensor_smoothed.min()) / (tensor_smoothed.max() - tensor_smoothed.min())

        # Dilate the normalized mask
        mask_dilated = dilate_mask(mask_normalized, dilation_factor)

        # Convert the mask to a heatmap and a binary mask
        heatmap = apply_colormap(mask_dilated, cm.viridis)
        binary_mask = apply_colormap(mask_dilated, cm.Greys_r)

        # Overlay the heatmap and binary mask on the original image
        dimensions = (image_np.shape[1], image_np.shape[0])
        heatmap_resized = resize_image(heatmap, dimensions)
        binary_mask_resized = resize_image(binary_mask, dimensions)

        alpha_heatmap, alpha_binary = 0.5, 1
        overlay_heatmap = overlay_image(image_np, heatmap_resized, alpha_heatmap)
        overlay_binary = overlay_image(image_np, binary_mask_resized, alpha_binary)

        # Convert the numpy arrays to tensors
        image_out_heatmap = numpy_to_tensor(overlay_heatmap)
        image_out_binary = numpy_to_tensor(overlay_binary)

        # Save or display the resulting binary mask
        binary_mask_image = Image.fromarray(binary_mask_resized[..., 0])

        # convert PIL image to numpy array
        tensor_bw = binary_mask_image.convert("RGB")
        tensor_bw = np.array(tensor_bw).astype(np.float32) / 255.0
        tensor_bw = torch.from_numpy(tensor_bw)[None,]
        tensor_bw = tensor_bw.squeeze(0)[..., 0]

        mask_norm_image = mask_normalized

        return tensor_bw, mask_norm_image, image_out_binary
