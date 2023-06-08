import numpy as np
import torch
from PIL import Image
from qtpy import QtWidgets, QtCore, QtGui

from ..ainodes_backend.resizeRight import resizeright, interp_methods
from ..ainodes_backend import pixmap_to_pil_image, torch_gc

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.node_engine.utils import dumpException

OP_NODE_LATENT = get_next_opcode()
OP_NODE_LATENT_COMPOSITE = get_next_opcode()
class LatentWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.create_widgets()
        self.create_main_layout(grid=1)

    def create_widgets(self):
        self.width = self.create_spin_box("Width", 64, 4096, 512, 64)
        self.height = self.create_spin_box("Height", 64, 4096, 512, 64)
        self.rescale_latent = self.create_check_box("Latent Rescale")
@register_node(OP_NODE_LATENT)
class LatentNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/empty_latent.png"
    op_code = OP_NODE_LATENT
    op_title = "Empty Latent Image"
    content_label_objname = "empty_latent_node"
    category = "Latent"

    def __init__(self, scene):

        super().__init__(scene, inputs=[2,5,1], outputs=[2,1])
        #self.eval()

    def initInnerClasses(self):
        self.content = LatentWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)

        self.input_socket_name = ["EXEC", "IMAGE", "LATENT"]
        self.output_socket_name = ["EXEC", "LATENT"]
        self.grNode.height = 210
        self.grNode.width = 200
        self.content.eval_signal.connect(self.evalImplementation)

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0):
        samples = []
        if self.getInput(0) != None:
            try:
                latent_node, index = self.getInput(0)
                samples = latent_node.getOutput(index)
                print(f"EMPTY LATENT NODE: Using Latent input with parameters: {samples}")
            except:
                print(f"EMPTY LATENT NODE: Tried using Latent input, but found an invalid value, generating latent with parameters: {self.content.width.value(), self.content.height.value()}")
                samples = [self.generate_latent()]
            self.markDirty(False)
            self.markInvalid(False)
        elif self.getInput(1) != None:
            try:
                node, index = self.getInput(1)
                pixmap_list = node.getOutput(index)
                samples = []
                gs.models["vae"].first_stage_model.cuda()
                for pixmap in pixmap_list:
                    image = pixmap_to_pil_image(pixmap)

                    #print("image", image)

                    """image, mask_image = load_img(image,
                                                 shape=(image.size[0], image.size[1]),
                                                 use_alpha_as_mask=True)
                    image = image.to("cuda")
                    image = repeat(image, '1 ... -> b ...', b=1)"""


                    image = image.convert("RGB")
                    image = np.array(image).astype(np.float32) / 255.0
                    image = image[None] #.transpose(0, 3, 1, 2)
                    image = torch.from_numpy(image)
                    image = image.detach().cpu()
                    torch_gc()

                    latent = gs.models["vae"].encode(image)
                    latent = latent.to("cpu")
                    image = image.detach().to("cpu")
                    del image
                    samples.append(latent)
                    shape = latent.shape
                    del latent
                    torch_gc()
                gs.models["vae"].first_stage_model.cpu()
                torch_gc()
                if gs.logging:
                    print(f"EMPTY LATENT NODE: Using Image input, encoding to Latent with parameters: {shape}")
            except Exception as e:
                print(e)
        else:
            samples = [self.generate_latent()]
        if self.content.rescale_latent.isChecked() == True:
            rescaled_samples = []
            for sample in samples:
                sample = sample.float()
                return_sample = resizeright.resize(sample, scale_factors=None,
                                                out_shape=[sample.shape[0], sample.shape[1], int(self.content.height.value() // 8),
                                                        int(self.content.width.value() // 8)],
                                                interp_method=interp_methods.lanczos3, support_sz=None,
                                                antialiasing=True, by_convs=True, scale_tolerance=None,
                                                max_numerator=10, pad_mode='reflect').half()

                rescaled_samples.append(return_sample)
            samples = rescaled_samples
            if gs.logging:
                print(f"{len(samples)}x Latents rescaled to: {samples[0].shape}")
        #print(samples[0].shape)

        return samples
            #return self.value

    ##@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        self.markDirty(False)
        self.markInvalid(False)
        self.setOutput(0, result)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
    def onMarkedDirty(self):
        self.value = None
    def encode_image(self, init_image=None):
        init_latent = gs.models["vae"].encode(init_image)
        init_latent.to("cpu")# move to latent space
        torch_gc()
        return init_latent

    def generate_latent(self):
        width = self.content.width.value()
        height = self.content.height.value()
        batch_size = 1
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return latent

class LatentCompositeWidget(QDMNodeContentWidget):
    def initUI(self):
        # Create a label to display the image
        self.width = QtWidgets.QSpinBox()
        self.width.setMinimum(64)
        self.width.setMaximum(4096)
        self.width.setValue(64)
        self.width.setSingleStep(64)

        self.height = QtWidgets.QSpinBox()
        self.height.setMinimum(64)
        self.height.setMaximum(4096)
        self.height.setValue(64)
        self.height.setSingleStep(64)

        self.feather = QtWidgets.QSpinBox()
        self.feather.setMinimum(0)
        self.feather.setMaximum(200)
        self.feather.setValue(10)
        self.feather.setSingleStep(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(15,15,15,20)
        layout.setSpacing(10)
        layout.addWidget(self.width)
        layout.addWidget(self.height)
        layout.addWidget(self.feather)
        self.setLayout(layout)

    def serialize(self):
        res = super().serialize()
        #res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            value = data['value']
            #self.image.setPixmap(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_LATENT_COMPOSITE)
class LatentCompositeNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/comp_latent.png"
    op_code = OP_NODE_LATENT_COMPOSITE
    op_title = "Composite Latent Images"
    content_label_objname = "latent_comp_node"
    category = "Latent"

    def __init__(self, scene):
        super().__init__(scene, inputs=[2,2,3], outputs=[2,3])
    def initInnerClasses(self):
        self.content = LatentCompositeWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.input_socket_name = ["EXEC", "LATENT1", "LATENT2"]
        self.output_socket_name = ["EXEC", "LATENT"]
        self.grNode.height = 220
        self.grNode.width = 240
        self.grNode.icon = self.icon
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):

        if self.isDirty() == True:
            if self.getInput(index) != None:

                self.value = self.composite()
        else:
            return self.value

    def onWorkerFinished(self, result):
        self.setOutput(0, result)
        self.markDirty(False)
        self.markInvalid(False)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)
        return self.value

    def onMarkedDirty(self):
        self.value = None

    def composite(self):
        width = self.content.width.value()
        height = self.content.height.value()
        feather = self.content.feather.value()
        x =  width // 8
        y = height // 8
        feather = feather // 8
        samples_out = self.getInput(0)
        s = self.getInput(0)
        samples_to = self.getInput(0)
        samples_from = self.getInput(1)
        if feather == 0:
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
        else:
            samples_from = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
            mask = torch.ones_like(samples_from)
            for t in range(feather):
                if y != 0:
                    mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))

                if y + samples_from.shape[2] < samples_to.shape[2]:
                    mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                if x != 0:
                    mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                if x + samples_from.shape[3] < samples_to.shape[3]:
                    mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
            rev_mask = torch.ones_like(mask) - mask
            s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x] * mask + s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] * rev_mask

        self.setOutput(0, s)
        return s
def load_img(image, shape=None, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    #if path.startswith('http://') or path.startswith('https://'):
    #    image = Image.open(requests.get(path, stream=True).raw)
    #else:
    #    image = Image.open(path)

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    if shape is not None:
        image = image.resize(shape, resample=Image.Resampling.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.

    return image, mask_image