import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.utils.data import Dataset, DataLoader, distributed
from transformers import SegformerForSemanticSegmentation
import rasterio
from tqdm import tqdm

# -------------------------
# Configurations
# -------------------------
BATCH_SIZE = 4
NUM_WORKERS = 4
CHIP_SIZE = 1024
OVERLAP = 128
THRESHOLD = 0.5
#MODEL_PATH = "/scratch1/09714/mpimenta/segformer_chpt1/results/segformer_mit-b3_1024_20250123_201329/model_epoch_13.pth"
MODEL_PATH = "/scratch1/projects/PDG_shared/Troughs_Segformer/model_epoch_13.pth"

INPUT_FOLDER = "/scratch1/projects/PDG_shared/Troughs_Segformer/AL_sample/"
OUTPUT_FOLDER = "/scratch1/projects/PDG_shared/Troughs_Segformer/AL_sample_out/"

# -------------------------
# Helper Functions
# -------------------------

def reflect_pad(image, pad_size):
    """Apply reflection padding to an image."""
    return np.pad(image, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), mode='reflect')

def generate_chips(w, h, chip_size, overlap):
    """Generate chip coordinates over the original image.
       Coordinates are based on the original image dimensions.
    """
    step = chip_size - overlap
    chips = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            chips.append((x, y))
    return chips

class TifDataset(Dataset):
    def __init__(self, image, chips, chip_size):
        """
        image: The padded image (numpy array of shape [channels, height + 2*OVERLAP, width + 2*OVERLAP])
        chips: List of (x, y) coordinates for the original image (top-left)
        chip_size: Desired chip size (square)
        """
        self.image = image
        self.chips = chips
        self.chip_size = chip_size

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        x, y = self.chips[idx]
        # Add OVERLAP offset because the image is padded on all sides
        chip = self.image[:, y + OVERLAP : y + OVERLAP + self.chip_size, x + OVERLAP : x + OVERLAP + self.chip_size]
        c, h, w = chip.shape
        pad_h = self.chip_size - h
        pad_w = self.chip_size - w
        if pad_h > 0 or pad_w > 0:
            chip = np.pad(chip, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        return torch.from_numpy(chip).float(), (x, y)

def my_collate(batch):
    """Custom collate function that stacks images and returns positions as a list."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    images, positions = zip(*batch)
    images = torch.stack(images, 0)
    return images, positions
def save_final_mask(input_raster_path, output_mask_path, sum_array, count_array):
    """Calculate the final mask and write to disk (only on rank 0).
       Also removes any nodata value from the profile.
    """
    if dist.get_rank() != 0:
        return

    valid_mask = count_array > 0
    avg_pred = np.zeros_like(sum_array, dtype=np.float32)
    avg_pred[valid_mask] = sum_array[valid_mask] / count_array[valid_mask]
    final_mask = (avg_pred > THRESHOLD).astype(np.uint8)

    with rasterio.open(input_raster_path) as src:
        profile = src.profile.copy()
    # Remove nodata from profile since our output is uint8
    profile.pop('nodata', None)
    profile.update({
        'count': 1,
        'dtype': 'uint8',
        'compress': 'lzw',
        'BIGTIFF': 'YES'
    })

    with rasterio.open(output_mask_path, 'w', **profile) as dst:
        dst.write(final_mask, 1)

# -------------------------
# Inference Function
# -------------------------
def process_tif_file(input_raster_path, output_folder, model, device):
    # Open the input raster and read RGB bands.
    with rasterio.open(input_raster_path) as src:
        full_image = src.read([1, 2, 3]).astype(np.float32)
        profile = src.profile

    orig_h, orig_w = full_image.shape[1:]
    # Pad the full image on all sides by OVERLAP.
    padded_image = reflect_pad(full_image, OVERLAP)
    # Generate chip coordinates over the original image.
    chips = generate_chips(orig_w, orig_h, CHIP_SIZE, OVERLAP)

    dataset = TifDataset(padded_image, chips, CHIP_SIZE)
    sampler = distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, collate_fn=my_collate)

    sum_array = np.zeros((orig_h, orig_w), dtype=np.float32)
    count_array = np.zeros((orig_h, orig_w), dtype=np.uint16)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference GPU {dist.get_rank()}"):
            if batch is None:
                continue
            images, positions = batch
            images = images.to(device)
            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=(CHIP_SIZE, CHIP_SIZE), mode='bilinear', align_corners=False)
            preds = torch.sigmoid(outputs).cpu().numpy().astype(np.float32)
            for pred, pos in zip(preds, positions):
                merge_pred_into_arrays(pred.squeeze(), pos, sum_array, count_array, orig_w, orig_h)
   dist.barrier()
    if dist.get_rank() == 0:
        output_filename = os.path.splitext(os.path.basename(input_raster_path))[0] + "_mask.tif"
        output_path = os.path.join(output_folder, output_filename)
        save_final_mask(input_raster_path, output_path, sum_array, count_array)
    dist.barrier()
# -------------------------
# Main Execution
# -------------------------
def main():

    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{dist.get_rank()}")
    print("loading model .................")
    # Load model and remove 'module.' prefix if necessary.
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3", num_labels=1)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
    print("..............loading model done")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    tif_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".tif")]

    for tif_file in tif_files:
        input_path = os.path.join(INPUT_FOLDER, tif_file)
        process_tif_file(input_path, OUTPUT_FOLDER, model, device)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

#conda activate /scratch1/projects/PDG_shared/Troughs_Segformer/segformer-env
#python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="129.114.44.118" --master_port=12355 inference.py
