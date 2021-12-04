import pandas as pd

from src.4_Model import get_model
from src.2_Configuration import *
from test_dataset import ds_test
import torch


def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))


def remove_overlapping_pixels(mask, other_masks):
    for other_mask in other_masks:
        if np.sum(np.logical_and(mask, other_mask)) > 0:
            mask[np.logical_and(mask, other_mask)] = 0
    return mask


def main():
    model = get_model()
    model.load_state_dict(torch.load("pytorch_model-e8.bin"))
    model.eval()

    submission = []
    for sample in ds_test:
        img = sample['image']
        image_id = sample['image_id']
        with torch.no_grad():
            result = model([img.to(DEVICE)])[0]

        previous_masks = []
        for i, mask in enumerate(result["masks"]):

            # Filter-out low-scoring results. Not tried yet.
            score = result["scores"][i].cpu().item()
            if score < MIN_SCORE:
                continue

            mask = mask.cpu().numpy()
            # Keep only highly likely pixels
            binary_mask = mask > MASK_THRESHOLD
            binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
            previous_masks.append(binary_mask)
            rle = rle_encoding(binary_mask)
            submission.append((image_id, rle))

        # Add empty prediction if no RLE was generated for this image
        all_images_ids = [image_id for image_id, rle in submission]
        if image_id not in all_images_ids:
            submission.append((image_id, ""))

    df_sub = pd.DataFrame(submission, columns=['id', 'predicted'])
    df_sub.to_csv("submission.csv", index=False)
    print(df_sub.head())

if __name__ == '__main__':
    main()
