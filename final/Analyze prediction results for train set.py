# Plots: the image, The image + the ground truth mask, The image + the predicted mask
import matplotlib.pyplot as plt
from comfig import *


def analyze_train_sample(model, ds_train, sample_index):
    img, targets = ds_train[sample_index]
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.title("Image")
    plt.show()

    masks = np.zeros((HEIGHT, WIDTH))
    for mask in targets['masks']:
        masks = np.logical_or(masks, mask)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.imshow(masks, alpha=0.3)
    plt.title("Ground truth")
    plt.show()

    model.eval()
    with torch.no_grad():
        preds = model([img.to(DEVICE)])[0]

    plt.imshow(img.cpu().numpy().transpose((1, 2, 0)))
    all_preds_masks = np.zeros((HEIGHT, WIDTH))
    for mask in preds['masks'].cpu().detach().numpy():
        all_preds_masks = np.logical_or(all_preds_masks, mask[0] > MASK_THRESHOLD)
    plt.imshow(all_preds_masks, alpha=0.4)
    plt.title("Predictions")
    plt.show()

    # NOTE: It puts the model in eval mode!! Revert for re-training
    analyze_train_sample(model, ds_train, 20)
