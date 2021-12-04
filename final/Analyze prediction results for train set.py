# Plots: the image, The image + the ground truth mask, The image + the predicted mask
import matplotlib.pyplot as plt
from Model import get_model
from comfig import *
from training import ds_train


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


def main():
    # NOTE: It puts the model in eval mode!! Revert for re-training
    model = get_model()
    model.load_state_dict(torch.load("pytorch_model-e8.bin"))
    analyze_train_sample(model, ds_train, 20)
    analyze_train_sample(model, ds_train, 100)
    analyze_train_sample(model, ds_train, 2)


if __name__ == '__main__':
    main()
