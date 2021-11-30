from dataset import data_utils


def prerocess_ct_2d_image(image, label, config):
    # Modift HU value
    image = data_utils.process_hu_value(image, low_cut_off=config.preprocess_config.hu_low, high_cut_off=config.preprocess_config.hu_high)

    return image, label
