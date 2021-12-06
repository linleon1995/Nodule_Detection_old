import sys
import os
import torch
import numpy as np
import tensorboardX
import random
import argparse
from pprint import pprint

import site_path
from dataset import build
from config import get_config
CONFIG_PATH = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\configs\_cnn_train_config.yml'
# sys.path.append("...")

from modules.model.image_calssification import img_classifier
from modules.model.unet import unet_2d
from modules.model import layers
from modules.train import trainer
from modules.utils import train_utils
from modules.utils import configuration

logger = train_utils.get_logger('train')


def parse_option():
    parser = argparse.ArgumentParser('LUNA16 training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', default=CONFIG_PATH)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel', default=0)

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    # TODO: combine to config.py
    # Configuration
    all_checkpoint_path = os.path.join(config.OUTPUT, 'checkpoints')
    checkpoint_path = train_utils.create_training_path(all_checkpoint_path)
    config.defrost()
    config.CHECKPOINT_PATH = checkpoint_path
    config.freeze()
    pprint(config)
    
    # Set deterministic
    manual_seed = config.get('SEED', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        train_utils.set_deterministic(manual_seed, random, np, torch)
            
    # Dataloader
    train_dataset, valid_dataset, train_dataloader, valid_dataloader, _ = build.build_loader(config)
    # train_dataset = AudioDataset(config, mode='train')
    # valid_dataset = AudioDataset(config, mode='valid')
    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=config.dataset.batch_size, shuffle=config.dataset.shuffle, pin_memory=config.train.pin_memory, num_workers=config.train.num_workers)
    # valid_dataloader = DataLoader(
    #     valid_dataset, batch_size=1, shuffle=False, pin_memory=config.train.pin_memory, num_workers=config.train.num_workers)

    # Logger
    logger.info("Start Training!!")
    logger.info("Training epoch: {} Batch size: {} Training Samples: {}".
            format(config.TRAIN.EPOCH, config.DATA.BATCH_SIZE, len(train_dataloader.dataset)))
    train_utils.config_logging(os.path.join(checkpoint_path, 'logging.txt'), config, access_mode='w+')

    # Model
    # model = img_classifier.ImageClassifier(
    #     backbone=config.MODEL.NAME, in_channels=config.MODEL.IN_CHANNELS, activation=config.MODEL.ACTIVATION,
    #     out_channels=config.MODEL.NUM_CLASSES, pretrained=config.TRAIN.USE_CHECKPOINT, dim=1, output_structure=None)
    f_maps = [64, 256, 512, 1024, 2048]
    f_maps = [64, 256, 512]
    model = unet_2d.UNet_2d_backbone(
        in_channels=config.MODEL.IN_CHANNELS, out_channels=config.MODEL.NUM_CLASSES, f_maps=f_maps, basic_module=layers.DoubleConv, pretrained=config.TRAIN.USE_CHECKPOINT)

    # Optimizer
    optimizer = train_utils.create_optimizer(config, model)

    # Criterion (Loss function)
    def criterion_wrap(outputs, labels):
        criterion = train_utils.create_criterion(config.TRAIN.LOSS)
        # TODO: loss = criterion(outputs, torch.argmax(labels.long(), axis=1))
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(outputs, torch.argmax(labels.long(), axis=1))
        else:
            loss = criterion(outputs, labels)
        return loss
    # criterion = train_utils.create_criterion(config.train.loss)

    # Final activation
    if config.MODEL.ACTIVATION:
        activation_func = train_utils.create_activation(config.MODEL.ACTIVATION)
    else:
        activation_func = None

    # Training
    trainer_instance = trainer.Trainer(config,
                                       model, 
                                       criterion_wrap, 
                                       optimizer, 
                                       train_dataloader, 
                                       valid_dataloader,
                                       logger,
                                       device=configuration.get_device({})['device'],
                                       activation_func=activation_func,
                                       )

    trainer_instance.fit()


if __name__ == '__main__':
    _, config = parse_option()
    print(config)
    main(config)