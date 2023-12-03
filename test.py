import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_tts.model as module_model
from hw_tts.utils import ROOT_PATH
from hw_tts.utils.parse_config import ConfigParser
import numpy as np
import torchaudio
from hw_tts.preproc import MelSpectrogram

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"

    

def main(config, out_file):
    logger = config.get_logger("test")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    mel = MelSpectrogram().to(device)
    print(device)
    model = model.to(device)
    model.eval()


    data_path = os.listdir(config["data"]["test"])

    
    for i, audio_path in enumerate(data_path):
        audio, sr = torchaudio.load(config["data"]["test"] / audio_path)
        audio = audio[0:1, :]
        mel_ = mel(audio)
        generated = model(mel_.to(device))["pred_audio"].squeeze(0)
        torchaudio.save(f"{config['data']['test']}/test_{i}.wav", generated, 22050)

        
        

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )
    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": test_data_folder
        }


    main(config, args.output)