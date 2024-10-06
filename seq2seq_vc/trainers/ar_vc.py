#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
import soundfile as sf
import time
import torch

from seq2seq_vc.trainers.base import Trainer
from seq2seq_vc.utils.model_io import (
    filter_modules,
    get_partial_state_dict,
    transfer_verification,
    print_new_keys,
)

# set to avoid matplotlib error in CLI environment
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ARVCTrainer(Trainer):
    """Customized trainer module for autoregressive VC training."""

    def load_trained_modules(self, checkpoint_path, init_mods):
        if self.config["distributed"]:
            main_state_dict = self.model.module.state_dict()
        else:
            main_state_dict = self.model.state_dict()

        if os.path.isfile(checkpoint_path):
            model_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

            # first make sure that all modules in `init_mods` are in `checkpoint_path`
            modules = filter_modules(model_state_dict, init_mods)
            # then, actually get the partial state_dict
            partial_state_dict = get_partial_state_dict(model_state_dict, modules)

            if partial_state_dict:
                if transfer_verification(main_state_dict, partial_state_dict, modules):
                    print_new_keys(partial_state_dict, modules, checkpoint_path)
                    main_state_dict.update(partial_state_dict)
        else:
            logging.error(f"Specified model was not found: {checkpoint_path}")
            exit(1)

        if self.config["distributed"]:
            self.model.module.load_state_dict(main_state_dict)
        else:
            self.model.load_state_dict(main_state_dict)

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        # xs, ilens, ys, labels, olens, spembs = tuple(
        # [_.to(self.device) if _ is not None else _ for _ in batch]
        # )
        xs = batch["xs"].to(self.device)
        ys = batch["ys"].to(self.device)
        ilens = batch["ilens"].to(self.device)
        olens = batch["olens"].to(self.device)
        labels = batch["labels"].to(self.device)

        # model forward
        (
            after_outs,
            before_outs,
            logits,
            ys_,
            labels_,
            olens_,
            (att_ws, ilens_ds_st, olens_in),
        ) = self.model(xs, ilens, ys, labels, olens)

        # seq2seq loss
        l1_loss, bce_loss = self.criterion["Seq2SeqLoss"](
            after_outs, before_outs, logits, ys_, labels_, olens_
        )
        gen_loss = l1_loss + bce_loss
        self.total_train_loss["train/l1_loss"] += l1_loss.item()
        self.total_train_loss["train/bce_loss"] += bce_loss.item()

        # guided attention loss
        if self.config.get("use_guided_attn_loss", False):
            ga_loss = self.criterion["guided_attn"](att_ws, ilens_ds_st, olens_in)
            gen_loss += ga_loss
            self.total_train_loss["train/guided_attn_loss"] += ga_loss.item()

        self.total_train_loss["train/loss"] += gen_loss.item()

        # update model
        self.optimizer.zero_grad()
        gen_loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["grad_norm"],
            )
        self.optimizer.step()
        self.scheduler.step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # define function for plot prob and att_ws
        def _plot_and_save(
            array, figname, figsize=(6, 4), dpi=150, ref=None, origin="upper"
        ):
            shape = array.shape
            if len(shape) == 1:
                # for eos probability
                plt.figure(figsize=figsize, dpi=dpi)
                plt.plot(array)
                plt.xlabel("Frame")
                plt.ylabel("Probability")
                plt.ylim([0, 1])
            elif len(shape) == 2:
                # for tacotron 2 attention weights, whose shape is (out_length, in_length)
                if ref is None:
                    plt.figure(figsize=figsize, dpi=dpi)
                    plt.imshow(array.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
                else:
                    plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
                    plt.subplot(1, 2, 1)
                    plt.imshow(array.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
                    plt.subplot(1, 2, 2)
                    plt.imshow(ref.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
            elif len(shape) == 4:
                # for transformer attention weights,
                # whose shape is (#leyers, #heads, out_length, in_length)
                plt.figure(
                    figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi
                )
                for idx1, xs in enumerate(array):
                    for idx2, x in enumerate(xs, 1):
                        plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                        plt.imshow(x, aspect="auto")
                        plt.xlabel("Input")
                        plt.ylabel("Output")
            else:
                raise NotImplementedError("Support only from 1D to 4D array.")
            plt.tight_layout()
            if not os.path.exists(os.path.dirname(figname)):
                # NOTE: exist_ok = True is needed for parallel process decoding
                os.makedirs(os.path.dirname(figname), exist_ok=True)
            plt.savefig(figname)
            plt.close()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # generate
        # xs, _, ys, _, olens, spembs = tuple(
        #     [_.to(self.device) if _ is not None else _ for _ in batch]
        # )
        xs = batch["xs"].to(self.device)
        ys = batch["ys"].to(self.device)
        ilens = batch["ilens"].to(self.device)
        olens = batch["olens"].to(self.device)
        labels = batch["labels"].to(self.device)
        spembs = [None] * len(xs)

        for idx, (x, y, olen, spemb) in enumerate(zip(xs, ys, olens, spembs)):
            start_time = time.time()
            outs, probs, att_ws = self.model.inference(
                x, self.config["inference"], spemb=spemb
            )
            logging.info(
                "inference speed = %.1f frames / sec."
                % (int(outs.size(0)) / (time.time() - start_time))
            )

            _plot_and_save(
                outs.cpu().numpy(),
                dirname + f"/outs/{idx}_out.png",
                ref=y[:olen].cpu().numpy(),
                origin="lower",
            )
            _plot_and_save(
                probs.cpu().numpy(),
                dirname + f"/probs/{idx}_prob.png",
            )
            _plot_and_save(
                att_ws.cpu().numpy(),
                dirname + f"/att_ws/{idx}_att_ws.png",
            )

            if self.vocoder is not None:
                if not os.path.exists(os.path.join(dirname, "wav")):
                    os.makedirs(os.path.join(dirname, "wav"), exist_ok=True)
                y, sr = self.vocoder.decode(outs)
                sf.write(
                    os.path.join(dirname, "wav", f"{idx}_gen.wav"),
                    y.cpu().numpy(),
                    sr,
                    "PCM_16",
                )

            if idx >= self.config["num_save_intermediate_results"]:
                break
