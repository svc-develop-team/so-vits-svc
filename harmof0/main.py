import argparse
from .pitch_tracker import PitchTracker
import torch
import os

# @ex.automain
# def main(
#         audio_path = 'wav/a.mp3',
#         device = "cuda" if torch.cuda.is_available() else "cpu",
#         checkpoint_path = 'checkpoints/checkpoint_mdb-stem-synth.pth',
#         output_dir = None,
#         save_activation = True,
#         frames_per_step = 6000,
#     ):
#         pitch_tracker = PitchTracker(checkpoint_path, device=device, frames_per_step=frames_per_step)
#         pitch_tracker.pred_file(audio_path, output_dir, save_activation)

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('audio_path', type=str,)
        parser.add_argument('--output-dir', type=str, default=None)
        parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
        parser.add_argument('--save-activation', type=eval, default=True, help="Save the activation as png.")
        parser.add_argument('--frames-per-step', type=int, default=1000, help="The number of frames for a step.")
        parser.add_argument('--hop-length', type=int, default=160, help="The sample rate is 16000, so the default 160 means 10 milliseconds.")
        parser.add_argument('--post-processing', type=eval, default=True, help="use post processing.")
        parser.add_argument('--high-threshold', type=float, default=0.8, help="high threshold for post processing.")
        parser.add_argument('--low-threshold', type=float, default=0.1, help="low threshold for post processing.")
        parser.add_argument('--min-pitch-dur', type=float, default=0.1, help="min pitch duration for post processing.")
        parser.add_argument("--n-beam", type=int, default=5, help="beam number of post processing.")
        parser.add_argument('--checkpoint-path', type=str, default=None, help="The path to pretrained model weight.")

        args = parser.parse_args()

        pitch_tracker = PitchTracker(
            args.checkpoint_path, 
            hop_length=args.hop_length,
            device=args.device, 
            frames_per_step=args.frames_per_step,
            post_processing=args.post_processing,
            high_threshold=args.high_threshold,
            low_threshold=args.low_threshold,
            min_pitch_dur=args.min_pitch_dur,
            n_beam=args.n_beam,
        )
        pitch_tracker.pred_file(args.audio_path, args.output_dir, args.save_activation)
