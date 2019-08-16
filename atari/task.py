import argparse
import os
import model

if __name__ == "__main__":
    """
    Default values of arguments from: https://github.com/fg91/Deep-Q-Learning
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        required=True,
    )
    parser.add_argument(
        "--run_id",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        required=True
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0000625
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=30000000
    )
    parser.add_argument(
        "--starting_step",
        type=int,
        default=0
    )
    parser.add_argument(
        "--input_shape",
        type=tuple,
        default=(84, 84)
    )
    parser.add_argument(
        "--input_window_length",
        type=int,
        default=4
    )
    parser.add_argument(
        "--hidden_layer_size",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--replay_memory_size",
        type=int,
        default=1000000
    )
    parser.add_argument(
        "--replay_memory_starting_size",
        type=int,
        default=50000
    )
    parser.add_argument(
        "--starting_epslon",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--annealed_epslon1",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--annealed_epslon2",
        type=float,
        default=0.05
    )
    parser.add_argument(
        "--annealed_steps1",
        type=int,
        default=1000000
    )
    parser.add_argument(
        "--annealed_steps2",
        type=int,
        default=25000000
    )
    parser.add_argument(
        "--discount_factor",
        type=float,
        default=0.99
    )
    parser.add_argument(
        "--target_update_frequency",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--gradient_update_frequency",
        type=int,
        default=4
    )
    parser.add_argument(
        "--dueling",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--double_dqn",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--record_video_every",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=100000
    )
    parser.add_argument(
        "--strarting_fire_steps",
        type=int,
        default=10
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=123
    )
    args = parser.parse_args()
    args = args.__dict__

    output_dir = args["output_dir"]
    checkpoint_path = os.path.join(output_dir, "checkpoint")
    monitor_path = os.path.join(output_dir, "monitor")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(monitor_path, exist_ok=True)

    checkpoint_weights_base = os.path.join(checkpoint_path, 'checkpoint_weights')
    checkpoint_weights_filename = checkpoint_weights_base + ".h5f"
    checkpoint_step_filename = checkpoint_weights_base + "_step.txt"
    log_filename = os.path.join(output_dir, "log.txt")
    weights_filename = os.path.join(checkpoint_path, 'final_weights.h5f')

    model.train_and_evaluate(args, monitor_path, checkpoint_step_filename,
                             checkpoint_weights_filename, weights_filename, log_filename)
