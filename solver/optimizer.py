import math


def adjust_learning_rate(
    optimizer,
    curr_step: int,
    num_training_steps: int,
    cfg,
):
    num_warmup_steps: int = round(cfg.FRACTION_WARMUP_STEPS * num_training_steps)
    if cfg.SCHEDULE == "linear_with_warmup":
        if curr_step < num_warmup_steps:
            gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )
        optimizer.param_groups[0]["lr"] = cfg.LR * gamma
    elif cfg.SCHEDULE == "":  # constant LR
        gamma = 1
        optimizer.param_groups[0]["lr"] = cfg.LR * gamma
    elif cfg.SCHEDULE == 'cosine_with_warmup':
        if curr_step < num_warmup_steps:
            gamma = float(curr_step) / float(max(1, num_warmup_steps))
            optimizer.param_groups[0]["lr"] = cfg.LR * gamma
        else:
            optimizer.param_groups[0]["lr"] = cfg.LR * (1 + math.cos(math.pi * float(curr_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)))) / 2
    else:
        raise NotImplementedError
