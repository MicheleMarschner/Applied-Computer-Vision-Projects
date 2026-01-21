import wandb

run = wandb.init(project="diffusion_model_assessment_v2")  

artifact = wandb.Artifact(
    name="diffusion_models03-checkpoints_final",
    type="model",
    description="All checkpoints from the checkpoints/ folder",
)

artifact.add_dir("./checkpoints")  # uploads entire folder recursively

run.log_artifact(artifact, aliases=["latest"])
run.finish()
