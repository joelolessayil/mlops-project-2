# train.py
import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb

from data import GLUEDataModule
from model import GLUETransformer

def main(args):
    # --- Set Seed ---
    L.seed_everything(args.seed)

    # --- Setup Logger ---
    run_name = f"[{args.run_tag}]-{args.model_name_or_path.split('/')[-1]}-{args.task_name}-lr{args.learning_rate}-bs{args.train_batch_size}"
    wandb_logger = WandbLogger(project="newcrosoft-project-2", name=run_name)
    
    # wandb.login(key="YOUR_API_KEY") 

    # --- Setup Data ---
    dm = GLUEDataModule(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        max_seq_length=args.max_seq_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    dm.setup("fit") 

    # --- Setup Model ---
    model = GLUETransformer(
        model_name_or_path=args.model_name_or_path,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    # --- Setup Trainer ---
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        default_root_dir=args.checkpoint_dir, # This saves checkpoints to the specified dir
    )

    # --- Start Training ---
    print(f"Starting training run: {run_name}")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GLUETransformer model.")

    # Add arguments from the prompt
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="models/", 
        help="Directory to save model checkpoints."
    )
    parser.add_argument(
        "--lr", "--learning_rate", 
        type=float, 
        default=5e-5, 
        help="Learning rate for the optimizer.",
        dest="learning_rate"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.05,  
        help="Weight decay for the optimizer."
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="local_py", # 
        help="Tag to identify the run source (e.g., local_py, docker_build)"
    )

    # Add other hyperparameters
    parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    parser.add_argument("--task_name", type=str, default="mrpc")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    main(args)