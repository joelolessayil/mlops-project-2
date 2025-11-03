# 1. Start from a stable, slim Python base image
FROM python:3.11-slim

# 2. Set a working directory inside the container
WORKDIR /app

# 3. Copy *only* the requirements file first
COPY requirements.txt .

# 4. Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project code into the /app directory
COPY . .

# 6. Set the command to run when the container starts
# Replace the arguments below with your best hyperparameters from Project 1
CMD ["python", "train.py", \
     "--max_epochs", "3", \
     "--learning_rate", "5e-5", \
     "--train_batch_size", "32", \
     "--weight_decay", "0.05", \
     "--warmup_steps", "0", \
     "--checkpoint_dir", "/app/models" \
    ]