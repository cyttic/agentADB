# CI/CD Setup Guide

Pipeline: `push to main` → lint → build Docker → push to DockerHub → deploy to Azure VM

---

## 1. DockerHub

1. Create an account at https://hub.docker.com
2. Create an **Access Token**: Account Settings → Security → New Access Token
3. Your image will be: `<your-username>/db-assistant-framework`

---

## 2. Azure VM

### Create the VM (Azure Portal or CLI)

```bash
az vm create \
  --resource-group db-assistant-rg \
  --name db-assistant-vm \
  --image Ubuntu2204 \
  --size Standard_B1ms \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# Open port 80
az vm open-port --port 80 --resource-group db-assistant-rg --name db-assistant-vm
```

### Install Docker on the VM

```bash
ssh azureuser@<VM_PUBLIC_IP>

curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker azureuser
# logout and re-login for group to take effect
```

### Create config.json on the VM

```bash
sudo mkdir -p /etc/db-assistant
sudo nano /etc/db-assistant/config.json
```

Paste your `config.json` (without API keys — those come from GitHub secrets):

```json
{
  "default_provider": "openai",
  "default_model": "gpt-4o",
  "local_server": { "host": "127.0.0.1", "port": 9001, "n_predict": 2048 },
  "ollama":       { "host": "127.0.0.1", "port": 11434 },
  "server":       { "host": "0.0.0.0",   "port": 8000 }
}
```

---

## 3. GitHub Secrets

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

| Secret name          | Value |
|----------------------|-------|
| `DOCKERHUB_USERNAME` | Your DockerHub username |
| `DOCKERHUB_TOKEN`    | DockerHub access token |
| `AZURE_VM_HOST`      | VM public IP address |
| `AZURE_VM_USER`      | `azureuser` (or your username) |
| `AZURE_VM_SSH_KEY`   | Private SSH key (contents of `~/.ssh/id_rsa`) |
| `OPENAI_API_KEY`     | Your OpenAI API key |
| `ANTHROPIC_API_KEY`  | Your Anthropic API key (optional) |

### How to get the SSH private key

```bash
# On your local machine (if you used --generate-ssh-keys above):
cat ~/.ssh/id_rsa
# Copy the entire output including -----BEGIN----- and -----END----- lines
# Paste it as the AZURE_VM_SSH_KEY secret
```

---

## 4. Pipeline Flow

```
push to main
    │
    ▼
[test]
  • ruff lint
  • syntax check all .py files
    │
    ▼ (only if tests pass)
[build-and-push]
  • docker buildx build
  • tag: latest + short SHA
  • push to DockerHub
    │
    ▼ (only if build succeeds)
[deploy]
  • SSH into Azure VM
  • docker pull latest
  • stop old container
  • docker run with env vars + config mount
  • health check: GET /health
  • prune old images
```

---

## 5. Manual deploy (without CI)

```bash
# Build locally
docker build -t db-assistant-framework .

# Run locally
docker run -d \
  --name db-assistant \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/config.json:/app/config.json:ro \
  db-assistant-framework

# Test
curl http://localhost:8000/health
```

---

## 6. Trigger the pipeline

```bash
git add .
git commit -m "deploy: initial CI/CD setup"
git push origin main
```

Then watch it run at: `https://github.com/<you>/agentADB/actions`
