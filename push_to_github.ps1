$ErrorActionPreference = "Stop"
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

$repoName = "KAN-KubeAgent"
$repoDir  = "d:\VS Code\rescech\KAN-KubeAgent"
$description = "Research: Trustworthy Autonomous Kubernetes Remediation using KAN Verification Layers inside LLM-based Agentic AI Systems"

Set-Location $repoDir

# Init git
git init
git config user.email "avanishkasar@research.local"
git config user.name  "Avanish Kasar"

# Stage everything
git add .
git commit -m "Initial research structure: KAN + Kubernetes + Agentic AI

- README with full system architecture and novelty claims
- Literature review: KAN original paper (arXiv:2404.19756)
- Literature review: KubeIntellect (arXiv:2509.02449)
- Formal research gap and novelty claims document
- Full methodology draft with KAN trust layer design
- Dataset documentation (K-RAD specification)
- Prioritised reading list with 15 papers
- BibTeX bibliography
- Weekly research log with setup commands"

# Create GitHub repo and push
gh repo create $repoName `
  --description $description `
  --public `
  --source . `
  --remote origin `
  --push

Write-Host ""
Write-Host "✅ Repository created and pushed!"
Write-Host "🔗 Visit: https://github.com/avanishkasar/$repoName"
