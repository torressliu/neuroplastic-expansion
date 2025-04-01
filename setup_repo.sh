#!/bin/bash

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install git first."
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    git init
fi

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Neuroplastic Expansion in Deep Reinforcement Learning"

# Create and switch to main branch
git branch -M main

# Ask for GitHub repository URL
echo "Please enter your GitHub repository URL:"
read repo_url

# Remove existing remote if it exists
git remote remove origin 2>/dev/null || true

# Add remote and push
git remote add origin "$repo_url"
git push -u origin main

echo "Repository has been pushed to GitHub!" 